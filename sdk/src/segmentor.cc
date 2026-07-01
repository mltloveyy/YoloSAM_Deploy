#include "segmentor.h"

#include "MNN/ImageProcess.hpp"
#include "MNN/expr/Executor.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/Module.hpp"
#include "cv/cv.hpp"

// ===========================================================================
// Segmentor::Impl
// ===========================================================================
struct Segmentor::Impl {
  std::shared_ptr<MNN::Express::Module> enc;
  std::shared_ptr<MNN::Express::Module> dec;
  std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr;
  int forward_type;
  int precision_mode;
  int num_threads;
  MNN::Express::VARP image_embed;
  MNN::Express::VARP high_feats0;
  MNN::Express::VARP high_feats1;
  MNN::Express::VARP original_image;
  static constexpr int TARGET_SIZE = 1024;
  int ih, iw, new_h, new_w;
  float scale;

  void encode(const MNN::Express::VARP& image);
  SegmentResult decode(const std::vector<int>& point_coords, const std::vector<int>& point_labels);
};

void Segmentor::Impl::encode(const MNN::Express::VARP& original_image) {
  this->original_image = original_image;
  auto dims = original_image->getInfo()->dim;
  this->ih = dims[0];
  this->iw = dims[1];
  this->scale = 1.0 * TARGET_SIZE / (this->ih > this->iw ? this->ih : this->iw);

  // Resize + Normalize
  this->new_h = static_cast<int>(this->ih * this->scale);
  this->new_w = static_cast<int>(this->iw * this->scale);
  auto image = MNN::CV::resize(original_image, MNN::CV::Size(this->new_w, this->new_h), 0, 0, MNN::CV::INTER_LINEAR, -1,
                               {123.675, 116.28, 103.53}, {1 / 58.395, 1 / 57.12, 1 / 57.375});

  // Pad to square
  std::vector<int> padvals{0, TARGET_SIZE - this->new_h, 0, TARGET_SIZE - this->new_w, 0, 0};
  auto pads = MNN::Express::_Const(static_cast<void*>(padvals.data()), {3, 2}, MNN::Express::NCHW, halide_type_of<int>());
  image = MNN::Express::_Pad(image, pads, MNN::Express::CONSTANT);

  // Forward
  image = MNN::Express::_Unsqueeze(image, {0});
  image = MNN::Express::_Convert(image, MNN::Express::NCHW);
  auto outputs = this->enc->onForward({image});
  this->image_embed = MNN::Express::_Convert(outputs[0], MNN::Express::NCHW);
  this->high_feats0 = MNN::Express::_Convert(outputs[1], MNN::Express::NCHW);
  this->high_feats1 = MNN::Express::_Convert(outputs[2], MNN::Express::NCHW);
}

SegmentResult Segmentor::Impl::decode(const std::vector<int>& point_coords, const std::vector<int>& point_labels) {
  // Prompt preprocess
  int num_coords = point_coords.size(), num_labels = point_labels.size();
  int num_points = num_labels < num_coords / 2 ? num_labels : num_coords / 2;
  std::vector<float> scale_coords, scale_labels;
  for (auto i = 0; i < num_points; i++) {
    scale_coords.push_back(point_coords[2 * i] * this->scale);
    scale_coords.push_back(point_coords[2 * i + 1] * this->scale);
    scale_labels.push_back(point_labels[i]);
  }
  scale_coords.push_back(0);
  scale_labels.push_back(0);
  auto coords = MNN::Express::_Const(static_cast<void*>(scale_coords.data()), {1, num_points, 2}, MNN::Express::NCHW);
  auto labels = MNN::Express::_Const(static_cast<void*>(scale_labels.data()), {1, num_points}, MNN::Express::NCHW);

  // Forward
  std::vector<MNN::Express::VARP> inputs{coords, labels, this->image_embed, this->high_feats0, this->high_feats1};
  auto outputs = this->dec->onForward(inputs);
  auto mask = MNN::Express::_Convert(outputs[0], MNN::Express::NCHW);

  // Postprocess(squeeze -> slice -> greater -> cast)
  mask = MNN::Express::_Squeeze(mask);
  int slice_starts[] = {0, 0}, slice_sizes[] = {this->new_h, this->new_w};
  mask = MNN::Express::_Slice(mask, MNN::Express::_Const(slice_starts, {2}, MNN::Express::NCHW),
                              MNN::Express::_Const(slice_sizes, {2}, MNN::Express::NCHW));
  mask = MNN::Express::_Greater(mask, MNN::Express::_Const(0.0, {1}, MNN::Express::NCHW));
  mask = MNN::Express::_Cast<uint8_t>(mask);

  // FindContour
  mask = MNN::CV::resize(mask, MNN::CV::Size(this->iw, this->ih), 0, 0, MNN::CV::INTER_NEAREST);
  auto contours = MNN::CV::findContours(mask, MNN::CV::RetrievalModes::RETR_EXTERNAL,
                                        MNN::CV::ContourApproximationModes::CHAIN_APPROX_SIMPLE);
  auto contour = contours[0];
  auto rect = MNN::CV::boundingRect(contour);
  auto ptr = contour->readMap<int>();
  auto len = contour->getInfo()->size;

  SegmentResult result;
  result.x0 = rect.tl().x;
  result.y0 = rect.tl().y;
  result.x1 = rect.br().x;
  result.y1 = rect.br().y;
  result.points.reserve(len);
  for (int i = 0; i < len; ++i) {
    result.points.push_back(ptr[i]);
  }
  return result;
}

// ===========================================================================
// Segmentor (public API)
// ===========================================================================

Segmentor::Segmentor(const std::string& enc_path, const std::string& dec_path, int forward_type, int precision_mode,
                     int num_threads, bool warmup)
    : pimpl_(std::make_unique<Impl>()) {
  pimpl_->forward_type = forward_type;
  pimpl_->precision_mode = precision_mode;
  pimpl_->num_threads = num_threads;

  MNN::ScheduleConfig scheduleConfig;
  scheduleConfig.type = static_cast<MNNForwardType>(forward_type);
  scheduleConfig.numThread = num_threads;

  MNN::BackendConfig backendConfig;
  backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision_mode);
  scheduleConfig.backendConfig = &backendConfig;

  pimpl_->rtmgr = std::shared_ptr<MNN::Express::Executor::RuntimeManager>(
      MNN::Express::Executor::RuntimeManager::createRuntimeManager(scheduleConfig));
  if (!pimpl_->rtmgr) {
    MNN_ERROR("Empty RuntimeManger\n");
    return;
  }

  pimpl_->enc = std::shared_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, enc_path.c_str(), pimpl_->rtmgr));
  if (!pimpl_->enc) {
    MNN_ERROR("Failed to load encoder: %s\n", enc_path.c_str());
    return;
  }

  pimpl_->dec = std::shared_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, dec_path.c_str(), pimpl_->rtmgr));
  if (!pimpl_->dec) {
    MNN_ERROR("Failed to load decoder: %s\n", dec_path.c_str());
    return;
  }

  // Warmup
  if (warmup) {
    MNN_PRINT("Starting Warmup...\n");
    auto dummy_input = MNN::Express::_Const(0.0f, {1, 3, Impl::TARGET_SIZE, Impl::TARGET_SIZE}, MNN::Express::NCHW);
    for (int i = 0; i < 3; ++i) {
      pimpl_->enc->onForward({dummy_input});
    }
    MNN_PRINT("Warmup finished.\n");
  }
}

Segmentor::~Segmentor() {
  if (pimpl_ && pimpl_->rtmgr) {
    pimpl_->rtmgr->updateCache();
  }
}

Segmentor::Segmentor(Segmentor&&) noexcept = default;
Segmentor& Segmentor::operator=(Segmentor&&) noexcept = default;

void Segmentor::set_image(const std::string& image_path) { pimpl_->encode(MNN::CV::imread(image_path)); }

SegmentResult Segmentor::forward(const std::vector<int>& point_coords, const std::vector<int>& point_labels) {
  return pimpl_->decode(point_coords, point_labels);
}