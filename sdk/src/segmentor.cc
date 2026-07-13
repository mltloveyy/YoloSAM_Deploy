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
  std::shared_ptr<MNN::Express::Executor::RuntimeManager> enc_rtmgr;
  std::shared_ptr<MNN::Express::Executor::RuntimeManager> dec_rtmgr;
  MNN::Express::VARP image_embed;
  MNN::Express::VARP high_feats0;
  MNN::Express::VARP high_feats1;
  int enc_forward_type, dec_forward_type;
  static constexpr int TARGET_SIZE = 1024;
  int ih, iw, new_h, new_w;
  float scale;

  void encode(const MNN::Express::VARP& image);
  SegmentationResult decode(const std::vector<int>& point_coords, const std::vector<int>& point_labels);
};

void Segmentor::Impl::encode(const MNN::Express::VARP& original_image) {
  auto dims = original_image->getInfo()->dim;
  ih = dims[0];
  iw = dims[1];
  scale = 1.0 * TARGET_SIZE / (ih > iw ? ih : iw);

  // Resize + Normalize
  new_h = static_cast<int>(ih * scale);
  new_w = static_cast<int>(iw * scale);
  auto image = MNN::CV::resize(original_image, MNN::CV::Size(new_w, new_h), 0, 0, MNN::CV::INTER_LINEAR, -1,
                               {123.675, 116.28, 103.53}, {1 / 58.395, 1 / 57.12, 1 / 57.375});

  // Pad to square
  std::vector<int> padvals{0, TARGET_SIZE - new_h, 0, TARGET_SIZE - new_w, 0, 0};
  auto pads = MNN::Express::_Const(static_cast<void*>(padvals.data()), {3, 2}, MNN::Express::NCHW, halide_type_of<int>());
  image = MNN::Express::_Pad(image, pads, MNN::Express::CONSTANT);

  // Forward
  image = MNN::Express::_Unsqueeze(image, {0});
  image = MNN::Express::_Convert(image, MNN::Express::NCHW);
  auto outputs = enc->onForward({image});

  if (enc_forward_type != 0 && dec_forward_type == 0) {
    // Features must be copied from GPU to CPU, as encoder run on GPU and decoder on CPU
    image_embed = MNN::Express::_Clone(MNN::Express::_Convert(outputs[0], MNN::Express::NCHW), true);
    high_feats0 = MNN::Express::_Clone(MNN::Express::_Convert(outputs[1], MNN::Express::NCHW), true);
    high_feats1 = MNN::Express::_Clone(MNN::Express::_Convert(outputs[2], MNN::Express::NCHW), true);
  } else {
    image_embed = MNN::Express::_Convert(outputs[0], MNN::Express::NCHW);
    high_feats0 = MNN::Express::_Convert(outputs[1], MNN::Express::NCHW);
    high_feats1 = MNN::Express::_Convert(outputs[2], MNN::Express::NCHW);
  }
}

SegmentationResult Segmentor::Impl::decode(const std::vector<int>& point_coords, const std::vector<int>& point_labels) {
  // Prompt preprocess
  int num_coords = point_coords.size(), num_labels = point_labels.size();
  int num_points = num_labels < num_coords / 2 ? num_labels : num_coords / 2;
  std::vector<float> scale_coords, scale_labels;
  for (auto i = 0; i < num_points; i++) {
    scale_coords.push_back(point_coords[2 * i] * scale);
    scale_coords.push_back(point_coords[2 * i + 1] * scale);
    scale_labels.push_back(point_labels[i]);
  }
  scale_coords.push_back(0);
  scale_labels.push_back(0);
  auto coords = MNN::Express::_Const(static_cast<void*>(scale_coords.data()), {1, num_points, 2}, MNN::Express::NCHW);
  auto labels = MNN::Express::_Const(static_cast<void*>(scale_labels.data()), {1, num_points}, MNN::Express::NCHW);

  // Forward
  std::vector<MNN::Express::VARP> inputs{coords, labels, image_embed, high_feats0, high_feats1};
  auto outputs = dec->onForward(inputs);
  auto mask = MNN::Express::_Convert(outputs[0], MNN::Express::NCHW);

  // Postprocess(squeeze -> slice -> greater -> cast)
  mask = MNN::Express::_Squeeze(mask);
  int slice_starts[] = {0, 0}, slice_sizes[] = {new_h, new_w};
  mask = MNN::Express::_Slice(mask, MNN::Express::_Const(slice_starts, {2}, MNN::Express::NCHW),
                              MNN::Express::_Const(slice_sizes, {2}, MNN::Express::NCHW));
  mask = MNN::Express::_Greater(mask, MNN::Express::_Const(0.0, {1}, MNN::Express::NCHW));
  mask = MNN::Express::_Cast<uint8_t>(mask);

  // Find contour
  mask = MNN::CV::resize(mask, MNN::CV::Size(iw, ih), 0, 0, MNN::CV::INTER_NEAREST);
  auto contours = MNN::CV::findContours(mask, MNN::CV::RetrievalModes::RETR_EXTERNAL,
                                        MNN::CV::ContourApproximationModes::CHAIN_APPROX_SIMPLE);
  auto contour = contours[0];
  auto rect = MNN::CV::boundingRect(contour);
  auto ptr = contour->readMap<int>();
  auto len = contour->getInfo()->size;

  SegmentationResult result;
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

Segmentor::Segmentor(const SegmentorConfig& enc_config, const SegmentorConfig& dec_config) : pimpl_(std::make_unique<Impl>()) {
  // Encoder init
  pimpl_->enc_forward_type = enc_config.forward_type;
  MNN::ScheduleConfig encScheduleConfig;
  encScheduleConfig.type = static_cast<MNNForwardType>(enc_config.forward_type);
  encScheduleConfig.numThread = enc_config.num_threads;

  MNN::BackendConfig encBackendConfig;
  encBackendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(enc_config.precision_mode);
  encBackendConfig.memory = static_cast<MNN::BackendConfig::MemoryMode>(enc_config.memory_mode);
  encScheduleConfig.backendConfig = &encBackendConfig;

  MNN::Express::Module::Config encModuleConfig;
  encModuleConfig.shapeMutable = false;

  pimpl_->enc_rtmgr = std::shared_ptr<MNN::Express::Executor::RuntimeManager>(
      MNN::Express::Executor::RuntimeManager::createRuntimeManager(encScheduleConfig));
  if (!pimpl_->enc_rtmgr) {
    MNN_ERROR("Empty RuntimeManger\n");
    return;
  }
  pimpl_->enc_rtmgr->setCache(".seg_enc_cache");

  pimpl_->enc = std::shared_ptr<MNN::Express::Module>(
      MNN::Express::Module::load({}, {}, enc_config.model_path.c_str(), pimpl_->enc_rtmgr, &encModuleConfig));
  if (!pimpl_->enc) {
    MNN_ERROR("Failed to load encoder: %s\n", enc_config.model_path.c_str());
    return;
  }

  if (enc_config.warmup > 0) {
    MNN_PRINT("Starting warmup encoder...\n");
    auto dummy_input = MNN::Express::_Const(0.0f, {1, 3, Impl::TARGET_SIZE, Impl::TARGET_SIZE}, MNN::Express::NCHW);
    for (int i = 0; i < enc_config.warmup; ++i) {
      pimpl_->enc->onForward({dummy_input});
    }
    MNN_PRINT("Encoder warmup finished.\n");
  }

  // Decoder init
  pimpl_->dec_forward_type = dec_config.forward_type;
  MNN::ScheduleConfig decScheduleConfig;
  decScheduleConfig.type = static_cast<MNNForwardType>(dec_config.forward_type);
  decScheduleConfig.numThread = dec_config.num_threads;

  MNN::BackendConfig decBackendConfig;
  decBackendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(dec_config.precision_mode);
  decBackendConfig.memory = static_cast<MNN::BackendConfig::MemoryMode>(dec_config.memory_mode);
  decScheduleConfig.backendConfig = &decBackendConfig;

  pimpl_->dec_rtmgr = std::shared_ptr<MNN::Express::Executor::RuntimeManager>(
      MNN::Express::Executor::RuntimeManager::createRuntimeManager(decScheduleConfig));
  if (!pimpl_->dec_rtmgr) {
    MNN_ERROR("Empty RuntimeManger\n");
    return;
  }
  pimpl_->dec_rtmgr->setCache(".seg_dec_cache");

  pimpl_->dec = std::shared_ptr<MNN::Express::Module>(
      MNN::Express::Module::load({}, {}, dec_config.model_path.c_str(), pimpl_->dec_rtmgr));
  if (!pimpl_->dec) {
    MNN_ERROR("Failed to load decoder: %s\n", dec_config.model_path.c_str());
    return;
  }
}

Segmentor::~Segmentor() {
  if (pimpl_) {
    if (pimpl_->enc_rtmgr) {
      pimpl_->enc_rtmgr->updateCache();
    }
    if (pimpl_->dec_rtmgr) {
      pimpl_->dec_rtmgr->updateCache();
    }
  }
}

Segmentor::Segmentor(Segmentor&&) noexcept = default;
Segmentor& Segmentor::operator=(Segmentor&&) noexcept = default;

void Segmentor::set_image(const std::string& image_path) { pimpl_->encode(MNN::CV::imread(image_path)); }

SegmentationResult Segmentor::forward(const std::vector<int>& point_coords, const std::vector<int>& point_labels) {
  return pimpl_->decode(point_coords, point_labels);
}