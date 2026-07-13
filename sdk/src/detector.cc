#include "detector.h"

#include "MNN/ImageProcess.hpp"
#include "MNN/expr/Executor.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/Module.hpp"
#include "cv/cv.hpp"

// ===========================================================================
// Detector::Impl
// ===========================================================================

struct Detector::Impl {
  std::shared_ptr<MNN::Express::Module> net;
  std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr;
  std::map<int, std::string> class_names;
  static constexpr int TARGET_SIZE = 640;

  std::vector<DetectionResult> process_image(const MNN::Express::VARP& image);
};

std::vector<DetectionResult> Detector::Impl::process_image(const MNN::Express::VARP& original_image) {
  auto dims = original_image->getInfo()->dim;
  int ih = dims[0];
  int iw = dims[1];
  int len = ih > iw ? ih : iw;
  float scale = len / static_cast<float>(TARGET_SIZE);

  // Pad to square
  std::vector<int> padvals{0, len - ih, 0, len - iw, 0, 0};
  auto pads = MNN::Express::_Const(static_cast<void*>(padvals.data()), {3, 2}, MNN::Express::NCHW, halide_type_of<int>());
  auto image = MNN::Express::_Pad(original_image, pads, MNN::Express::CONSTANT);

  // Resize + Normalize
  image = MNN::CV::resize(image, MNN::CV::Size(TARGET_SIZE, TARGET_SIZE), 0, 0, MNN::CV::INTER_LINEAR, -1, {0., 0., 0.},
                          {1. / 255., 1. / 255., 1. / 255.});

  auto input = MNN::Express::_Unsqueeze(image, {0});
  input = MNN::Express::_Convert(input, MNN::Express::NCHW);

  auto outputs = net->onForward({input});
  auto output = MNN::Express::_Convert(outputs[0], MNN::Express::NCHW);
  output = MNN::Express::_Transpose(MNN::Express::_Squeeze(output), {1, 0});

  auto x0 = MNN::Express::_Gather(output, MNN::Express::_Scalar<int>(0));
  auto y0 = MNN::Express::_Gather(output, MNN::Express::_Scalar<int>(1));
  auto x1 = MNN::Express::_Gather(output, MNN::Express::_Scalar<int>(2));
  auto y1 = MNN::Express::_Gather(output, MNN::Express::_Scalar<int>(3));
  auto scores = MNN::Express::_Gather(output, MNN::Express::_Scalar<int>(4));
  auto ids = MNN::Express::_Gather(output, MNN::Express::_Scalar<int>(5));
  auto boxes = MNN::Express::_Stack({x0, y0, x1, y1}, 1);

  auto result_ids = MNN::Express::_Nms(boxes, scores, 100, 0.8f, 0.1f);

  auto result_ptr = result_ids->readMap<int>();
  auto box_ptr = boxes->readMap<float>();
  auto ids_ptr = ids->readMap<float>();
  auto score_ptr = scores->readMap<float>();

  std::vector<DetectionResult> result;
  for (int i = 0; i < 100; ++i) {
    auto idx = result_ptr[i];
    if (idx < 0) break;
    int class_id = static_cast<int>(ids_ptr[idx]);
    auto it = class_names.find(class_id);
    std::string class_name = it != class_names.end() ? it->second : "unknown_" + std::to_string(class_id);
    int x0 = static_cast<int>(box_ptr[idx * 4] * scale);
    int y0 = static_cast<int>(box_ptr[idx * 4 + 1] * scale);
    int x1 = static_cast<int>(box_ptr[idx * 4 + 2] * scale);
    int y1 = static_cast<int>(box_ptr[idx * 4 + 3] * scale);
    result.push_back({x0, y0, x1, y1, class_id, class_name, score_ptr[idx]});
  }
  return result;
}

// ===========================================================================
// Detector (public API)
// ===========================================================================

Detector::Detector(const DetectorConfig& config) : pimpl_(std::make_unique<Impl>()) {
  pimpl_->class_names = config.class_names;

  MNN::ScheduleConfig scheduleConfig;
  scheduleConfig.type = static_cast<MNNForwardType>(config.forward_type);
  scheduleConfig.numThread = config.num_threads;

  MNN::BackendConfig backendConfig;
  backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(config.precision_mode);
  backendConfig.memory = static_cast<MNN::BackendConfig::MemoryMode>(config.memory_mode);
  scheduleConfig.backendConfig = &backendConfig;

  MNN::Express::Module::Config moduleConfig;
  moduleConfig.shapeMutable = false;

  pimpl_->rtmgr = std::shared_ptr<MNN::Express::Executor::RuntimeManager>(
      MNN::Express::Executor::RuntimeManager::createRuntimeManager(scheduleConfig));
  if (!pimpl_->rtmgr) {
    MNN_ERROR("Empty RuntimeManger\n");
    return;
  }
  pimpl_->rtmgr->setCache(".det_cache");

  pimpl_->net = std::shared_ptr<MNN::Express::Module>(
      MNN::Express::Module::load({}, {}, config.model_path.c_str(), pimpl_->rtmgr, &moduleConfig));
  if (!pimpl_->net) {
    MNN_ERROR("Failed to load model: %s\n", config.model_path.c_str());
    return;
  }

  // Warmup
  if (config.warmup > 0) {
    MNN_PRINT("Starting Warmup...\n");
    auto dummy_input = MNN::Express::_Const(0.0f, {1, 3, Impl::TARGET_SIZE, Impl::TARGET_SIZE}, MNN::Express::NCHW);
    for (int i = 0; i < config.warmup; ++i) {
      pimpl_->net->onForward({dummy_input});
    }
    MNN_PRINT("Warmup finished.\n");
  }
}

Detector::~Detector() {
  if (pimpl_ && pimpl_->rtmgr) {
    pimpl_->rtmgr->updateCache();
  }
}

Detector::Detector(Detector&&) noexcept = default;
Detector& Detector::operator=(Detector&&) noexcept = default;

std::vector<DetectionResult> Detector::run(const std::string& image_path) {
  return pimpl_->process_image(MNN::CV::imread(image_path));
}
