#include "detector.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "MNN/ImageProcess.hpp"
#include "MNN/expr/Executor.hpp"
#include "MNN/expr/Module.hpp"
#include "cv/cv.hpp"
#include "json/json.h"
#include "yaml-cpp/yaml.h"

namespace fs = std::filesystem;
namespace MNN_Express = MNN::Express;

// ---------------------------------------------------------------------------
// Internal types (not exported)
// ---------------------------------------------------------------------------
namespace {

struct LabelMeShape {
  std::string label;
  std::vector<std::vector<double>> points;
  std::string shape_type = "rectangle";
  int flags = 0;
};

}  // anonymous namespace

// ===========================================================================
// Detector::Impl
// ===========================================================================

struct Detector::Impl {
  std::shared_ptr<MNN_Express::Module> net;
  std::shared_ptr<MNN_Express::Executor::RuntimeManager> rtmgr;
  int forward_type;
  int precision_mode;
  int num_threads;

  static constexpr int TARGET_SIZE = 640;

  std::vector<DetectionResult> process_image(const MNN_Express::VARP& image);
};

std::vector<DetectionResult> Detector::Impl::process_image(const MNN_Express::VARP& original_image) {
  auto dims = original_image->getInfo()->dim;
  int ih = dims[0];
  int iw = dims[1];
  int len = std::max(ih, iw);
  float scale = len / static_cast<float>(TARGET_SIZE);

  // Pad to square
  std::vector<int> padvals{0, len - ih, 0, len - iw, 0, 0};
  auto pads = MNN_Express::_Const(static_cast<void*>(padvals.data()), {3, 2}, MNN_Express::NCHW, halide_type_of<int>());
  auto image = MNN_Express::_Pad(original_image, pads, MNN_Express::CONSTANT);

  // Resize + Normalize
  image = MNN::CV::resize(image, MNN::CV::Size(TARGET_SIZE, TARGET_SIZE), 0, 0, MNN::CV::INTER_LINEAR, -1, {0., 0., 0.},
                          {1. / 255., 1. / 255., 1. / 255.});

  auto input = MNN_Express::_Unsqueeze(image, {0});
  input = MNN_Express::_Convert(input, MNN_Express::NC4HW4);

  auto outputs = net->onForward({input});
  auto output = MNN_Express::_Convert(outputs[0], MNN_Express::NCHW);
  output = MNN_Express::_Transpose(MNN_Express::_Squeeze(output), {1, 0});

  auto x0 = MNN_Express::_Gather(output, MNN_Express::_Scalar<int>(0));
  auto y0 = MNN_Express::_Gather(output, MNN_Express::_Scalar<int>(1));
  auto x1 = MNN_Express::_Gather(output, MNN_Express::_Scalar<int>(2));
  auto y1 = MNN_Express::_Gather(output, MNN_Express::_Scalar<int>(3));
  auto scores = MNN_Express::_Gather(output, MNN_Express::_Scalar<int>(4));
  auto ids = MNN_Express::_Gather(output, MNN_Express::_Scalar<int>(5));
  auto boxes = MNN_Express::_Stack({x0, y0, x1, y1}, 1);

  auto result_ids = MNN_Express::_Nms(boxes, scores, 100, 0.8f, 0.25f);

  auto result_ptr = result_ids->readMap<int>();
  auto box_ptr = boxes->readMap<float>();
  auto ids_ptr = ids->readMap<int>();
  auto score_ptr = scores->readMap<float>();

  std::vector<DetectionResult> results;
  for (int i = 0; i < 100; ++i) {
    auto idx = result_ptr[i];
    if (idx < 0) break;

    results.push_back({box_ptr[idx * 4 + 0] * scale, box_ptr[idx * 4 + 1] * scale, box_ptr[idx * 4 + 2] * scale,
                       box_ptr[idx * 4 + 3] * scale, ids_ptr[idx], score_ptr[idx]});
  }
  return results;
}

// ===========================================================================
// Detector (public API)
// ===========================================================================

Detector::Detector(const std::string& model_path, int forward_type, int precision_mode, int num_threads)
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

  pimpl_->rtmgr = std::shared_ptr<MNN_Express::Executor::RuntimeManager>(
      MNN_Express::Executor::RuntimeManager::createRuntimeManager(scheduleConfig));
  if (!pimpl_->rtmgr) {
    MNN_ERROR("Empty RuntimeManger\n");
    return;
  }
  pimpl_->rtmgr->setCache(".cachefile");

  pimpl_->net = std::shared_ptr<MNN_Express::Module>(MNN_Express::Module::load({}, {}, model_path.c_str(), pimpl_->rtmgr));
  if (!pimpl_->net) {
    MNN_ERROR("Failed to load model: %s\n", model_path.c_str());
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

std::map<std::string, std::vector<DetectionResult>> Detector::run_directory(const std::string& input_dir) {
  std::map<std::string, std::vector<DetectionResult>> results;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (!fs::is_regular_file(entry)) continue;
    const auto ext = entry.path().extension().string();
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
      results[entry.path().filename().string()] = run(entry.path().string());
    }
  }
  return results;
}

// ===========================================================================
// ResultProcessor::Impl
// ===========================================================================

struct ResultProcessor::Impl {
  std::map<int, std::string> class_names;

  std::vector<LabelMeShape> to_shapes(const std::vector<DetectionResult>& detections) const {
    std::vector<LabelMeShape> shapes;
    shapes.reserve(detections.size());
    for (const auto& d : detections) {
      LabelMeShape s;
      auto it = class_names.find(d.class_id);
      s.label = it != class_names.end() ? it->second : "unknown_" + std::to_string(d.class_id);
      s.points = {{d.x0, d.y0}, {d.x1, d.y1}};
      shapes.push_back(std::move(s));
    }
    return shapes;
  }
};

// ===========================================================================
// ResultProcessor (public API)
// ===========================================================================

ResultProcessor::ResultProcessor(const std::string& config_path) : pimpl_(std::make_unique<Impl>()) {
  YAML::Node config = YAML::LoadFile(config_path);
  if (config["names"]) {
    for (auto it = config["names"].begin(); it != config["names"].end(); ++it) {
      pimpl_->class_names[it->first.as<int>()] = it->second.as<std::string>();
    }
  } else {
    std::cerr << "Error: 'names' key not found in " << config_path << std::endl;
  }
}

ResultProcessor::~ResultProcessor() = default;
ResultProcessor::ResultProcessor(ResultProcessor&&) noexcept = default;
ResultProcessor& ResultProcessor::operator=(ResultProcessor&&) noexcept = default;

void ResultProcessor::draw_and_save_image(const std::string& image_path, const std::vector<DetectionResult>& detections,
                                          const std::string& output_path) {
  auto image = MNN::CV::imread(image_path);
  for (const auto& d : detections) {
    MNN::CV::rectangle(image, {d.x0, d.y0}, {d.x1, d.y1}, {0, 0, 255}, 2);
  }
  MNN::CV::imwrite(output_path, image);
}

void ResultProcessor::save_labelme_json(const std::string& image_path, const std::vector<DetectionResult>& detections,
                                        const std::string& json_output_path) {
  Json::Value root;
  root["version"] = "4.5.6";
  root["flags"] = Json::objectValue;
  root["shapes"] = Json::arrayValue;

  for (const auto& s : pimpl_->to_shapes(detections)) {
    Json::Value obj;
    obj["label"] = s.label;
    obj["points"] = Json::arrayValue;
    for (const auto& pt : s.points) {
      Json::Value p(Json::arrayValue);
      p.append(pt[0]);
      p.append(pt[1]);
      obj["points"].append(std::move(p));
    }
    obj["shape_type"] = s.shape_type;
    obj["flags"] = s.flags;
    root["shapes"].append(std::move(obj));
  }

  auto img = MNN::CV::imread(image_path);
  auto dims = img->getInfo()->dim;
  root["imageHeight"] = dims[0];
  root["imageWidth"] = dims[1];

  std::ofstream file(json_output_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << json_output_path << std::endl;
    return;
  }
  Json::StreamWriterBuilder builder;
  builder.settings_["indentation"] = "  ";
  std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
  writer->write(root, &file);
}
