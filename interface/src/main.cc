#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "MNN/ImageProcess.hpp"
#include "cv/cv.hpp"
#include "detector.h"
#include "json/json.h"
#include "yaml-cpp/yaml.h"

namespace fs = std::filesystem;

std::map<int, std::string> ParseClassNames(const std::string& config_path) {
  YAML::Node config = YAML::LoadFile(config_path);
  std::map<int, std::string> class_names;
  if (config["names"]) {
    for (auto it = config["names"].begin(); it != config["names"].end(); ++it) {
      class_names[it->first.as<int>()] = it->second.as<std::string>();
    }
  } else {
    std::cerr << "Error: 'names' key not found in " << config_path << std::endl;
  }
  return class_names;
}

void draw_and_save_image(const std::string& image_path, const std::vector<DetectionResult>& result,
                         const std::string& output_path) {
  auto image = MNN::CV::imread(image_path);
  for (const auto& d : result) {
    MNN::CV::rectangle(image, {d.x0, d.y0}, {d.x1, d.y1}, {0, 0, 255}, 2);
  }
  MNN::CV::imwrite(output_path, image);
}

void save_labelme_json(const std::string& image_path, const std::vector<DetectionResult>& result,
                       const std::string& json_path) {
  Json::Value root;
  root["version"] = "4.5.6";
  root["flags"] = Json::objectValue;
  root["shapes"] = Json::arrayValue;
  for (const auto& d : result) {
    Json::Value shape(Json::objectValue);
    shape["label"] = d.class_name;
    shape["shape_type"] = "rectangle";
    Json::Value points(Json::arrayValue);
    Json::Value p0(Json::arrayValue);
    p0.append(d.x0);
    p0.append(d.y0);
    points.append(p0);
    Json::Value p1(Json::arrayValue);
    p1.append(d.x1);
    p1.append(d.y1);
    points.append(p1);
    shape["points"] = points;
    shape["flags"] = Json::objectValue;
    root["shapes"].append(shape);
  }

  auto image = MNN::CV::imread(image_path);
  auto dims = image->getInfo()->dim;
  root["imageHeight"] = dims[0];
  root["imageWidth"] = dims[1];

  std::ofstream file(json_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << json_path << std::endl;
    return;
  }
  Json::StreamWriterBuilder builder;
  builder.settings_["indentation"] = "  ";
  std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
  writer->write(root, &file);
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <model_path> <input_image_or_dir> <config_path>\n";
    return -1;
  }

  std::string model_path = argv[1];
  std::string input_path = argv[2];
  std::string config_path = argv[3];

  Detector detector(model_path, ParseClassNames(config_path));

  if (fs::is_directory(input_path)) {
    for (const auto& entry : fs::directory_iterator(input_path)) {
      if (!fs::is_regular_file(entry)) continue;
      const auto ext = entry.path().extension().string();
      if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

      auto full_path = entry.path().string();
      auto filename = entry.path().filename().string();
      auto start = std::chrono::high_resolution_clock::now();
      auto result = detector.run(full_path);
      auto end = std::chrono::high_resolution_clock::now();
      auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

      draw_and_save_image(full_path, result, input_path + "/result_" + filename);
      save_labelme_json(full_path, result, input_path + "/" + fs::path(filename).replace_extension(".json").string());
      std::cout << "image: " << full_path << ", latency: " << std::to_string(latency) << "ms" << std::endl;
    }
  } else {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = detector.run(input_path);
    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    draw_and_save_image(input_path, result, "result_" + fs::path(input_path).filename().string());
    save_labelme_json(input_path, result, fs::path(input_path).replace_extension(".json").string());
    std::cout << "image: " << input_path << ", latency: " << std::to_string(latency) << "ms" << std::endl;
  }

  return 0;
}
