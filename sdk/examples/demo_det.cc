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

void VisualizeAndExport(const std::string& image_path, const std::vector<DetectionResult>& result, float threshold = 0.2) {
  auto image = MNN::CV::imread(image_path);
  auto dims = image->getInfo()->dim;

  Json::Value root;
  root["version"] = "4.5.6";
  root["flags"] = Json::objectValue;
  root["imageHeight"] = dims[0];
  root["imageWidth"] = dims[1];
  root["shapes"] = Json::arrayValue;
  for (const auto& r : result) {
    if (r.confidence < threshold) continue;
    std::cout << r.class_name << ": " << r.confidence << std::endl;

    Json::Value shape(Json::objectValue);
    shape["label"] = r.class_name;
    shape["shape_type"] = "rectangle";
    Json::Value points(Json::arrayValue);
    Json::Value p0(Json::arrayValue);
    p0.append(r.x0);
    p0.append(r.y0);
    points.append(p0);
    Json::Value p1(Json::arrayValue);
    p1.append(r.x1);
    p1.append(r.y1);
    points.append(p1);
    shape["points"] = points;
    shape["flags"] = Json::objectValue;
    root["shapes"].append(shape);

    float x0 = static_cast<float>(r.x0);
    float y0 = static_cast<float>(r.y0);
    float x1 = static_cast<float>(r.x1);
    float y1 = static_cast<float>(r.y1);
    MNN::CV::rectangle(image, {x0, y0}, {x1, y1}, {0, 0, 180}, 2);
  }

  auto parent_path = fs::absolute(image_path).parent_path().string();
  auto stemname = fs::path(image_path).stem().string();
  auto draw_path = parent_path + "/" + stemname + "_det.jpg";
  auto json_path = parent_path + "/" + stemname + ".json";

  MNN::CV::imwrite(draw_path, image);

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
  float threshold = argc > 4 ? std::stof(argv[4]) : 0.2;
  int num_threads = argc > 5 ? std::stoi(argv[5]) : 2;
  int precision_mode = argc > 6 ? std::stoi(argv[6]) : 0;

  Detector detector(model_path, ParseClassNames(config_path), num_threads, precision_mode);

  if (fs::is_directory(input_path)) {
    std::vector<std::string> image_paths;
    for (const auto& entry : fs::directory_iterator(input_path)) {
      if (!fs::is_regular_file(entry)) continue;
      const auto ext = entry.path().extension().string();
      if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;
      image_paths.push_back(entry.path().string());
    }

    int image_num = image_paths.size();
    for (int i = 0; i < image_num; ++i) {
      auto image_path = image_paths[i];
      auto start = std::chrono::high_resolution_clock::now();
      auto result = detector.run(image_path);
      auto end = std::chrono::high_resolution_clock::now();
      auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      std::cout << "image " << i + 1 << "/" << image_num << " " << image_path << ", latency: " << latency << "ms" << std::endl;
      for (const auto& r : result) {
        if (r.confidence < threshold) continue;
        std::cout << r.class_name << ": " << r.confidence << " (" << r.x0 << ", " << r.y0 << ", " << r.x1 << ", " << r.y1 << ")"
                  << std::endl;
      }
      VisualizeAndExport(image_path, result, threshold);
    }
  } else {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = detector.run(input_path);
    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "image " << input_path << ", latency: " << latency << "ms" << std::endl;
    for (const auto& r : result) {
      if (r.confidence < threshold) continue;
      std::cout << r.class_name << ": " << r.confidence << " (" << r.x0 << ", " << r.y0 << ", " << r.x1 << ", " << r.y1 << ")"
                << std::endl;
    }
    VisualizeAndExport(input_path, result, threshold);
  }

  return 0;
}
