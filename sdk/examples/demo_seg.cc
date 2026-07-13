#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>

#include "MNN/ImageProcess.hpp"
#include "cv/cv.hpp"
#include "segmentor.h"

namespace fs = std::filesystem;

std::vector<int> parseIntVec(std::string s) {
  std::vector<int> res;
  std::stringstream ss(s);
  std::string t;
  while (getline(ss, t, ','))
    if (!t.empty()) res.push_back(std::stoi(t));
  return res;
}

void Visualize(const std::string& image_path, const SegmentationResult& result, const std::vector<int>& coords,
               const std::vector<int>& labels) {
  auto image = MNN::CV::imread(image_path);

  int num_coords = coords.size(), num_labels = labels.size();
  int num_points = num_labels < num_coords / 2 ? num_labels : num_coords / 2;
  for (auto i = 0; i < num_points; i++) {
    float x = static_cast<float>(coords[2 * i]);
    float y = static_cast<float>(coords[2 * i + 1]);
    if (labels[i] > 1) {
      MNN::CV::circle(image, {x, y}, 5, {0, 255, 0}, -1);
    } else {
      MNN::CV::circle(image, {x, y}, 5, {0, 180, 0}, -1);
    }
  }

  std::vector<MNN::CV::Point> contour;
  for (int i = 0; i < result.points.size() / 2; ++i) {
    float x = static_cast<float>(result.points[2 * i]);
    float y = static_cast<float>(result.points[2 * i + 1]);
    contour.push_back({x, y});
  }
  MNN::CV::drawContours(image, {contour}, -1, {255, 0, 0}, 2);
  float x0 = static_cast<float>(result.x0);
  float y0 = static_cast<float>(result.y0);
  float x1 = static_cast<float>(result.x1);
  float y1 = static_cast<float>(result.y1);
  MNN::CV::rectangle(image, {x0, y0}, {x1, y1}, {0, 0, 255}, 2);

  auto parent_path = fs::absolute(image_path).parent_path().string();
  auto stemname = fs::path(image_path).stem().string();
  auto draw_path = parent_path + "/" + stemname + "_seg.jpg";

  MNN::CV::imwrite(draw_path, image);
}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " enc_path dec_path image_path coords labels [loop_count] [enc_warmup] [enc_forward_type] [enc_num_threads] "
                 "[enc_precision_mode] [enc_memory_mode]"
              << std::endl;
    return -1;
  }

  std::string enc_path = argv[1];
  std::string dec_path = argv[2];
  std::string image_path = argv[3];
  auto coords = parseIntVec(argv[4]);
  auto labels = parseIntVec(argv[5]);
  int loop_count = argc > 6 ? std::stoi(argv[6]) : 5;
  int warmup = argc > 7 ? std::stoi(argv[7]) : 0;
  int forward_type = argc > 8 ? std::stoi(argv[8]) : 0;  // MNN_FORWARD_CPU
  int num_threads = argc > 9 ? std::stoi(argv[9]) : 1;
  int precision_mode = argc > 10 ? std::stoi(argv[10]) : 0;  // Precision_Normal
  int memory_mode = argc > 11 ? std::stoi(argv[11]) : 0;     // Memory_Normal

  SegmentorConfig enc_cfg, dec_cfg;
  enc_cfg.model_path = enc_path;
  enc_cfg.forward_type = forward_type;
  enc_cfg.num_threads = num_threads;
  enc_cfg.precision_mode = precision_mode;
  enc_cfg.memory_mode = memory_mode;
  enc_cfg.warmup = warmup;
  dec_cfg.model_path = dec_path;
  dec_cfg.forward_type = 0;  // MNN_FORWARD_CPU
  dec_cfg.num_threads = 1;
  dec_cfg.precision_mode = 2;  // Precision_Low
  dec_cfg.memory_mode = 0;     // Memory_Normal
  dec_cfg.warmup = 0;
  Segmentor segmentor(enc_cfg, dec_cfg);

  for (int i = 0; i < loop_count; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    segmentor.set_image(image_path);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto result = segmentor.forward(coords, labels);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto enc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto dec_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "image " << image_path << ", latency: " << enc_duration << "ms(encode) + " << dec_duration << "ms(decode)"
              << " bbox: (" << result.x0 << ", " << result.y0 << ", " << result.x1 << ", " << result.y1 << ")" << std::endl;

    // Visualize(image_path, result, coords, labels);
  }

  return 0;
}
