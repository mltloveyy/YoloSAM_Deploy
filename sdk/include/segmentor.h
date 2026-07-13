#ifndef SEGMENTOR_H
#define SEGMENTOR_H

#pragma once

#define INTERFACE_API __attribute__((visibility("default")))

#include <map>
#include <memory>
#include <string>
#include <vector>

struct SegmentorConfig {
  std::string model_path;
  int forward_type = 0;
  int num_threads = 1;
  int precision_mode = 0;
  int memory_mode = 0;
  int warmup = 0;
};

struct SegmentationResult {
  int x0, y0, x1, y1;
  std::vector<int> points;
};

// ---------------------------------------------------------------------------
// Segmentor — MNN inference engine.
// ---------------------------------------------------------------------------
class INTERFACE_API Segmentor {
 public:
  Segmentor(const SegmentorConfig& enc_config, const SegmentorConfig& dec_config);
  ~Segmentor();

  Segmentor(const Segmentor&) = delete;
  Segmentor& operator=(const Segmentor&) = delete;
  Segmentor(Segmentor&&) noexcept;
  Segmentor& operator=(Segmentor&&) noexcept;

  void set_image(const std::string& image_path);
  SegmentationResult forward(const std::vector<int>& point_coords, const std::vector<int>& point_labels);

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

#endif  // SEGMENTOR_H
