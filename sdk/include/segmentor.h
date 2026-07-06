#ifndef SEGMENTOR_H
#define SEGMENTOR_H

#pragma once

#define INTERFACE_API __attribute__((visibility("default")))

#include <map>
#include <memory>
#include <string>
#include <vector>

struct SegmentResult {
  int x0, y0, x1, y1;
  std::vector<int> points;
};

// ---------------------------------------------------------------------------
// Segmentor — MNN inference engine.
// ---------------------------------------------------------------------------
class INTERFACE_API Segmentor {
 public:
  Segmentor(const std::string& enc_path, const std::string& dec_path, int num_threads = 2, int precision_mode = 0,
            int forward_type = 0, bool warmup = false);
  ~Segmentor();

  Segmentor(const Segmentor&) = delete;
  Segmentor& operator=(const Segmentor&) = delete;
  Segmentor(Segmentor&&) noexcept;
  Segmentor& operator=(Segmentor&&) noexcept;

  void set_image(const std::string& image_path);
  SegmentResult forward(const std::vector<int>& point_coords, const std::vector<int>& point_labels);

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

#endif  // SEGMENTOR_H
