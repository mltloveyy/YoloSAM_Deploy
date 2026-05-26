#ifndef DETECTOR_H
#define DETECTOR_H

#pragma once

#define INTERFACE_API __attribute__((visibility("default")))

#include <map>
#include <memory>
#include <string>
#include <vector>

struct DetectionResult {
  float x0, y0, x1, y1;
  int class_id;
  std::string class_name;
  float confidence;
};

// ---------------------------------------------------------------------------
// Detector — MNN inference engine.
// ---------------------------------------------------------------------------
class INTERFACE_API Detector {
 public:
  Detector(const std::string& model_path, const std::map<int, std::string>& class_names, int forward_type = 0,
           int precision_mode = 0, int num_threads = 4, bool warmup = true);
  ~Detector();

  Detector(const Detector&) = delete;
  Detector& operator=(const Detector&) = delete;
  Detector(Detector&&) noexcept;
  Detector& operator=(Detector&&) noexcept;

  std::vector<DetectionResult> run(const std::string& image_path);

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

#endif  // DETECTOR_H
