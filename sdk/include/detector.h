#ifndef DETECTOR_H
#define DETECTOR_H

#pragma once

#define INTERFACE_API __attribute__((visibility("default")))

#include <map>
#include <memory>
#include <string>
#include <vector>

struct DetectorConfig {
  std::string model_path;
  std::map<int, std::string> class_names;
  int forward_type = 0;
  int num_threads = 1;
  int precision_mode = 0;
  int memory_mode = 0;
  int warmup = 0;
};

struct DetectionResult {
  int x0, y0, x1, y1;
  int class_id;
  std::string class_name;
  float confidence;
};

// ---------------------------------------------------------------------------
// Detector — MNN inference engine.
// ---------------------------------------------------------------------------
class INTERFACE_API Detector {
 public:
  Detector(const DetectorConfig& config);
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
