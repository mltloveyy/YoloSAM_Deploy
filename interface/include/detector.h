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
  float confidence;
};

// ---------------------------------------------------------------------------
// Detector — MNN inference engine. All MNN headers stay in .cc.
// ---------------------------------------------------------------------------
class INTERFACE_API Detector {
 public:
  Detector(const std::string& model_path, int forward_type = 0, int precision_mode = 0, int num_threads = 1);
  ~Detector();

  Detector(const Detector&) = delete;
  Detector& operator=(const Detector&) = delete;
  Detector(Detector&&) noexcept;
  Detector& operator=(Detector&&) noexcept;

  std::vector<DetectionResult> run(const std::string& image_path);
  std::map<std::string, std::vector<DetectionResult>> run_directory(const std::string& input_dir);

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

// ---------------------------------------------------------------------------
// ResultProcessor — annotates images and writes LabelMe JSON.
// yaml-cpp / jsoncpp are implementation details.
// ---------------------------------------------------------------------------
class INTERFACE_API ResultProcessor {
 public:
  explicit ResultProcessor(const std::string& config_path);
  ~ResultProcessor();

  ResultProcessor(const ResultProcessor&) = delete;
  ResultProcessor& operator=(const ResultProcessor&) = delete;
  ResultProcessor(ResultProcessor&&) noexcept;
  ResultProcessor& operator=(ResultProcessor&&) noexcept;

  void draw_and_save_image(const std::string& image_path, const std::vector<DetectionResult>& detections,
                           const std::string& output_path);

  void save_labelme_json(const std::string& image_path, const std::vector<DetectionResult>& detections,
                         const std::string& json_output_path);

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

#endif  // DETECTOR_H
