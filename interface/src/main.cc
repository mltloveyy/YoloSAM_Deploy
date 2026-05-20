#include <filesystem>
#include <iostream>

#include "detector.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <model_path> <input_image_or_dir> <config_path>\n";
    return -1;
  }

  std::string model_path = argv[1];
  std::string input_path = argv[2];
  std::string config_path = argv[3];

  Detector detector(model_path);
  ResultProcessor processor(config_path);

  if (fs::is_directory(input_path)) {
    auto results = detector.run_directory(input_path);
    for (const auto& [filename, detections] : results) {
      std::string full_path = input_path + "/" + filename;
      processor.draw_and_save_image(full_path, detections, input_path + "/results_" + filename);
      processor.save_labelme_json(full_path, detections,
                                  input_path + "/" + fs::path(filename).replace_extension(".json").string());
      std::cout << ">>> " << full_path << std::endl;
    }
  } else {
    auto detections = detector.run(input_path);
    processor.draw_and_save_image(input_path, detections, "results_" + fs::path(input_path).filename().string());
    processor.save_labelme_json(input_path, detections, fs::path(input_path).replace_extension(".json").string());
    std::cout << ">>> " << input_path << std::endl;
  }

  return 0;
}
