#include "nx_control/controller.hpp"
#include "nx_control/io.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> result;
  std::stringstream stream(line);
  std::string value;
  while (std::getline(stream, value, ',')) result.push_back(value);
  return result;
}

double number(const std::vector<std::string>& row, std::size_t index, double fallback = 0.0) {
  return index < row.size() && !row[index].empty() ? std::stod(row[index]) : fallback;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 5) {
      std::cerr
          << "usage: ball_nx_replay INPUT.csv [OUTPUT.csv] [CONFIG] [TASK]\n"
             "  TASK: center (default) or task3\n";
      return 2;
    }
    const std::string output_path = argc >= 3 ? argv[2] : "replay-output.csv";
    const std::string config_path = argc >= 4 ? argv[3] : "config/nx-control.conf";
    const std::string task_name = argc >= 5 ? argv[4] : "center";
    const nx_control::TaskMode task_mode =
        task_name == "task3" || task_name == "3"
            ? nx_control::TaskMode::Contest3
            : nx_control::TaskMode::HoldCenter;
    if (task_name != "center" && task_name != "task3" && task_name != "3") {
      throw std::runtime_error("replay TASK must be center or task3");
    }
    const nx_control::ControlConfig config = nx_control::load_config(config_path);
    nx_control::NxController controller(config);
    controller.configure_task(task_mode, 0.0, true);
    nx_control::CsvLogger logger;
    if (!logger.open(output_path)) throw std::runtime_error("cannot open replay output");
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("cannot open replay input");
    std::string line;
    std::getline(input, line);  // header
    bool initialized = false;
    std::uint32_t sequence = 0;
    while (std::getline(input, line)) {
      if (line.empty()) continue;
      const auto row = split(line);
      if (row.size() < 10) throw std::runtime_error("replay row needs 10 columns");
      const double now = number(row, 0);
      if (!initialized) {
        controller.reset(now);
        initialized = true;
      }
      nx_control::VisionMeasurement vision;
      vision.frame_id = ++sequence;
      vision.status = static_cast<nx_control::VisionStatus>(static_cast<int>(number(row, 1)));
      vision.position_m = number(row, 2);
      vision.ball_confidence = number(row, 3, 1.0);
      vision.tube_confidence = number(row, 4, 1.0);
      vision.capture_time_s = now;
      vision.receive_time_s = now;
      vision.capture_time_ms = static_cast<std::uint32_t>(now * 1000.0);
      vision.has_capture_time = true;
      nx_control::TubeStatus tube;
      tube.sequence = sequence;
      tube.dmmc_time_ms = static_cast<std::uint32_t>(now * 1000.0);
      tube.theta_actual_rad = number(row, 5);
      tube.receive_time_s = now;
      nx_control::ChassisState chassis;
      chassis.sequence = sequence;
      chassis.chassis_time_ms = static_cast<std::uint32_t>(now * 1000.0);
      chassis.acceleration_actual_m_s2 = number(row, 6);
      chassis.acceleration_ref_m_s2 = number(row, 7);
      chassis.jerk_ref_m_s3 = number(row, 8);
      chassis.motion_phase = static_cast<nx_control::MotionPhase>(static_cast<int>(number(row, 9)));
      chassis.ttl_ms = 100;
      chassis.track_quality = 1.0;
      chassis.receive_time_s = now;
      controller.ingest_tube_status(tube);
      controller.ingest_chassis_state(chassis);
      controller.ingest_vision(vision);
      const nx_control::ControlOutput output = controller.tick(now);
      logger.write(now, output, &tube, &chassis);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "replay failed: " << error.what() << '\n';
    return 1;
  }
}
