#include "nx_control/controller.hpp"
#include "nx_control/io.hpp"
#include "nx_control/protocol.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

std::atomic<bool> stopping{false};

void signal_handler(int) { stopping.store(true); }

struct Options {
  std::string config = "config/nx-control.conf";
  std::string dmmc = "/dev/ttyACM0";
  std::string vision_bind = "127.0.0.1";
  std::uint16_t vision_port = 29001;
  int baud = 921600;
  std::string log = "logs/nx-control.csv";
  nx_control::TaskMode task = nx_control::TaskMode::HoldCenter;
  double target_m = 0.0;
  bool start_immediately = true;
  bool dry_run = false;
  double max_seconds = 0.0;
};

nx_control::TaskMode parse_task(const std::string& value) {
  if (value == "idle") return nx_control::TaskMode::Idle;
  if (value == "static") return nx_control::TaskMode::StaticSequence;
  if (value == "center") return nx_control::TaskMode::HoldCenter;
  if (value == "target") return nx_control::TaskMode::HoldTarget;
  if (value == "auto") return nx_control::TaskMode::AutoVehicle;
  throw std::invalid_argument("task must be idle, static, center, target, or auto");
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> std::string {
      if (++index >= argc) throw std::invalid_argument("missing value after " + argument);
      return argv[index];
    };
    if (argument == "--config") options.config = value();
    else if (argument == "--dmmc") options.dmmc = value();
    else if (argument == "--vision-bind") options.vision_bind = value();
    else if (argument == "--vision-port") options.vision_port = static_cast<std::uint16_t>(std::stoul(value()));
    else if (argument == "--baud") options.baud = std::stoi(value());
    else if (argument == "--log") options.log = value();
    else if (argument == "--task") options.task = parse_task(value());
    else if (argument == "--target-cm") options.target_m = std::stod(value()) / 100.0;
    else if (argument == "--wait-start") options.start_immediately = false;
    else if (argument == "--dry-run") options.dry_run = true;
    else if (argument == "--max-seconds") options.max_seconds = std::stod(value());
    else if (argument == "--help") {
      std::cout << "ball_nx_control [--config FILE] [--dmmc DEVICE] [--vision-port PORT]\n"
                   "  [--task idle|static|center|target|auto] [--target-cm CM] [--wait-start]\n"
                   "  [--log CSV] [--dry-run] [--max-seconds SECONDS]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (std::abs(options.target_m) > 0.10) throw std::invalid_argument("target must be within +/-10 cm");
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const nx_control::ControlConfig config = nx_control::load_config(options.config);
    nx_control::NxController controller(config);
    const double started = nx_control::monotonic_seconds();
    controller.reset(started);
    controller.configure_task(options.task, options.target_m, options.start_immediately);

    nx_control::UdpReceiver vision;
    const bool vision_udp_enabled = !options.dry_run;
    if (vision_udp_enabled && !vision.open(options.vision_bind, options.vision_port)) {
      throw std::runtime_error("cannot bind vision UDP: " + vision.last_error());
    }
    nx_control::SerialPort serial;
    double next_serial_retry = started;
    if (!options.dry_run && !serial.open(options.dmmc, options.baud)) {
      std::cerr << "DMMC serial waiting: " << serial.last_error() << '\n';
    }
    nx_control::CsvLogger logger;
    if (!options.log.empty() && !logger.open(options.log)) {
      throw std::runtime_error("cannot open log: " + options.log);
    }

    nx_control::protocol::StreamParser vision_parser;
    nx_control::protocol::StreamParser dmmc_parser;
    nx_control::TubeStatus latest_tube;
    nx_control::ChassisState latest_chassis;
    bool have_tube = false;
    bool have_chassis = false;
    std::uint8_t buffer[512];
    double next_tick = started;
    double next_report = started;
    std::uint32_t dry_sequence = 0;
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    while (!stopping.load()) {
      double now = nx_control::monotonic_seconds();
      while (vision_udp_enabled) {
        const std::ptrdiff_t count = vision.read(buffer, sizeof(buffer));
        if (count <= 0) break;
        for (const auto& frame : vision_parser.feed(buffer, static_cast<std::size_t>(count))) {
          const auto measurement = nx_control::protocol::decode_vision(frame, now);
          if (measurement) controller.ingest_vision(*measurement);
        }
      }
      if (serial.fd() >= 0) {
        while (true) {
          const std::ptrdiff_t count = serial.read(buffer, sizeof(buffer));
          if (count <= 0) break;
          for (const auto& frame : dmmc_parser.feed(buffer, static_cast<std::size_t>(count))) {
            if (const auto status = nx_control::protocol::decode_tube_status(frame, now)) {
              latest_tube = *status;
              have_tube = true;
              controller.ingest_tube_status(*status);
            } else if (const auto chassis = nx_control::protocol::decode_chassis_state(frame, now)) {
              latest_chassis = *chassis;
              have_chassis = true;
              controller.ingest_chassis_state(*chassis);
            }
          }
        }
      } else if (!options.dry_run && now >= next_serial_retry) {
        if (serial.open(options.dmmc, options.baud)) {
          std::cerr << "DMMC serial connected: " << options.dmmc << '\n';
        }
        next_serial_retry = now + 1.0;
      }

      if (now >= next_tick) {
        if (options.dry_run) {
          latest_tube.sequence = ++dry_sequence;
          latest_tube.receive_time_s = now;
          latest_tube.dmmc_time_ms = static_cast<std::uint32_t>(std::llround(now * 1000.0));
          latest_tube.theta_actual_rad = 0.0;
          latest_tube.faults = 0;
          latest_chassis.sequence = dry_sequence;
          latest_chassis.receive_time_s = now;
          latest_chassis.chassis_time_ms =
              static_cast<std::uint32_t>(std::llround(now * 1000.0));
          latest_chassis.ttl_ms = 100;
          latest_chassis.track_quality = 1.0;
          controller.ingest_tube_status(latest_tube);
          controller.ingest_chassis_state(latest_chassis);
          nx_control::VisionMeasurement vision_measurement;
          vision_measurement.frame_id = dry_sequence;
          vision_measurement.status = nx_control::VisionStatus::Measured;
          vision_measurement.ball_confidence = 1.0;
          vision_measurement.tube_confidence = 1.0;
          vision_measurement.capture_time_ms =
              static_cast<std::uint32_t>(std::llround(now * 1000.0));
          vision_measurement.has_capture_time = true;
          vision_measurement.receive_time_s = now;
          controller.ingest_vision(vision_measurement);
          have_tube = have_chassis = true;
        }
        const nx_control::ControlOutput output = controller.tick(now);
        const std::vector<std::uint8_t> packet =
            nx_control::protocol::encode_control_command(output.command);
        if (serial.fd() >= 0 && !serial.write_all(packet.data(), packet.size())) {
          std::cerr << "DMMC serial write failed: " << serial.last_error() << '\n';
          serial.close();
        }
        logger.write(now, output, have_tube ? &latest_tube : nullptr,
                     have_chassis ? &latest_chassis : nullptr);
        if (now >= next_report) {
          std::cerr << "state=" << static_cast<int>(output.command.control_state)
                    << " x=" << output.estimate.position_m * 100.0 << "cm"
                    << " ref=" << output.reference.position_m * 100.0 << "cm"
                    << " theta=" << output.command.theta_cmd_rad * 180.0 / 3.14159265358979323846
                    << "deg mpc=" << output.mpc_solve_ms << "ms"
                    << " flags=0x" << std::hex << static_cast<int>(output.command.flags) << std::dec
                    << " reason=" << output.reason << '\n';
          next_report = now + 1.0;
        }
        next_tick += config.period_s;
        const double completed = nx_control::monotonic_seconds();
        if (completed - next_tick > config.period_s) next_tick = completed + config.period_s;
      }
      if (options.max_seconds > 0.0 && now - started >= options.max_seconds) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}
