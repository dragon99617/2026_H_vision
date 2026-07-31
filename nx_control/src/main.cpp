#include "nx_control/controller.hpp"
#include "nx_control/io.hpp"
#include "nx_control/protocol.hpp"

#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

std::atomic<bool> stopping{false};

void signal_handler(int) { stopping.store(true); }

std::uint64_t parse_u64(const std::string& value, const std::string& option) {
  if (value.empty() || value.front() == '-') {
    throw std::invalid_argument("invalid value for " + option);
  }
  std::size_t consumed = 0;
  const unsigned long long result = std::stoull(value, &consumed, 0);
  if (consumed != value.size()) throw std::invalid_argument("invalid value for " + option);
  return static_cast<std::uint64_t>(result);
}

struct Options {
  std::string config = "config/nx-control.conf";
  std::string dmmc = "/dev/ttyACM0";
  std::string vision_bind = "127.0.0.1";
  std::uint16_t vision_port = 29001;
  int baud = 921600;
  std::string log = "logs/nx-control.csv";
  std::string state_file = "state/tube-control-v3.seq";
  std::uint32_t start_command_id = 1U;
  nx_control::TaskMode task = nx_control::TaskMode::HoldCenter;
  double target_m = 0.0;
  bool target_set = false;
  bool start_immediately = true;
  bool wait_start = false;
  bool key_start = false;
  bool dry_run = false;
  double max_seconds = 0.0;
};

class KeyboardStart {
 public:
  explicit KeyboardStart(bool enabled) {
    if (!enabled) return;
    if (::isatty(STDIN_FILENO) == 0) {
      throw std::runtime_error("--key-start requires an interactive terminal");
    }
    if (::tcgetattr(STDIN_FILENO, &saved_) != 0) {
      throw std::runtime_error("cannot read terminal settings");
    }
    termios settings = saved_;
    settings.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
    settings.c_cc[VMIN] = 0;
    settings.c_cc[VTIME] = 0;
    if (::tcsetattr(STDIN_FILENO, TCSANOW, &settings) != 0) {
      throw std::runtime_error("cannot enable keyboard start");
    }
    active_ = true;
  }

  ~KeyboardStart() {
    if (active_) ::tcsetattr(STDIN_FILENO, TCSANOW, &saved_);
  }

  KeyboardStart(const KeyboardStart&) = delete;
  KeyboardStart& operator=(const KeyboardStart&) = delete;

  bool consume_start() const {
    bool requested = false;
    char input[32];
    while (true) {
      const ssize_t count = ::read(STDIN_FILENO, input, sizeof(input));
      if (count > 0) {
        for (ssize_t index = 0; index < count; ++index) {
          if (input[index] == 'd' || input[index] == 'D') requested = true;
        }
      } else if (count == 0 || errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      } else if (errno != EINTR) {
        throw std::runtime_error("cannot read keyboard start key");
      }
    }
    return requested;
  }

 private:
  termios saved_{};
  bool active_ = false;
};

nx_control::TaskMode parse_task(const std::string& value) {
  if (value == "idle") return nx_control::TaskMode::Idle;
  if (value == "static") return nx_control::TaskMode::StaticSequence;
  if (value == "center") return nx_control::TaskMode::HoldCenter;
  if (value == "target") return nx_control::TaskMode::HoldTarget;
  if (value == "auto") return nx_control::TaskMode::AutoVehicle;
  if (value == "3" || value == "task3") return nx_control::TaskMode::Contest3;
  if (value == "4" || value == "5" || value == "45" || value == "4-5" ||
      value == "task4" || value == "task5" || value == "task45" ||
      value == "task4_5") {
    return nx_control::TaskMode::Contest45;
  }
  if (value == "6" || value == "task6") return nx_control::TaskMode::Contest6;
  throw std::invalid_argument(
      "task must be 3, 45, 6, idle, static, center, target, or auto");
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
    else if (argument == "--state-file") options.state_file = value();
    else if (argument == "--start-command-id") {
      const std::uint64_t parsed = parse_u64(value(), argument);
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--start-command-id exceeds uint32");
      }
      options.start_command_id = static_cast<std::uint32_t>(parsed);
    }
    else if (argument == "--task") options.task = parse_task(value());
    else if (argument == "--target-cm") {
      options.target_m = std::stod(value()) / 100.0;
      options.target_set = true;
    }
    else if (argument == "--wait-start") {
      options.wait_start = true;
      options.start_immediately = false;
    }
    else if (argument == "--key-start" || argument == "--keyboard-start") {
      options.key_start = true;
      options.start_immediately = false;
    }
    else if (argument == "--dry-run") options.dry_run = true;
    else if (argument == "--max-seconds") options.max_seconds = std::stod(value());
    else if (argument == "--help") {
      std::cout << "ball_nx_control [--config FILE] [--dmmc DEVICE] [--vision-port PORT]\n"
                   "  [--task 3|45|6|idle|static|center|target|auto] [--target-cm CM]\n"
                   "  [--wait-start | --key-start]  (--wait-start is unavailable for 45/6)\n"
                   "  [--log CSV] [--state-file FILE] [--start-command-id ID]\n"
                   "  [--dry-run] [--max-seconds SECONDS]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (!std::isfinite(options.target_m) || std::abs(options.target_m) > 0.10) {
    throw std::invalid_argument("target must be finite and within +/-10 cm");
  }
  if (options.task == nx_control::TaskMode::Contest6 && !options.target_set) {
    throw std::invalid_argument("task 6 requires --target-cm CM");
  }
  if (options.wait_start &&
      (options.task == nx_control::TaskMode::Contest45 ||
       options.task == nx_control::TaskMode::Contest6)) {
    throw std::invalid_argument(
        "tasks 4/5/6 do not use chassis events; use immediate start or --key-start");
  }
  if (options.state_file.empty() && !options.dry_run) {
    throw std::invalid_argument("--state-file cannot be empty");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    const nx_control::ControlConfig config = nx_control::load_config(options.config);
    nx_control::NxController controller(config);
    std::cerr << "Controller: cascaded PID (NX position outer loop, MC02 angle inner loop)\n";
    const double started = nx_control::monotonic_seconds();
    controller.reset(started);
    controller.configure_task(options.task, options.target_m, options.start_immediately,
                              !options.key_start);
    KeyboardStart keyboard(options.key_start);
    if (options.key_start) {
      std::cerr << "keyboard start enabled: press d to start/restart the task\n";
    }

    nx_control::UdpReceiver vision;
    const bool vision_udp_enabled = !options.dry_run;
    if (vision_udp_enabled && !vision.open(options.vision_bind, options.vision_port)) {
      throw std::runtime_error("cannot bind vision UDP: " + vision.last_error());
    }
    std::unique_ptr<nx_control::PersistentSequence> persistent_sequence;
    if (!options.dry_run) {
      persistent_sequence = std::make_unique<nx_control::PersistentSequence>(
          options.state_file, options.start_command_id);
      persistent_sequence->prepare();
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
    nx_control::protocol::Mc02StreamParser dmmc_parser;
    nx_control::TubeStatus latest_tube;
    nx_control::ChassisState latest_chassis;
    bool have_tube = false;
    bool have_chassis = false;
    std::uint8_t buffer[512];
    double next_tick = started;
    double next_report = started;
    bool last_reported_safety_latched = false;
    std::uint64_t last_reported_safety_event_id = 0;
    std::uint32_t dry_sequence = 0;
    while (!stopping.load()) {
      double now = nx_control::monotonic_seconds();
      if (keyboard.consume_start()) {
        controller.start_task(now);
        std::cerr << "task started/restarted by key d\n";
      }
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
        const auto command_id =
            persistent_sequence
                ? std::optional<std::uint32_t>(persistent_sequence->next())
                : std::nullopt;
        const nx_control::ControlOutput output = controller.tick(now, command_id);
        const std::vector<std::uint8_t> packet =
            nx_control::protocol::encode_control_command(output.command);
        if (serial.fd() >= 0 && !serial.write_all(packet.data(), packet.size())) {
          std::cerr << "DMMC serial write failed after command_id="
                    << output.command.command_id << ": " << serial.last_error()
                    << "; ID will not be retried\n";
          serial.close();
        }
        logger.write(now, output, have_tube ? &latest_tube : nullptr,
                     have_chassis ? &latest_chassis : nullptr);
        if (output.safety_event_id != last_reported_safety_event_id) {
          std::cerr << "SAFETY LATCHED event_id=" << output.safety_event_id
                    << " command_id=" << output.command.command_id
                    << " reason=" << output.last_stop_reason << '\n';
        } else if (last_reported_safety_latched && !output.safety_latched) {
          std::cerr << "SAFETY CLEARED by operator restart command_id="
                    << output.command.command_id << '\n';
        }
        last_reported_safety_latched = output.safety_latched;
        last_reported_safety_event_id = output.safety_event_id;
        if (now >= next_report) {
          std::cerr << "state=" << static_cast<int>(output.command.control_state)
                    << " x=" << output.estimate.position_m * 100.0 << "cm"
                     << " ref=" << output.reference.position_m * 100.0 << "cm"
                     << " theta=" << output.command.theta_cmd_rad * 180.0 / 3.14159265358979323846
                     << "deg pid=" << output.pid.proportional_m_s2 << '/'
                     << output.pid.integral_m_s2 << '/'
                     << output.pid.derivative_m_s2
                     << " inner_error="
                     << output.inner_angle_error_rad * 180.0 /
                            3.14159265358979323846
                     << "deg"
                    << " early_brake="
                    << (output.task3_early_braking ? 1 : 0)
                    << " overshoot_recovery="
                    << (output.task3_positive_overshoot_recovery ? 1 : 0)
                    << " reverse_balance="
                    << (output.task3_reverse_balance_active ? 1 : 0)
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
    if (persistent_sequence) persistent_sequence->checkpoint();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}
