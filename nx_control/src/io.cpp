#include "nx_control/io.hpp"
#include "nx_control/friction_compensator.hpp"
#include "nx_control/protocol.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace nx_control {
namespace {

constexpr std::uint64_t kSequenceReservationSize = 4096U;
constexpr std::size_t kSequenceStateRecordSize = 21U;
constexpr double kRadiansToDegrees =
    180.0 / 3.14159265358979323846;

speed_t baud_flag(int baud) {
  switch (baud) {
    case 115200:
      return B115200;
    case 230400:
      return B230400;
    case 460800:
      return B460800;
    case 921600:
      return B921600;
    default:
      throw std::invalid_argument("unsupported serial baud");
  }
}

bool parse_bool(const std::string& value) {
  if (value == "true" || value == "1" || value == "yes") return true;
  if (value == "false" || value == "0" || value == "no") return false;
  throw std::invalid_argument("invalid boolean: " + value);
}

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

}  // namespace

double monotonic_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

PersistentSequence::PersistentSequence(const std::string& path, std::uint32_t initial)
    : path_(path) {
  const std::filesystem::path state_path(path);
  if (!state_path.parent_path().empty()) {
    std::filesystem::create_directories(state_path.parent_path());
  }
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd_ < 0) throw_system_error("cannot open sequence file");
  if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
    const int saved_errno = errno;
    ::close(fd_);
    fd_ = -1;
    errno = saved_errno;
    throw_system_error("sequence file is already locked by another sender");
  }
  next_unreserved_ = read_state(initial);
  next_ = next_unreserved_;
  reserved_end_ = next_unreserved_;
}

PersistentSequence::~PersistentSequence() {
  if (fd_ >= 0) ::close(fd_);
}

void PersistentSequence::prepare() {
  if (next_ == reserved_end_) reserve();
}

std::uint32_t PersistentSequence::next() {
  if (next_ == reserved_end_) reserve();
  const std::uint64_t value = next_++;
  return static_cast<std::uint32_t>(value);
}

void PersistentSequence::checkpoint() { persist(next_); }

[[noreturn]] void PersistentSequence::throw_system_error(
    const std::string& message) const {
  throw std::runtime_error(message + " '" + path_ + "': " + std::strerror(errno));
}

std::uint64_t PersistentSequence::read_state(std::uint32_t initial) {
  char record[kSequenceStateRecordSize + 1U];
  const ssize_t count = ::pread(fd_, record, sizeof(record), 0);
  if (count < 0) throw_system_error("cannot read sequence file");
  if (count == 0) return initial;
  if (count != static_cast<ssize_t>(kSequenceStateRecordSize) || record[20] != '\n') {
    throw std::runtime_error("invalid sequence file '" + path_ +
                             "'; refusing to risk an ID reuse");
  }
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < 20U; ++index) {
    if (record[index] < '0' || record[index] > '9') {
      throw std::runtime_error("invalid sequence file '" + path_ +
                               "'; refusing to risk an ID reuse");
    }
    value = value * 10U + static_cast<unsigned>(record[index] - '0');
  }
  if (value > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1U) {
    throw std::runtime_error("sequence file value is out of range");
  }
  return value;
}

void PersistentSequence::reserve() {
  constexpr std::uint64_t kEnd =
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1U;
  if (next_unreserved_ >= kEnd) {
    throw std::runtime_error("command_id space exhausted; refusing to wrap or repeat");
  }
  next_ = next_unreserved_;
  reserved_end_ = std::min(kEnd, next_ + kSequenceReservationSize);
  persist(reserved_end_);
  next_unreserved_ = reserved_end_;
}

void PersistentSequence::persist(std::uint64_t value) {
  std::ostringstream stream;
  stream << std::setw(20) << std::setfill('0') << value << '\n';
  const std::string record = stream.str();
  const ssize_t count = ::pwrite(fd_, record.data(), record.size(), 0);
  if (count != static_cast<ssize_t>(record.size())) {
    if (count >= 0) errno = EIO;
    throw_system_error("cannot update sequence file");
  }
  if (::fdatasync(fd_) != 0) throw_system_error("cannot sync sequence file");
}

ControlConfig load_config(const std::string& path) {
  ControlConfig config;
  if (path.empty()) return config;
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open config: " + path);
  std::unordered_map<std::string, std::string> values;
  std::string line;
  int line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    line = trim(line.substr(0, line.find('#')));
    if (line.empty()) continue;
    const auto separator = line.find('=');
    if (separator == std::string::npos) {
      throw std::runtime_error("invalid config line " + std::to_string(line_number));
    }
    values[trim(line.substr(0, separator))] = trim(line.substr(separator + 1));
  }
  auto number = [&](const char* key, double& target) {
    const auto found = values.find(key);
    if (found != values.end()) target = std::stod(found->second);
  };
  auto integer = [&](const char* key, int& target) {
    const auto found = values.find(key);
    if (found != values.end()) target = std::stoi(found->second);
  };
  auto boolean = [&](const char* key, bool& target) {
    const auto found = values.find(key);
    if (found != values.end()) target = parse_bool(found->second);
  };
  number("period_s", config.period_s);
  number("actuator_tau_s", config.actuator_tau_s);
  number("actuator_delay_s", config.actuator_delay_s);
  number("rolling_lambda", config.rolling_lambda);
  integer("horizon", config.horizon);
  number("theta_limit_rad", config.theta_limit_rad);
  number("theta_rate_limit_rad_s", config.theta_rate_limit_rad_s);
  number("position_soft_limit_m", config.position_soft_limit_m);
  number("position_safe_limit_m", config.position_safe_limit_m);
  number("position_scale_m", config.position_scale_m);
  number("velocity_scale_m_s", config.velocity_scale_m_s);
  number("input_scale_m_s2", config.input_scale_m_s2);
  number("delta_input_scale_m_s2", config.delta_input_scale_m_s2);
  number("hold_enter_position_error_m", config.hold_enter_position_error_m);
  number("hold_enter_velocity_m_s", config.hold_enter_velocity_m_s);
  number("hold_exit_position_error_m", config.hold_exit_position_error_m);
  number("task3_reference_max_velocity_m_s",
         config.task3_reference_max_velocity_m_s);
  number("task3_reference_max_acceleration_m_s2",
         config.task3_reference_max_acceleration_m_s2);
  number("task3_reference_max_jerk_m_s3",
         config.task3_reference_max_jerk_m_s3);
  number("task3_settle_position_error_m",
         config.task3_settle_position_error_m);
  number("task3_settle_velocity_m_s", config.task3_settle_velocity_m_s);
  number("task3_settle_dwell_s", config.task3_settle_dwell_s);
  number("task3_settle_theta_tolerance_rad",
         config.task3_settle_theta_tolerance_rad);
  number("task3_theta_bias_rad", config.task3_theta_bias_rad);
  number("task3_theta_static_rad", config.task3_theta_static_rad);
  number("task3_theta_margin_rad", config.task3_theta_margin_rad);
  number("task3_rolling_compensation_rad",
         config.task3_rolling_compensation_rad);
  number("task3_friction_blend_time_s",
         config.task3_friction_blend_time_s);
  number("task3_friction_breakaway_timeout_s",
         config.task3_friction_breakaway_timeout_s);
  number("task3_braking_deceleration_m_s2",
         config.task3_braking_deceleration_m_s2);
  number("task3_early_brake_position_m",
         config.task3_early_brake_position_m);
  number("task3_friction_rolling_enter_velocity_m_s",
         config.task3_friction_rolling_enter_velocity_m_s);
  number("task3_friction_stationary_enter_velocity_m_s",
         config.task3_friction_stationary_enter_velocity_m_s);
  number("task3_friction_disable_position_error_m",
         config.task3_friction_disable_position_error_m);
  number("task3_friction_disable_velocity_m_s",
         config.task3_friction_disable_velocity_m_s);
  number("task3_friction_request_acceleration_m_s2",
         config.task3_friction_request_acceleration_m_s2);
  number("slack_weight", config.slack_weight);
  number("measurement_sigma_m", config.measurement_sigma_m);
  number("predicted_min_confidence", config.predicted_min_confidence);
  number("process_accel_sigma_m_s2", config.process_accel_sigma_m_s2);
  number("process_disturbance_sigma_m_s3", config.process_disturbance_sigma_m_s3);
  number("innovation_gate_sigma", config.innovation_gate_sigma);
  number("observer_history_s", config.observer_history_s);
  number("vision_v2_latency_s", config.vision_v2_latency_s);
  number("vision_frame_rate_hz", config.vision_frame_rate_hz);
  number("vision_decay_start_s", config.vision_decay_start_s);
  if (values.find("vision_loss_hold_s") != values.end()) {
    number("vision_loss_hold_s", config.vision_loss_hold_s);
  } else {
    // Backward compatibility: the old stop threshold is now the soft HOLD threshold.
    number("vision_loss_stop_s", config.vision_loss_hold_s);
  }
  number("vision_loss_safe_s", config.vision_loss_safe_s);
  number("chassis_filter_tau_s", config.chassis_filter_tau_s);
  number("chassis_stale_s", config.chassis_stale_s);
  number("dmmc_stale_s", config.dmmc_stale_s);
  number("solver_warning_ms", config.solver_warning_ms);
  number("solver_deadline_ms", config.solver_deadline_ms);
  integer("fallback_after_failures", config.fallback_after_failures);
  integer("stop_after_failures", config.stop_after_failures);
  number("stop_after_failure_s", config.stop_after_failure_s);
  number("fallback_kp", config.fallback_kp);
  number("fallback_kd", config.fallback_kd);
  number("fallback_disturbance_gain", config.fallback_disturbance_gain);
  boolean("require_osqp", config.require_osqp);
  integer("qp_max_iterations", config.qp_max_iterations);
  number("qp_eps_abs", config.qp_eps_abs);
  number("qp_eps_rel", config.qp_eps_rel);
  number("qp_rho", config.qp_rho);
  integer("qp_adaptive_rho_interval", config.qp_adaptive_rho_interval);
  integer("qp_check_termination_interval",
          config.qp_check_termination_interval);
  boolean("qp_scaled_termination", config.qp_scaled_termination);
  if (!(config.period_s > 0.0 && config.actuator_tau_s > 0.0 && config.horizon > 0 &&
        config.predicted_min_confidence >= 0.0 && config.predicted_min_confidence <= 1.0 &&
        config.vision_frame_rate_hz > 0.0 &&
        config.vision_decay_start_s >= 0.0 &&
        config.vision_decay_start_s < config.vision_loss_hold_s &&
        config.vision_loss_hold_s < config.vision_loss_safe_s &&
        config.position_soft_limit_m > 0.0 &&
        config.position_safe_limit_m > config.position_soft_limit_m &&
        config.task3_reference_max_velocity_m_s > 0.0 &&
        config.task3_reference_max_acceleration_m_s2 > 0.0 &&
        config.task3_reference_max_jerk_m_s3 > 0.0 &&
        config.task3_settle_position_error_m > 0.0 &&
        config.task3_settle_velocity_m_s > 0.0 &&
        config.task3_settle_dwell_s >= 0.0 &&
        config.task3_settle_theta_tolerance_rad > 0.0 &&
        config.task3_theta_static_rad >= 0.0 &&
        config.task3_theta_margin_rad >= 0.0 &&
        config.task3_rolling_compensation_rad >= 0.0 &&
        config.task3_friction_blend_time_s > 0.0 &&
        config.task3_friction_breakaway_timeout_s > 0.0 &&
        config.task3_braking_deceleration_m_s2 > 0.0 &&
        config.task3_early_brake_position_m > 0.0 &&
        config.task3_early_brake_position_m < 0.05 &&
        config.task3_friction_stationary_enter_velocity_m_s >= 0.0 &&
        config.task3_friction_rolling_enter_velocity_m_s >
            config.task3_friction_stationary_enter_velocity_m_s &&
        config.task3_friction_disable_position_error_m > 0.0 &&
        config.task3_friction_disable_velocity_m_s > 0.0 &&
        config.task3_friction_request_acceleration_m_s2 >= 0.0 &&
        config.qp_eps_abs > 0.0 && config.qp_eps_rel > 0.0 &&
        config.qp_rho > 0.0 &&
        config.qp_adaptive_rho_interval >= 0 &&
        config.qp_check_termination_interval > 0)) {
    throw std::runtime_error("invalid control config limits");
  }
  return config;
}

UdpReceiver::~UdpReceiver() {
  if (fd_ >= 0) ::close(fd_);
}

bool UdpReceiver::open(const std::string& bind_address, std::uint16_t port) {
  if (fd_ >= 0) ::close(fd_);
  fd_ = ::socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd_ < 0) {
    last_error_ = std::strerror(errno);
    return false;
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  if (::inet_pton(AF_INET, bind_address.c_str(), &address.sin_addr) != 1 ||
      ::bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    last_error_ = std::strerror(errno);
    ::close(fd_);
    fd_ = -1;
    return false;
  }
  return true;
}

std::ptrdiff_t UdpReceiver::read(std::uint8_t* data, std::size_t capacity) {
  const ssize_t count = ::recv(fd_, data, capacity, 0);
  if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK) last_error_ = std::strerror(errno);
  return count;
}

SerialPort::~SerialPort() { close(); }

bool SerialPort::open(const std::string& path, int baud) {
  close();
  fd_ = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
  if (fd_ < 0) {
    last_error_ = std::strerror(errno);
    return false;
  }
  termios settings{};
  try {
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
      throw std::runtime_error("serial device is already locked: " +
                               std::string(std::strerror(errno)));
    }
    if (::ioctl(fd_, TIOCEXCL) != 0) {
      throw std::runtime_error("cannot claim exclusive serial access: " +
                               std::string(std::strerror(errno)));
    }
    if (::tcgetattr(fd_, &settings) != 0) throw std::runtime_error(std::strerror(errno));
    ::cfmakeraw(&settings);
    const speed_t speed = baud_flag(baud);
    ::cfsetispeed(&settings, speed);
    ::cfsetospeed(&settings, speed);
    settings.c_cflag |= CLOCAL | CREAD;
    settings.c_cc[VMIN] = 0;
    settings.c_cc[VTIME] = 0;
    if (::tcsetattr(fd_, TCSANOW, &settings) != 0) throw std::runtime_error(std::strerror(errno));
    ::tcflush(fd_, TCIOFLUSH);
    return true;
  } catch (const std::exception& error) {
    last_error_ = error.what();
    close();
    return false;
  }
}

void SerialPort::close() {
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
}

std::ptrdiff_t SerialPort::read(std::uint8_t* data, std::size_t capacity) {
  const ssize_t count = ::read(fd_, data, capacity);
  if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
    last_error_ = std::strerror(errno);
    close();
  }
  return count;
}

bool SerialPort::write_all(const std::uint8_t* data, std::size_t size) {
  std::size_t offset = 0;
  while (offset < size) {
    const ssize_t count = ::write(fd_, data + offset, size - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      pollfd descriptor{fd_, POLLOUT, 0};
      if (::poll(&descriptor, 1, 2) <= 0) {
        last_error_ = "serial write deadline exceeded";
        return false;
      }
    } else {
      last_error_ = count < 0 ? std::strerror(errno) : "zero-length serial write";
      return false;
    }
  }
  return true;
}

bool CsvLogger::open(const std::string& path) {
  if (path.empty()) return false;
  const std::filesystem::path file(path);
  if (!file.parent_path().empty()) std::filesystem::create_directories(file.parent_path());
  stream_.open(file);
  if (!stream_) return false;
  rows_ = 0;
  have_safety_state_ = false;
  last_safety_latched_ = false;
  last_safety_event_id_ = 0;
  last_wire_control_state_ = 0;
  stream_ << "time_s,command_id,source_frame_id,state,wire_control_state,wire_flags,"
             "dmmc_controller_state,safety_latched,safety_event_id,last_stop_reason,"
             "task3_stage,planned_x_m,planned_v_m_s,planned_a_m_s2,"
             "settle_position_ok,settle_velocity_ok,settle_theta_ok,settle_elapsed_ms,"
             "task3_early_braking,"
             "theta_mpc_deg,theta_bias_deg,theta_friction_deg,theta_command_deg,"
             "friction_mode,friction_direction,theta_actual_deg,"
             "x_m,v_m_s,d_m_s2,x_ref_m,u_cmd_m_s2,"
             "theta_cmd_rad,theta_actual_rad,motor_position_rad,motor_velocity_rad_s,"
             "motor_torque_nm,a_actual_m_s2,a_ref_m_s2,v_actual_m_s,v_ref_m_s,jerk_ref_m_s3,"
             "track_error_m,track_quality,chassis_events,vision_age_ms,vision_capture_age_ms,"
             "chassis_age_ms,dmmc_age_ms,mpc_ms,qp_build_ms,qp_setup_ms,qp_update_ms,"
             "qp_backend_solve_ms,qp_iteration_us,solver_iterations,slack_m,solver_failures,"
             "fallback,slow,stop,tube_faults,motion_phase,track_segment,reason,prediction_m\n";
  return true;
}

void CsvLogger::write(double now_s, const ControlOutput& output, const TubeStatus* tube,
                      const ChassisState* chassis) {
  if (!stream_) return;
  const std::uint8_t wire_control_state =
      protocol::control_state_to_mc02(output.command.control_state);
  stream_ << std::fixed << std::setprecision(9) << now_s << ',' << output.command.command_id << ','
          << output.command.source_frame_id << ',' << static_cast<int>(output.command.control_state)
          << ',' << static_cast<int>(wire_control_state)
          << ',' << static_cast<int>(output.command.flags)
          << ',' << (tube ? static_cast<int>(tube->state) : -1)
          << ',' << (output.safety_latched ? 1 : 0)
          << ',' << output.safety_event_id
          << ',' << output.last_stop_reason
          << ',' << output.task3_stage
          << ',' << output.reference.position_m
          << ',' << output.reference.velocity_m_s
          << ',' << output.reference.acceleration_m_s2
          << ',' << (output.settle_position_ok ? 1 : 0)
          << ',' << (output.settle_velocity_ok ? 1 : 0)
          << ',' << (output.settle_theta_ok ? 1 : 0)
          << ',' << output.settle_elapsed_ms
          << ',' << (output.task3_early_braking ? 1 : 0)
          << ',' << output.theta_mpc_rad * kRadiansToDegrees
          << ',' << output.theta_bias_rad * kRadiansToDegrees
          << ',' << output.theta_friction_rad * kRadiansToDegrees
          << ',' << output.command.theta_cmd_rad * kRadiansToDegrees
          << ',' << friction_mode_name(output.friction_mode)
          << ',' << output.friction_direction
          << ',' << (tube ? tube->theta_actual_rad * kRadiansToDegrees : 0.0)
          << ',' << output.estimate.position_m << ',' << output.estimate.velocity_m_s << ','
          << output.estimate.disturbance_m_s2 << ',' << output.reference.position_m << ','
          << output.u_command_m_s2 << ',' << output.command.theta_cmd_rad << ','
          << (tube ? tube->theta_actual_rad : 0.0) << ','
          << (tube ? tube->motor_position_rad : 0.0) << ','
          << (tube ? tube->motor_velocity_rad_s : 0.0) << ','
          << (tube ? tube->motor_torque_nm : 0.0) << ',' << output.acceleration_used_m_s2 << ','
          << (chassis ? chassis->acceleration_ref_m_s2 : 0.0) << ','
          << (chassis ? chassis->velocity_actual_m_s : 0.0) << ','
          << (chassis ? chassis->velocity_ref_m_s : 0.0) << ','
          << (chassis ? chassis->jerk_ref_m_s3 : 0.0) << ','
          << (chassis ? chassis->track_error_m : 0.0) << ','
          << (chassis ? chassis->track_quality : 0.0) << ','
          << (chassis ? chassis->events : 0) << ',' << output.vision_age_ms
          << ',' << output.vision_capture_age_ms
          << ',' << output.chassis_age_ms << ',' << output.dmmc_age_ms << ','
          << output.mpc_solve_ms << ',' << output.qp_build_ms << ','
          << output.qp_setup_ms << ','
          << output.qp_update_ms << ',' << output.qp_backend_solve_ms << ','
          << output.qp_iteration_us << ',' << output.solver_iterations << ','
          << output.max_predicted_slack_m << ','
          << output.solver_failures << ',' << (output.used_fallback ? 1 : 0) << ','
          << (output.request_slowdown ? 1 : 0) << ',' << (output.request_stop ? 1 : 0) << ','
          << (tube ? tube->faults : 0) << ','
          << (chassis ? static_cast<int>(chassis->motion_phase) : 0) << ','
          << (chassis ? static_cast<int>(chassis->track_segment) : 0) << ','
          << output.reason << ',';
  for (std::size_t index = 0; index < output.predicted_position_m.size(); ++index) {
    if (index != 0) stream_ << ';';
    stream_ << output.predicted_position_m[index];
  }
  stream_ << '\n';
  const bool safety_transition =
      !have_safety_state_ || output.safety_latched != last_safety_latched_ ||
      output.safety_event_id != last_safety_event_id_ ||
      (wire_control_state != last_wire_control_state_ &&
       (wire_control_state >= 3U || last_wire_control_state_ >= 3U));
  have_safety_state_ = true;
  last_safety_latched_ = output.safety_latched;
  last_safety_event_id_ = output.safety_event_id;
  last_wire_control_state_ = wire_control_state;
  ++rows_;
  if (safety_transition || rows_ % 50 == 0) stream_.flush();
}

}  // namespace nx_control
