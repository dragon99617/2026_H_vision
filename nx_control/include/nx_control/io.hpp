#pragma once

#include "nx_control/types.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>

namespace nx_control {

double monotonic_seconds();
ControlConfig load_config(const std::string& path);

class PersistentSequence {
 public:
  PersistentSequence(const std::string& path, std::uint32_t initial);
  ~PersistentSequence();
  PersistentSequence(const PersistentSequence&) = delete;
  PersistentSequence& operator=(const PersistentSequence&) = delete;

  void prepare();
  std::uint32_t next();
  void checkpoint();

 private:
  [[noreturn]] void throw_system_error(const std::string& message) const;
  std::uint64_t read_state(std::uint32_t initial);
  void reserve();
  void persist(std::uint64_t value);

  std::string path_;
  int fd_ = -1;
  std::uint64_t next_ = 0;
  std::uint64_t reserved_end_ = 0;
  std::uint64_t next_unreserved_ = 0;
};

class UdpReceiver {
 public:
  UdpReceiver() = default;
  ~UdpReceiver();
  UdpReceiver(const UdpReceiver&) = delete;
  UdpReceiver& operator=(const UdpReceiver&) = delete;

  bool open(const std::string& bind_address, std::uint16_t port);
  std::ptrdiff_t read(std::uint8_t* data, std::size_t capacity);
  int fd() const { return fd_; }
  const std::string& last_error() const { return last_error_; }

 private:
  int fd_ = -1;
  std::string last_error_;
};

class SerialPort {
 public:
  SerialPort() = default;
  ~SerialPort();
  SerialPort(const SerialPort&) = delete;
  SerialPort& operator=(const SerialPort&) = delete;

  bool open(const std::string& path, int baud);
  void close();
  std::ptrdiff_t read(std::uint8_t* data, std::size_t capacity);
  bool write_all(const std::uint8_t* data, std::size_t size);
  int fd() const { return fd_; }
  const std::string& last_error() const { return last_error_; }

 private:
  int fd_ = -1;
  std::string last_error_;
};

struct TaskLogContext {
  std::uint64_t run_id = 0;
  std::string task;
  std::string control_mode;
  double target_m = std::numeric_limits<double>::quiet_NaN();
  std::string start_trigger;
  std::string request_id;
  double start_monotonic_s = 0.0;
  double start_unix_s = 0.0;
};

struct RuntimeIoDiagnostics {
  bool vision_udp_open = false;
  std::string vision_udp_last_error;
  std::uint64_t vision_datagrams = 0;
  std::uint64_t vision_decoded_frames = 0;
  std::uint64_t vision_decode_errors = 0;
  std::uint64_t vision_crc_errors = 0;
  std::uint64_t vision_length_errors = 0;
  std::uint64_t vision_discarded_bytes = 0;
  bool serial_connected = false;
  std::string serial_last_error;
  std::uint64_t serial_connect_attempts = 0;
  std::uint64_t serial_connect_failures = 0;
  std::uint64_t serial_connections = 0;
  std::uint64_t serial_read_failures = 0;
  std::uint64_t serial_write_failures = 0;
  std::uint64_t dmmc_decoded_status_frames = 0;
  std::uint64_t dmmc_decoded_chassis_frames = 0;
  std::uint64_t dmmc_unknown_frames = 0;
  std::uint64_t dmmc_crc_errors = 0;
  std::uint64_t dmmc_length_errors = 0;
  std::uint64_t dmmc_discarded_bytes = 0;
};

std::string make_task_log_path(const std::string& directory,
                               const std::string& task,
                               std::uint64_t unix_time_ms,
                               std::uint64_t run_id,
                               std::uint32_t process_id);

class CsvLogger {
 public:
  bool open(const std::string& path);
  void close();
  void set_task_context(const TaskLogContext& context) { context_ = context; }
  const std::string& path() const { return path_; }
  bool is_open() const { return stream_.is_open(); }
  void write(double now_s, const ControlOutput& output, const TubeStatus* tube,
             const ChassisState* chassis);
  void write(double now_s, const ControlOutput& output, const TubeStatus* tube,
             const ChassisState* chassis, const RuntimeIoDiagnostics* io,
             const std::string& event);

 private:
  std::ofstream stream_;
  std::string path_;
  TaskLogContext context_;
  std::size_t rows_ = 0;
  bool have_safety_state_ = false;
  bool last_safety_latched_ = false;
  std::uint64_t last_safety_event_id_ = 0;
  std::uint8_t last_wire_control_state_ = 0;
};

}  // namespace nx_control
