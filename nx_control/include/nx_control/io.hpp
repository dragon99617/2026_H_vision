#pragma once

#include "nx_control/types.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

namespace nx_control {

double monotonic_seconds();
ControlConfig load_config(const std::string& path);

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

class CsvLogger {
 public:
  bool open(const std::string& path);
  void write(double now_s, const ControlOutput& output, const TubeStatus* tube,
             const ChassisState* chassis);

 private:
  std::ofstream stream_;
  std::size_t rows_ = 0;
};

}  // namespace nx_control
