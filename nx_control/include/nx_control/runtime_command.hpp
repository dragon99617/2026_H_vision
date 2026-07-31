#pragma once

#include <array>
#include <optional>
#include <string>

namespace nx_control {

enum class RuntimeCommandType { Invalid, Status, Task, Start, Stop, Reset };

struct RuntimeCommand {
  RuntimeCommandType type = RuntimeCommandType::Invalid;
  std::string request_id;
  std::string task;
  std::optional<double> target_cm;
  std::string error_code;
};

RuntimeCommand parse_runtime_command(const std::string& payload);
std::string json_escape(const std::string& value);

class LocalCommandServer {
 public:
  LocalCommandServer() = default;
  ~LocalCommandServer();
  LocalCommandServer(const LocalCommandServer&) = delete;
  LocalCommandServer& operator=(const LocalCommandServer&) = delete;

  bool open(const std::string& path);
  bool receive(std::string& payload);
  bool reply(const std::string& payload);
  const std::string& last_error() const { return last_error_; }

 private:
  void close();

  int fd_ = -1;
  std::string path_;
  std::string last_error_;
  std::array<unsigned char, 256> peer_address_{};
  unsigned int peer_address_size_ = 0;
  bool have_peer_ = false;
};

}  // namespace nx_control
