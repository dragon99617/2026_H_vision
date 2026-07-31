#include "nx_control/runtime_command.hpp"

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>

namespace nx_control {
namespace {

bool valid_request_id(const std::string& value) {
  if (value.empty() || value.size() > 64U) return false;
  for (const unsigned char ch : value) {
    const bool valid = (ch >= 'a' && ch <= 'z') ||
                       (ch >= 'A' && ch <= 'Z') ||
                       (ch >= '0' && ch <= '9') || ch == '-' || ch == '_';
    if (!valid) return false;
  }
  return true;
}

std::vector<std::string> tokens(const std::string& payload) {
  std::istringstream stream(payload);
  std::vector<std::string> result;
  std::string token;
  while (stream >> token) result.push_back(token);
  return result;
}

RuntimeCommand invalid_command(const std::string& request_id,
                               const std::string& error_code) {
  RuntimeCommand result;
  result.request_id = request_id;
  result.error_code = error_code;
  return result;
}

}  // namespace

RuntimeCommand parse_runtime_command(const std::string& payload) {
  const std::vector<std::string> parts = tokens(payload);
  if (parts.size() < 2U) return invalid_command("0", "INVALID_COMMAND");
  const std::string& request_id = parts[1];
  if (!valid_request_id(request_id)) {
    return invalid_command("0", "INVALID_REQUEST_ID");
  }

  RuntimeCommand result;
  result.request_id = request_id;
  if (parts[0] == "STATUS" && parts.size() == 2U) {
    result.type = RuntimeCommandType::Status;
    return result;
  }
  if (parts[0] == "START" && parts.size() == 2U) {
    result.type = RuntimeCommandType::Start;
    return result;
  }
  if (parts[0] == "STOP" && parts.size() == 2U) {
    result.type = RuntimeCommandType::Stop;
    return result;
  }
  if (parts[0] == "RESET" && parts.size() == 2U) {
    result.type = RuntimeCommandType::Reset;
    return result;
  }
  if (parts[0] != "TASK" || (parts.size() != 3U && parts.size() != 4U)) {
    return invalid_command(request_id, "INVALID_COMMAND");
  }

  result.type = RuntimeCommandType::Task;
  result.task = parts[2];
  if (parts.size() == 4U) {
    try {
      std::size_t consumed = 0;
      const double target = std::stod(parts[3], &consumed);
      if (consumed != parts[3].size() || !std::isfinite(target)) {
        return invalid_command(request_id, "INVALID_TARGET");
      }
      result.target_cm = target;
    } catch (const std::exception&) {
      return invalid_command(request_id, "INVALID_TARGET");
    }
  }
  return result;
}

std::string json_escape(const std::string& value) {
  std::ostringstream escaped;
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"':
        escaped << "\\\"";
        break;
      case '\\':
        escaped << "\\\\";
        break;
      case '\b':
        escaped << "\\b";
        break;
      case '\f':
        escaped << "\\f";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        if (ch < 0x20U) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<int>(ch) << std::dec;
        } else {
          escaped << static_cast<char>(ch);
        }
    }
  }
  return escaped.str();
}

LocalCommandServer::~LocalCommandServer() { close(); }

bool LocalCommandServer::open(const std::string& path) {
  close();
  if (path.empty()) {
    last_error_ = "command socket path is empty";
    return false;
  }
  sockaddr_un address{};
  if (path.size() >= sizeof(address.sun_path)) {
    last_error_ = "command socket path is too long";
    return false;
  }

  struct stat existing {};
  if (::lstat(path.c_str(), &existing) == 0) {
    if (!S_ISSOCK(existing.st_mode)) {
      last_error_ = "refusing to replace non-socket path: " + path;
      return false;
    }
    const int probe = ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (probe < 0) {
      last_error_ = std::strerror(errno);
      return false;
    }
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1U);
    const bool active =
        ::connect(probe, reinterpret_cast<const sockaddr*>(&address),
                  sizeof(address)) == 0;
    const int probe_error = errno;
    ::close(probe);
    if (active) {
      last_error_ = "another controller is using command socket: " + path;
      return false;
    }
    if (probe_error != ECONNREFUSED && probe_error != ENOENT) {
      last_error_ = std::strerror(probe_error);
      return false;
    }
    if (::unlink(path.c_str()) != 0) {
      last_error_ = std::strerror(errno);
      return false;
    }
  } else if (errno != ENOENT) {
    last_error_ = std::strerror(errno);
    return false;
  }

  fd_ = ::socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd_ < 0) {
    last_error_ = std::strerror(errno);
    return false;
  }
  address = sockaddr_un{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1U);
  if (::bind(fd_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
    last_error_ = std::strerror(errno);
    close();
    return false;
  }
  path_ = path;
  if (::chmod(path.c_str(), 0660) != 0) {
    last_error_ = std::strerror(errno);
    close();
    return false;
  }
  last_error_.clear();
  return true;
}

bool LocalCommandServer::receive(std::string& payload) {
  if (fd_ < 0) return false;
  std::array<char, 512> buffer{};
  socklen_t peer_size = static_cast<socklen_t>(peer_address_.size());
  const ssize_t count = ::recvfrom(
      fd_, buffer.data(), buffer.size() - 1U, 0,
      reinterpret_cast<sockaddr*>(peer_address_.data()), &peer_size);
  if (count < 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      last_error_ = std::strerror(errno);
    }
    return false;
  }
  peer_address_size_ = peer_size;
  have_peer_ = true;
  payload.assign(buffer.data(), static_cast<std::size_t>(count));
  return true;
}

bool LocalCommandServer::reply(const std::string& payload) {
  if (fd_ < 0 || !have_peer_) {
    last_error_ = "no command peer available for reply";
    return false;
  }
  const ssize_t sent = ::sendto(
      fd_, payload.data(), payload.size(), 0,
      reinterpret_cast<const sockaddr*>(peer_address_.data()),
      static_cast<socklen_t>(peer_address_size_));
  have_peer_ = false;
  if (sent != static_cast<ssize_t>(payload.size())) {
    last_error_ = sent < 0 ? std::strerror(errno) : "short command reply";
    return false;
  }
  return true;
}

void LocalCommandServer::close() {
  have_peer_ = false;
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  if (!path_.empty()) {
    ::unlink(path_.c_str());
    path_.clear();
  }
}

}  // namespace nx_control
