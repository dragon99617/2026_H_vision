#pragma once

#include "nx_control/types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace nx_control::protocol {

constexpr std::uint8_t kVisionV2 = 0x02;
constexpr std::uint8_t kVisionV3 = 0x03;
constexpr std::uint8_t kControlV3 = 0x80;
constexpr std::uint8_t kTubeStatusV3 = 0x90;
constexpr std::uint8_t kChassisStateV1 = 0x91;
constexpr std::size_t kVisionV2Length = 18;
constexpr std::size_t kVisionV3Length = 22;
constexpr std::size_t kControlV3Length = 28;

std::uint16_t crc16_ccitt_false(const std::uint8_t* data, std::size_t size);
inline std::uint16_t crc16_ccitt_false(const std::vector<std::uint8_t>& data) {
  return crc16_ccitt_false(data.data(), data.size());
}

struct Frame {
  std::uint8_t type = 0;
  std::vector<std::uint8_t> payload;
};

class StreamParser {
 public:
  std::vector<Frame> feed(const std::uint8_t* data, std::size_t size);
  std::vector<Frame> feed(const std::vector<std::uint8_t>& data) {
    return feed(data.data(), data.size());
  }
  std::uint64_t crc_errors() const { return crc_errors_; }
  std::uint64_t length_errors() const { return length_errors_; }
  std::uint64_t discarded_bytes() const { return discarded_bytes_; }

 private:
  std::vector<std::uint8_t> buffer_;
  std::uint64_t crc_errors_ = 0;
  std::uint64_t length_errors_ = 0;
  std::uint64_t discarded_bytes_ = 0;
};

class Mc02StreamParser {
 public:
  std::vector<Frame> feed(const std::uint8_t* data, std::size_t size);
  std::vector<Frame> feed(const std::vector<std::uint8_t>& data) {
    return feed(data.data(), data.size());
  }
  std::uint64_t crc_errors() const { return crc_errors_; }
  std::uint64_t length_errors() const { return length_errors_; }
  std::uint64_t discarded_bytes() const { return discarded_bytes_; }

 private:
  std::vector<std::uint8_t> buffer_;
  std::uint64_t crc_errors_ = 0;
  std::uint64_t length_errors_ = 0;
  std::uint64_t discarded_bytes_ = 0;
};

std::optional<VisionMeasurement> decode_vision(const Frame& frame, double receive_time_s);
std::optional<TubeStatus> decode_tube_status(const Frame& frame, double receive_time_s);
std::optional<ChassisState> decode_chassis_state(const Frame& frame, double receive_time_s);
std::vector<std::uint8_t> encode_tube_control_v3(const TubeControlV3Command& command);
std::vector<std::uint8_t> encode_control_command(const ControlCommand& command);

}  // namespace nx_control::protocol
