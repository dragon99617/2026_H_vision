#include "nx_control/protocol.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace nx_control::protocol {
namespace {

template <typename T>
T read_le(const std::vector<std::uint8_t>& data, std::size_t offset) {
  using Unsigned = typename std::make_unsigned<T>::type;
  Unsigned value = 0;
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    value |= static_cast<Unsigned>(data[offset + index]) << (8U * index);
  }
  T result{};
  std::memcpy(&result, &value, sizeof(T));
  return result;
}

template <typename T>
void append_le(std::vector<std::uint8_t>& output, T value) {
  using Unsigned = typename std::make_unsigned<T>::type;
  Unsigned raw{};
  std::memcpy(&raw, &value, sizeof(T));
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    output.push_back(static_cast<std::uint8_t>((raw >> (8U * index)) & 0xFFU));
  }
}

double cdeg_to_rad(std::int16_t value) {
  return static_cast<double>(value) * 0.01 * 3.14159265358979323846 / 180.0;
}

std::int16_t rad_to_cdeg(double value) {
  const double scaled = value * 180.0 / 3.14159265358979323846 * 100.0;
  return static_cast<std::int16_t>(std::clamp(
      std::lround(scaled), static_cast<long>(std::numeric_limits<std::int16_t>::min()),
      static_cast<long>(std::numeric_limits<std::int16_t>::max())));
}

}  // namespace

std::uint16_t crc16_ccitt_false(const std::uint8_t* data, std::size_t size) {
  std::uint16_t crc = 0xFFFFU;
  for (std::size_t index = 0; index < size; ++index) {
    crc ^= static_cast<std::uint16_t>(data[index]) << 8U;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x8000U) != 0U ? static_cast<std::uint16_t>((crc << 1U) ^ 0x1021U)
                                 : static_cast<std::uint16_t>(crc << 1U);
    }
  }
  return crc;
}

std::vector<Frame> StreamParser::feed(const std::uint8_t* data, std::size_t size) {
  buffer_.insert(buffer_.end(), data, data + size);
  std::vector<Frame> frames;
  while (true) {
    const std::array<std::uint8_t, 2> marker{0xA5, 0x5A};
    auto magic = std::search(buffer_.begin(), buffer_.end(), marker.begin(), marker.end());
    if (magic == buffer_.end()) {
      const bool keep_prefix = !buffer_.empty() && buffer_.back() == marker.front();
      discarded_bytes_ += buffer_.size() - (keep_prefix ? 1U : 0U);
      if (keep_prefix) {
        buffer_.erase(buffer_.begin(), buffer_.end() - 1);
      } else {
        buffer_.clear();
      }
      break;
    }
    if (magic != buffer_.begin()) {
      const auto count = static_cast<std::size_t>(std::distance(buffer_.begin(), magic));
      discarded_bytes_ += count;
      buffer_.erase(buffer_.begin(), magic);
    }
    if (buffer_.size() < 5) break;
    const std::uint16_t payload_size = static_cast<std::uint16_t>(buffer_[3]) |
                                       (static_cast<std::uint16_t>(buffer_[4]) << 8U);
    if (payload_size > 128U) {
      ++length_errors_;
      ++discarded_bytes_;
      buffer_.erase(buffer_.begin());
      continue;
    }
    const std::size_t frame_size = 2U + 1U + 2U + payload_size + 2U;
    if (buffer_.size() < frame_size) break;
    const std::uint16_t expected = static_cast<std::uint16_t>(buffer_[frame_size - 2U]) |
                                   (static_cast<std::uint16_t>(buffer_[frame_size - 1U]) << 8U);
    const std::uint16_t actual = crc16_ccitt_false(buffer_.data() + 2U, frame_size - 4U);
    if (actual != expected) {
      ++crc_errors_;
      ++discarded_bytes_;
      buffer_.erase(buffer_.begin());
      continue;
    }
    Frame frame;
    frame.type = buffer_[2];
    frame.payload.assign(buffer_.begin() + 5, buffer_.begin() + 5 + payload_size);
    frames.push_back(std::move(frame));
    buffer_.erase(buffer_.begin(), buffer_.begin() + frame_size);
  }
  return frames;
}

std::optional<VisionMeasurement> decode_vision(const Frame& frame, double receive_time_s) {
  if (!((frame.type == kVisionV2 && frame.payload.size() == 11U) ||
        (frame.type == kVisionV3 && frame.payload.size() == 15U))) {
    return std::nullopt;
  }
  const std::uint8_t raw_status = frame.payload[4];
  if (raw_status > static_cast<std::uint8_t>(VisionStatus::Predicted)) return std::nullopt;
  VisionMeasurement result;
  result.frame_id = read_le<std::uint32_t>(frame.payload, 0);
  result.status = static_cast<VisionStatus>(raw_status);
  result.position_m = static_cast<double>(read_le<std::int16_t>(frame.payload, 5)) / 10000.0;
  result.ball_confidence = static_cast<double>(read_le<std::uint16_t>(frame.payload, 7)) / 1000.0;
  result.tube_confidence = static_cast<double>(read_le<std::uint16_t>(frame.payload, 9)) / 1000.0;
  if (std::abs(result.position_m) > 0.13 || result.ball_confidence > 1.0 ||
      result.tube_confidence > 1.0) {
    return std::nullopt;
  }
  result.receive_time_s = receive_time_s;
  if (frame.type == kVisionV3) {
    result.capture_time_ms = read_le<std::uint32_t>(frame.payload, 11);
    result.has_capture_time = true;
  }
  if (result.status == VisionStatus::Lost &&
      (result.position_m != 0.0 || result.ball_confidence != 0.0 || result.tube_confidence != 0.0)) {
    return std::nullopt;
  }
  return result;
}

std::optional<TubeStatus> decode_tube_status(const Frame& frame, double receive_time_s) {
  if (frame.type != kTubeStatusV3 || frame.payload.size() != 32U) return std::nullopt;
  TubeStatus value;
  value.sequence = read_le<std::uint32_t>(frame.payload, 0);
  value.dmmc_time_ms = read_le<std::uint32_t>(frame.payload, 4);
  value.theta_target_rad = cdeg_to_rad(read_le<std::int16_t>(frame.payload, 8));
  value.theta_reference_rad = cdeg_to_rad(read_le<std::int16_t>(frame.payload, 10));
  value.theta_actual_rad = cdeg_to_rad(read_le<std::int16_t>(frame.payload, 12));
  value.motor_position_rad = static_cast<double>(read_le<std::int32_t>(frame.payload, 14)) / 1000.0;
  value.motor_velocity_rad_s = static_cast<double>(read_le<std::int16_t>(frame.payload, 18)) / 100.0;
  value.motor_torque_nm = static_cast<double>(read_le<std::int16_t>(frame.payload, 20)) / 1000.0;
  value.faults = read_le<std::uint16_t>(frame.payload, 22);
  value.can_age_ms = read_le<std::uint16_t>(frame.payload, 24);
  value.usb_crc_errors = read_le<std::uint16_t>(frame.payload, 26);
  value.control_age_ms = read_le<std::uint16_t>(frame.payload, 28);
  value.state = frame.payload[30];
  value.flags = frame.payload[31];
  value.receive_time_s = receive_time_s;
  if (std::abs(value.theta_actual_rad) > 10.0 * 3.14159265358979323846 / 180.0) {
    return std::nullopt;
  }
  return value;
}

std::optional<ChassisState> decode_chassis_state(const Frame& frame, double receive_time_s) {
  if (frame.type != kChassisStateV1 || frame.payload.size() != 32U) return std::nullopt;
  ChassisState value;
  value.sequence = read_le<std::uint32_t>(frame.payload, 0);
  value.chassis_time_ms = read_le<std::uint32_t>(frame.payload, 4);
  value.velocity_ref_m_s = static_cast<double>(read_le<std::int16_t>(frame.payload, 8)) / 1000.0;
  value.acceleration_ref_m_s2 = static_cast<double>(read_le<std::int16_t>(frame.payload, 10)) / 1000.0;
  value.jerk_ref_m_s3 = static_cast<double>(read_le<std::int16_t>(frame.payload, 12)) / 1000.0;
  value.velocity_actual_m_s = static_cast<double>(read_le<std::int16_t>(frame.payload, 14)) / 1000.0;
  value.acceleration_actual_m_s2 = static_cast<double>(read_le<std::int16_t>(frame.payload, 16)) / 1000.0;
  value.track_error_m = static_cast<double>(read_le<std::int16_t>(frame.payload, 18)) / 10000.0;
  value.track_quality = static_cast<double>(read_le<std::uint16_t>(frame.payload, 20)) / 1000.0;
  if (value.track_quality > 1.0) return std::nullopt;
  if (frame.payload[22] > static_cast<std::uint8_t>(MotionPhase::Decel) || frame.payload[23] > 4U) {
    return std::nullopt;
  }
  value.motion_phase = static_cast<MotionPhase>(frame.payload[22]);
  value.track_segment = static_cast<TrackSegment>(frame.payload[23]);
  value.events = read_le<std::uint16_t>(frame.payload, 24);
  value.faults = read_le<std::uint16_t>(frame.payload, 26);
  value.ttl_ms = read_le<std::uint16_t>(frame.payload, 28);
  value.yaw_rate_rad_s = static_cast<double>(read_le<std::int16_t>(frame.payload, 30)) / 1000.0;
  value.receive_time_s = receive_time_s;
  return value;
}

std::vector<std::uint8_t> encode_control_command(const ControlCommand& command) {
  std::vector<std::uint8_t> output{0xA5, 0x5A, kControlV3, 21, 0};
  append_le(output, command.command_id);
  append_le(output, command.source_frame_id);
  append_le(output, command.nx_time_ms);
  append_le(output, rad_to_cdeg(command.theta_cmd_rad));
  append_le(output, rad_to_cdeg(command.theta_rate_limit_rad_s));
  append_le(output, command.ttl_ms);
  output.push_back(static_cast<std::uint8_t>(command.control_state));
  output.push_back(command.flags);
  output.push_back(0U);
  const std::uint16_t crc = crc16_ccitt_false(output.data() + 2, output.size() - 2);
  append_le(output, crc);
  return output;
}

}  // namespace nx_control::protocol
