#include "nx_control/controller.hpp"
#include "nx_control/friction_compensator.hpp"
#include "nx_control/io.hpp"
#include "nx_control/observer.hpp"
#include "nx_control/pid.hpp"
#include "nx_control/protocol.hpp"
#include "nx_control/runtime_command.hpp"
#include "nx_control/task_manager.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

void put_u16(std::vector<std::uint8_t>& data, std::size_t offset, std::uint16_t value) {
  data[offset] = static_cast<std::uint8_t>(value);
  data[offset + 1U] = static_cast<std::uint8_t>(value >> 8U);
}

void put_u32(std::vector<std::uint8_t>& data, std::size_t offset, std::uint32_t value) {
  for (std::size_t index = 0; index < 4U; ++index) {
    data[offset + index] = static_cast<std::uint8_t>(value >> (8U * index));
  }
}

void put_i16(std::vector<std::uint8_t>& data, std::size_t offset, std::int16_t value) {
  put_u16(data, offset, static_cast<std::uint16_t>(value));
}

void put_i32(std::vector<std::uint8_t>& data, std::size_t offset, std::int32_t value) {
  put_u32(data, offset, static_cast<std::uint32_t>(value));
}

std::vector<std::uint8_t> mc02_packet(std::uint8_t type,
                                     const std::vector<std::uint8_t>& payload) {
  std::vector<std::uint8_t> packet{0xA5, 0x5A, type,
                                   static_cast<std::uint8_t>(payload.size())};
  packet.insert(packet.end(), payload.begin(), payload.end());
  const auto crc = nx_control::protocol::crc16_ccitt_false(packet);
  packet.push_back(static_cast<std::uint8_t>(crc));
  packet.push_back(static_cast<std::uint8_t>(crc >> 8U));
  return packet;
}

std::vector<std::uint8_t> vision_v3_packet() {
  std::vector<std::uint8_t> packet{0xA5, 0x5A, 0x03, 15, 0,
                                   0x78, 0x56, 0x34, 0x12,
                                   0x01, 0x8F, 0xFD, 0x6C, 0x03, 0xAF, 0x03,
                                   0x04, 0x03, 0x02, 0x01};
  const auto crc = nx_control::protocol::crc16_ccitt_false(packet.data() + 2, packet.size() - 2);
  packet.push_back(static_cast<std::uint8_t>(crc & 0xFF));
  packet.push_back(static_cast<std::uint8_t>(crc >> 8));
  return packet;
}

void test_protocol() {
  check(nx_control::protocol::crc16_ccitt_false(
            reinterpret_cast<const std::uint8_t*>("123456789"), 9) == 0x29B1,
        "CRC standard vector");
  const auto packet = vision_v3_packet();
  check(packet.size() == nx_control::protocol::kVisionV3Length, "tube-v3 fixed length");
  nx_control::protocol::StreamParser parser;
  const std::vector<std::uint8_t> noise{0x00, 0xA5, 0x01};
  check(parser.feed(noise).empty(), "parser holds partial/noisy input");
  std::vector<std::uint8_t> combined{0xA5, 0x5A};
  combined.insert(combined.end(), packet.begin() + 2, packet.end());
  const auto frames = parser.feed(combined);
  check(frames.size() == 1, "parser resynchronizes");
  if (!frames.empty()) {
    const auto vision = nx_control::protocol::decode_vision(frames.front(), 10.0);
    check(vision.has_value(), "decode tube-v3");
    if (vision) {
      check(vision->frame_id == 0x12345678U, "tube-v3 frame id");
      check(std::abs(vision->position_m + 0.0625) < 1e-12, "tube-v3 signed position");
      check(vision->capture_time_ms == 0x01020304U, "tube-v3 capture timestamp");
    }
  }
  nx_control::protocol::StreamParser fragmented;
  check(fragmented.feed(packet.data(), 1).empty(), "parser preserves split magic prefix");
  check(fragmented.feed(packet.data() + 1, packet.size() - 1).size() == 1,
        "parser accepts frame split inside magic");
  nx_control::ControlCommand command;
  command.command_id = 9;
  command.theta_cmd_rad = 0.01;
  command.theta_rate_limit_rad_s = 0.8;
  command.ttl_ms = 60;
  const auto encoded = nx_control::protocol::encode_control_command(command);
  check(encoded.size() == nx_control::protocol::kControlV3Length, "control-v3 fixed length");

  nx_control::TubeControlV3Command hold_command;
  hold_command.command_id = 10;
  hold_command.source_frame_id = 20;
  hold_command.nx_time_ms = 30;
  hold_command.theta_cmd_cdeg = 20;
  hold_command.theta_rate_limit_cdeg_s = 200;
  hold_command.ttl_ms = 200;
  hold_command.control_state = 1;
  hold_command.flags = 0x03U;
  hold_command.reserved = 0;
  const auto encoded_hold = nx_control::protocol::encode_tube_control_v3(hold_command);
  check(encoded_hold.size() == nx_control::protocol::kControlV3Length,
        "HOLD control-v3 fixed length");
  check(encoded_hold[4] == 10U && encoded_hold[8] == 20U && encoded_hold[12] == 30U,
        "HOLD control-v3 identifiers and timestamp");
  check(encoded_hold[16] == 0x14U && encoded_hold[17] == 0U,
        "HOLD control-v3 +20 cdeg angle");
  check(encoded_hold[18] == 0xC8U && encoded_hold[19] == 0x00U,
        "HOLD control-v3 200 cdeg/s rate limit");
  check(encoded_hold[20] == 0xC8U && encoded_hold[21] == 0U,
        "HOLD control-v3 200ms TTL");
  check(encoded_hold[22] == 1U && encoded_hold[23] == 0x03U &&
            encoded_hold[24] == 0U && encoded_hold[25] == 0U,
        "HOLD control-v3 state, first flags, and reserved");
  check(nx_control::protocol::crc16_ccitt_false(encoded_hold.data(), encoded_hold.size() - 2U) ==
            (static_cast<std::uint16_t>(encoded_hold[26]) |
             (static_cast<std::uint16_t>(encoded_hold[27]) << 8U)),
        "HOLD control-v3 CRC");

  nx_control::ControlCommand fault_command;
  fault_command.command_id = 1;
  fault_command.theta_rate_limit_rad_s = 2.0 * 3.14159265358979323846 / 180.0;
  fault_command.ttl_ms = 60;
  fault_command.control_state = nx_control::TaskState::Fault;
  fault_command.flags = 0x08U;
  const std::vector<std::uint8_t> expected_control{
      0xA5, 0x5A, 0x80, 0x16, 0x01, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0xC8, 0x00, 0x3C, 0x00, 0x04, 0x08,
      0x00, 0x00, 0x6B, 0x58};
  check(nx_control::protocol::encode_control_command(fault_command) == expected_control,
        "MC02 control golden vector");

  const std::array<std::pair<nx_control::TaskState, std::uint8_t>, 10> state_mapping{{
      {nx_control::TaskState::Idle, 0U},
      {nx_control::TaskState::StandbyHold, 1U},
      {nx_control::TaskState::StaticMove, 2U},
      {nx_control::TaskState::HoldCenter, 2U},
      {nx_control::TaskState::HoldTarget, 2U},
      {nx_control::TaskState::VehicleAccel, 2U},
      {nx_control::TaskState::VehicleCruise, 2U},
      {nx_control::TaskState::VehicleDecel, 2U},
      {nx_control::TaskState::Safe, 3U},
      {nx_control::TaskState::Fault, 4U},
  }};
  for (const auto& mapping : state_mapping) {
    nx_control::ControlCommand mapped_command;
    mapped_command.control_state = mapping.first;
    const auto mapped_packet = nx_control::protocol::encode_control_command(mapped_command);
    check(mapped_packet[22] == mapping.second, "NX task state maps to MC02 control state");
  }

  std::vector<std::uint8_t> status_payload(52U, 0U);
  put_u32(status_payload, 0, 0x01020304U);
  put_u32(status_payload, 4, 1234U);
  put_u32(status_payload, 8, 9U);
  put_i16(status_payload, 12, 100);
  put_i16(status_payload, 14, -50);
  put_i16(status_payload, 16, 25);
  put_i32(status_payload, 18, 1234);
  put_i32(status_payload, 22, -2500);
  put_i16(status_payload, 26, 750);
  status_payload[28] = 3U;
  status_payload[29] = 4U;
  put_u32(status_payload, 30, 0x00000300U);
  put_u16(status_payload, 34, 12U);
  put_u16(status_payload, 36, 34U);
  put_u16(status_payload, 38, 56U);
  put_u16(status_payload, 40, 78U);
  put_u16(status_payload, 50, 0x000CU);
  const auto status_packet = mc02_packet(nx_control::protocol::kTubeStatusV3, status_payload);

  nx_control::protocol::Mc02StreamParser mc02_parser;
  const std::vector<std::uint8_t> mc02_noise{0x00, 0xA5, 0x00};
  check(mc02_parser.feed(mc02_noise).empty(), "MC02 parser discards noise");
  check(mc02_parser.feed(status_packet.data(), 1U).empty(), "MC02 parser holds split magic");
  const auto status_frames = mc02_parser.feed(status_packet.data() + 1U, status_packet.size() - 1U);
  check(status_frames.size() == 1U, "MC02 parser accepts status frame");
  if (!status_frames.empty()) {
    const auto status = nx_control::protocol::decode_tube_status(status_frames.front(), 10.0);
    check(status.has_value(), "decode MC02 status");
    if (status) {
      check(status->sequence == 0x01020304U && status->dmmc_time_ms == 1234U,
            "decode MC02 status identifiers");
      check(std::abs(status->theta_actual_rad -
                     0.25 * 3.14159265358979323846 / 180.0) < 1e-12,
            "decode MC02 tube angle");
      check(std::abs(status->motor_position_rad - 1.234) < 1e-12 &&
                std::abs(status->motor_velocity_rad_s + 2.5) < 1e-12 &&
                std::abs(status->motor_torque_nm - 0.75) < 1e-12,
            "decode MC02 motor feedback");
      check(status->faults == 0x00000300U && status->can_age_ms == 12U &&
                status->control_age_ms == 34U && status->usb_crc_errors == 78U &&
                status->state == 3U && status->flags == 0x0CU,
            "decode MC02 status diagnostics");
    }
  }

  auto bad_status_packet = status_packet;
  bad_status_packet[12] ^= 0x01U;
  check(mc02_parser.feed(bad_status_packet).empty() && mc02_parser.crc_errors() == 1U,
        "MC02 parser rejects bad CRC");

  std::vector<std::uint8_t> chassis_payload(40U, 0U);
  put_u32(chassis_payload, 0, 11U);
  put_u32(chassis_payload, 4, 2000U);
  put_u32(chassis_payload, 8, 2010U);
  put_i32(chassis_payload, 12, 1000);
  put_i32(chassis_payload, 16, -2000);
  put_i32(chassis_payload, 20, 3000);
  put_i32(chassis_payload, 24, 4000);
  put_i32(chassis_payload, 28, -5000);
  chassis_payload[32] = 3U;
  chassis_payload[33] = 2U;
  chassis_payload[34] = 255U;
  chassis_payload[35] = 0x5AU;
  put_u16(chassis_payload, 36, 100U);
  put_u16(chassis_payload, 38, 0x1234U);
  const auto chassis_packet =
      mc02_packet(nx_control::protocol::kChassisStateV1, chassis_payload);
  const auto chassis_frames = mc02_parser.feed(chassis_packet);
  check(chassis_frames.size() == 1U, "MC02 parser accepts chassis frame");
  if (!chassis_frames.empty()) {
    const auto chassis = nx_control::protocol::decode_chassis_state(chassis_frames.front(), 11.0);
    check(chassis.has_value(), "decode MC02 chassis state");
    if (chassis) {
      check(chassis->sequence == 11U && chassis->chassis_time_ms == 2000U,
            "decode MC02 chassis identifiers");
      check(std::abs(chassis->velocity_ref_m_s - 1.0) < 1e-12 &&
                std::abs(chassis->acceleration_ref_m_s2 + 2.0) < 1e-12 &&
                std::abs(chassis->jerk_ref_m_s3 - 3.0) < 1e-12 &&
                std::abs(chassis->velocity_actual_m_s - 4.0) < 1e-12 &&
                std::abs(chassis->acceleration_actual_m_s2 + 5.0) < 1e-12,
            "decode MC02 chassis dynamics");
      check(chassis->motion_phase == nx_control::MotionPhase::Curve &&
                chassis->track_segment == nx_control::TrackSegment::BC &&
                chassis->track_quality == 1.0 && chassis->events == 0x5AU &&
                chassis->ttl_ms == 100U && chassis->faults == 0x1234U,
            "decode MC02 chassis metadata");
    }
  }
}

void test_delayed_observer() {
  nx_control::ControlConfig config;
  config.innovation_gate_sigma = 20.0;
  nx_control::DelayedKalmanObserver current(config);
  nx_control::DelayedKalmanObserver delayed(config);
  current.reset(1.0);
  delayed.reset(1.0);
  current.predict(1.02, 0.2, 0.1);
  current.update_position(1.02, 0.004, 1.0, 1.0, nx_control::VisionStatus::Measured);
  current.predict(1.04, 0.2, 0.1);
  delayed.predict(1.02, 0.2, 0.1);
  delayed.predict(1.04, 0.2, 0.1);
  check(delayed.update_position(1.02, 0.004, 1.0, 1.0,
                                nx_control::VisionStatus::Measured),
        "accept delayed measurement");
  check(std::abs(current.state().position_m - delayed.state().position_m) < 1e-9,
        "delayed replay equals in-order update");
  check(!delayed.update_position(0.5, 0.0, 1.0, 1.0,
                                 nx_control::VisionStatus::Measured),
        "reject measurement older than history");
  check(!delayed.update_position(1.04, delayed.state().position_m, 0.2, 0.9,
                                 nx_control::VisionStatus::Predicted),
        "reject low-confidence predicted vision");

  nx_control::DelayedKalmanObserver ordered(config);
  nx_control::DelayedKalmanObserver out_of_sequence(config);
  ordered.reset(2.0);
  out_of_sequence.reset(2.0);
  ordered.predict(2.02, 0.0, 0.0);
  ordered.update_position(2.02, 0.002, 1.0, 1.0, nx_control::VisionStatus::Measured);
  ordered.predict(2.03, 0.0, 0.0);
  ordered.update_position(2.03, 0.003, 1.0, 1.0, nx_control::VisionStatus::Measured);
  ordered.predict(2.04, 0.0, 0.0);
  out_of_sequence.predict(2.04, 0.0, 0.0);
  out_of_sequence.update_position(2.02, 0.002, 1.0, 1.0,
                                  nx_control::VisionStatus::Measured);
  out_of_sequence.update_position(2.03, 0.003, 1.0, 1.0,
                                  nx_control::VisionStatus::Measured);
  check(std::abs(ordered.state().position_m - out_of_sequence.state().position_m) < 1e-9,
        "multiple delayed measurements retain earlier updates");
}

void test_vision_position_filter() {
  nx_control::ControlConfig filtered_config;
  filtered_config.vision_position_filter_tau_s = 0.010;
  filtered_config.measurement_sigma_m = 1e-6;
  filtered_config.innovation_gate_sigma = 1e6;
  nx_control::NxController filtered(filtered_config);
  filtered.reset(10.0);

  nx_control::ControlConfig raw_config = filtered_config;
  raw_config.vision_position_filter_tau_s = 0.0;
  nx_control::NxController raw(raw_config);
  raw.reset(10.0);

  auto ingest = [](nx_control::NxController& controller,
                   std::uint32_t frame_id, double time_s,
                   double position_m) {
    nx_control::VisionMeasurement vision;
    vision.frame_id = frame_id;
    vision.status = nx_control::VisionStatus::Measured;
    vision.position_m = position_m;
    vision.ball_confidence = vision.tube_confidence = 1.0;
    vision.capture_time_ms =
        static_cast<std::uint32_t>(std::llround(time_s * 1000.0));
    vision.has_capture_time = true;
    vision.receive_time_s = time_s;
    controller.ingest_vision(vision);
  };

  ingest(filtered, 1U, 10.0, 0.0);
  ingest(raw, 1U, 10.0, 0.0);
  ingest(filtered, 2U, 10.02, 0.010);
  ingest(raw, 2U, 10.02, 0.010);
  const double filtered_position = filtered.observer().state().position_m;
  const double raw_position = raw.observer().state().position_m;
  check(filtered_position > 0.008 && filtered_position < raw_position - 0.0005,
        "vision position low-pass mildly attenuates a frame-to-frame step");

  nx_control::NxController reset_after_gap(filtered_config);
  reset_after_gap.reset(20.0);
  ingest(reset_after_gap, 1U, 20.0, 0.0);
  ingest(reset_after_gap, 2U, 20.20, 0.010);
  check(reset_after_gap.observer().state().position_m > 0.0095,
        "vision position filter resets after the soft-loss interval");
}

void test_remote_clock_sync() {
  nx_control::RemoteClockSynchronizer clock;
  check(std::abs(clock.to_local_seconds(1000, 10.0) - 10.0) < 1e-9,
        "remote clock initializes at receive time");
  const double synchronized = clock.to_local_seconds(1020, 10.03);
  check(synchronized >= 10.019 && synchronized <= 10.03,
        "remote clock removes positive transport jitter");
  clock.reset();
  check(std::abs(clock.to_local_seconds(0, 20.0) - 20.0) < 1e-9,
        "remote clock reset handles MCU reboot");
}

void test_runtime_command_protocol() {
  const auto task6 = nx_control::parse_runtime_command("TASK req_42 6 -7.3");
  check(task6.type == nx_control::RuntimeCommandType::Task &&
            task6.request_id == "req_42" && task6.task == "6" &&
            task6.target_cm.has_value() &&
            std::abs(*task6.target_cm + 7.3) < 1e-12,
        "runtime command parses task 6 target");
  const auto task4 = nx_control::parse_runtime_command("TASK 43 4");
  check(task4.type == nx_control::RuntimeCommandType::Task &&
            task4.task == "4" && !task4.target_cm.has_value(),
        "runtime command preserves separate task 4 selection");
  check(nx_control::parse_runtime_command("STOP 44").type ==
            nx_control::RuntimeCommandType::Stop &&
            nx_control::parse_runtime_command("RESET 45").type ==
                nx_control::RuntimeCommandType::Reset &&
            nx_control::parse_runtime_command("STATUS 46").type ==
                nx_control::RuntimeCommandType::Status,
        "runtime command parses stop, reset, and status verbs");
  check(nx_control::parse_runtime_command("TASK bad/id 6 0").error_code ==
            "INVALID_REQUEST_ID" &&
            nx_control::parse_runtime_command("TASK 47 6 nan").error_code ==
                "INVALID_TARGET",
        "runtime command rejects unsafe request IDs and non-finite targets");
  check(nx_control::json_escape("a\n\"b\\c") == "a\\n\\\"b\\\\c",
        "runtime command JSON response escapes control characters");
}

void test_persistent_sequence() {
  char path[] = "/tmp/nx-control-sequence-XXXXXX";
  const int temporary_fd = ::mkstemp(path);
  check(temporary_fd >= 0, "create temporary sequence file");
  if (temporary_fd < 0) return;
  ::close(temporary_fd);

  {
    nx_control::PersistentSequence sequence(path, 42U);
    sequence.prepare();
    check(sequence.next() == 42U && sequence.next() == 43U,
          "persistent sequence starts at the configured ID and increments");
    bool rejected_second_owner = false;
    try {
      nx_control::PersistentSequence duplicate(path, 1U);
    } catch (const std::exception&) {
      rejected_second_owner = true;
    }
    check(rejected_second_owner, "persistent sequence rejects a concurrent sender");
    sequence.checkpoint();
  }
  {
    nx_control::PersistentSequence sequence(path, 1U);
    check(sequence.next() == 44U,
          "persistent sequence resumes after the last consumed ID");
  }
  ::unlink(path);
}

void test_pid_controller() {
  nx_control::ControlConfig config;
  check(std::abs(config.period_s - 0.020) < 1e-12,
        "NX default control period remains 20 ms (50 Hz)");
  check(std::abs(config.theta_limit_rad -
                 4.0 * 3.14159265358979323846 / 180.0) < 1e-12,
        "NX default angle limit matches DMMC02 4.0 degree hard limit");
  check(std::abs(config.theta_rate_limit_rad_s -
                 2.0 * 3.14159265358979323846 / 180.0) < 1e-12,
        "NX default angle rate limit is 2 degrees per second");
  check(std::abs(config.task3_reverse_balance_rate_limit_rad_s -
                 4.0 * 3.14159265358979323846 / 180.0) < 1e-12,
        "Task 3 reverse-balance rate limit remains 4 degrees per second");
  check(std::abs(config.inner_angle_warning_dwell_s - 0.20) < 1e-12 &&
            std::abs(config.inner_angle_safe_dwell_s - 0.50) < 1e-12,
        "inner-angle watchdog defaults remain 0.2 s warning and 0.5 s SAFE");
  check(std::abs(config.pid_kp_s2 - 10.0) < 1e-12 &&
            std::abs(config.pid_ki_s3 - 0.8) < 1e-12 &&
            std::abs(config.pid_kd_s_inv - 5.0) < 1e-12,
        "PID defaults use the commissioned conservative starting gains");
  check(std::abs(config.hold_enter_position_error_m - 0.004) < 1e-12 &&
            std::abs(config.hold_enter_velocity_m_s - 0.015) < 1e-12 &&
            std::abs(config.hold_exit_position_error_m - 0.008) < 1e-12,
        "HoldTarget deadband defaults are 4 mm, 15 mm/s, and 8 mm");
  nx_control::BallPid pid(config);
  nx_control::ObserverState estimate;
  estimate.position_m = 0.010;
  estimate.velocity_m_s = 0.020;
  estimate.disturbance_m_s2 = 0.020;
  nx_control::ReferencePoint reference;
  reference.position_m = 0.020;
  reference.velocity_m_s = 0.030;
  reference.acceleration_m_s2 = 0.040;
  const nx_control::PidResult result =
      pid.calculate(estimate, reference, 0.050, true);
  check(std::abs(result.position_error_m - 0.010) < 1e-12 &&
            std::abs(result.velocity_error_m_s - 0.010) < 1e-12,
        "PID uses reference-minus-estimate position and velocity errors");
  check(std::abs(result.proportional_m_s2 - 0.100) < 1e-12 &&
            std::abs(result.integral_m_s2 - 0.00016) < 1e-12 &&
            std::abs(result.derivative_m_s2 - 0.050) < 1e-12,
        "PID P/I/D terms use observer velocity instead of differentiating camera position");
  check(std::abs(result.feedforward_m_s2 -
                 (0.050 + 0.040 / config.rolling_lambda)) < 1e-12 &&
            std::abs(result.disturbance_m_s2 +
                     config.pid_disturbance_gain * 0.020 /
                         config.rolling_lambda) < 1e-12,
        "PID retains reference, chassis, and observed-disturbance feedforward");

  const double integral_before_tracking = pid.integral_output_m_s2();
  const nx_control::PidResult tracked = pid.track(0.0, false);
  check(tracked.saturated &&
            pid.integral_output_m_s2() < integral_before_tracking,
        "PID back-calculation unwinds the integral when the command is limited");

  estimate.position_m = -0.050;
  reference.position_m = 0.0;
  const nx_control::PidResult separated =
      pid.calculate(estimate, reference, 0.0, true);
  check(separated.integrator_frozen,
        "PID integral separation freezes integration outside the 3 cm band");
  pid.reset();
  check(std::abs(pid.integral_output_m_s2()) < 1e-12,
        "PID reset removes stored integral state");

  config.pid_integral_output_limit_m_s2 = 0.001;
  nx_control::BallPid limited_pid(config);
  estimate = nx_control::ObserverState{};
  reference = nx_control::ReferencePoint{0.020, 0.0, 0.0};
  bool reported_integral_limit = false;
  for (int iteration = 0; iteration < 20; ++iteration) {
    reported_integral_limit =
        limited_pid.calculate(estimate, reference, 0.0, true)
            .integral_limited ||
        reported_integral_limit;
  }
  check(reported_integral_limit &&
            std::abs(limited_pid.integral_output_m_s2()) <= 0.001 + 1e-12,
        "PID integral contribution is explicitly clamped and diagnosed");
}

void test_deployed_timing_contract() {
  const auto config = nx_control::load_config(
      std::string(NX_CONTROL_TEST_SOURCE_DIR) + "/config/nx-control.conf");
  check(std::abs(config.period_s - 0.020) < 1e-12,
        "nx-control.conf keeps the NX outer loop at 20 ms (50 Hz)");
  check(std::abs(config.theta_limit_rad -
                 4.0 * 3.14159265358979323846 / 180.0) < 1e-12 &&
            std::abs(config.theta_rate_limit_rad_s -
                     2.0 * 3.14159265358979323846 / 180.0) < 1e-12 &&
            std::abs(config.task3_reverse_balance_rate_limit_rad_s -
                     4.0 * 3.14159265358979323846 / 180.0) < 1e-12,
        "deployed config preserves 4 degree clamp and physical 2/4 degree per second limits");
  check(std::abs(config.inner_angle_warning_rad -
                 1.0 * 3.14159265358979323846 / 180.0) < 1e-12 &&
            std::abs(config.inner_angle_safe_rad -
                     2.0 * 3.14159265358979323846 / 180.0) < 1e-12 &&
            std::abs(config.inner_angle_warning_dwell_s - 0.20) < 1e-12 &&
            std::abs(config.inner_angle_safe_dwell_s - 0.50) < 1e-12,
        "deployed config preserves inner-angle thresholds and real-time dwell durations");
}

void test_task_manager() {
  nx_control::ControlConfig config;
  nx_control::TaskManager task(config);
  task.configure(nx_control::TaskMode::Contest3, 0.0, true);
  check(std::abs(task.target_m() - 0.05) < 1e-12,
        "contest task 3 starts toward +5 cm");
  nx_control::TubeStatus tube;
  tube.theta_actual_rad = config.task3_theta_bias_rad;
  nx_control::ObserverState state;

  nx_control::ReferencePoint previous =
      task.update(1.0, state, nullptr, &tube, true);
  check(std::abs(previous.position_m) < 1e-12 &&
            std::abs(previous.velocity_m_s) < 1e-12 &&
            std::abs(previous.acceleration_m_s2) < 1e-12,
        "contest task 3 reference starts continuously at the center");
  bool reference_limits_ok = true;
  bool reference_continuity_ok = true;
  constexpr double sample_dt_s = 0.01;
  for (int sample = 1; sample <= 400; ++sample) {
    const nx_control::ReferencePoint point =
        task.update(1.0 + sample * sample_dt_s, state, nullptr, &tube, true);
    reference_limits_ok =
        reference_limits_ok &&
        std::abs(point.velocity_m_s) <=
            config.task3_reference_max_velocity_m_s + 1e-9 &&
        std::abs(point.acceleration_m_s2) <=
            config.task3_reference_max_acceleration_m_s2 + 1e-9 &&
        std::abs(point.acceleration_m_s2 - previous.acceleration_m_s2) <=
            config.task3_reference_max_jerk_m_s3 * sample_dt_s + 2e-5;
    reference_continuity_ok =
        reference_continuity_ok &&
        std::abs(point.position_m - previous.position_m) <=
            config.task3_reference_max_velocity_m_s * sample_dt_s + 1e-6 &&
        std::abs(point.velocity_m_s - previous.velocity_m_s) <=
            config.task3_reference_max_acceleration_m_s2 * sample_dt_s + 1e-6;
    previous = point;
  }
  check(reference_limits_ok,
        "Task 3 quintic reference respects velocity, acceleration, and jerk limits");
  check(reference_continuity_ok,
        "Task 3 reference position, velocity, and acceleration remain continuous");
  nx_control::TaskManager timed_task(config);
  timed_task.configure(nx_control::TaskMode::Contest3, 0.0, true);
  nx_control::ObserverState tracking_state;
  nx_control::TubeStatus tracking_tube;
  tracking_tube.theta_actual_rad = config.task3_theta_bias_rad;
  nx_control::ReferencePoint tracking_reference =
      timed_task.update(20.0, tracking_state, nullptr, &tracking_tube, true);
  double reference_sequence_elapsed_s = -1.0;
  for (int sample = 1; sample <= 250; ++sample) {
    tracking_state.position_m = tracking_reference.position_m;
    tracking_state.velocity_m_s = tracking_reference.velocity_m_s;
    tracking_reference =
        timed_task.update(20.0 + sample * config.period_s, tracking_state,
                          nullptr, &tracking_tube, true);
    if (timed_task.task3_stage() >= 1 &&
        std::abs(tracking_reference.position_m + 0.05) < 1e-9 &&
        std::abs(tracking_reference.velocity_m_s) < 1e-9) {
      reference_sequence_elapsed_s = sample * config.period_s;
      break;
    }
  }
  check(reference_sequence_elapsed_s > 0.0 &&
            reference_sequence_elapsed_s <= 5.0,
        "Task 3 perfect-tracking reference completes 0 -> +5 -> -5 cm within 5 seconds");

  state = nx_control::ObserverState{0.0400, 0.050, 0.0};
  tube.theta_actual_rad =
      config.task3_theta_bias_rad +
      config.task3_settle_theta_tolerance_rad + 1e-4;
  task.update(5.10, state, nullptr, &tube, true);
  check(task.task3_stage() == 0,
        "Task 3 does not count exactly +4.0 cm as exceeding the threshold");

  state.position_m = 0.0401;
  const nx_control::ReferencePoint switch_reference =
      task.update(5.20, state, nullptr, &tube, false);
  check(task.task3_stage() == 1 &&
            task.settle_position_ok() &&
            task.settle_velocity_ok() &&
            task.settle_theta_ok() &&
            std::abs(task.settle_elapsed_s()) < 1e-12 &&
            std::abs(task.target_m() + 0.05) < 1e-12 &&
            std::abs(switch_reference.position_m -
                     previous.position_m) < 0.05,
        "Task 3 counts >+4 cm immediately without speed, angle, feedback, or dwell requirements");
  const auto return_reference =
      task.update(5.20 + config.period_s, state, nullptr, &tube, true);
  check(std::abs(return_reference.position_m - switch_reference.position_m) <
                1e-4 &&
            return_reference.velocity_m_s < 0.0,
        "Task 3 return reference begins smoothly and heads toward -5 cm");

  state.position_m = -0.05;
  state.velocity_m_s = 0.0;
  tube.theta_actual_rad =
      config.task3_theta_bias_rad +
      config.task3_settle_theta_tolerance_rad + 1e-4;
  task.update(15.50, state, nullptr, &tube, true);
  check(!task.static_sequence_complete() && !task.settle_theta_ok(),
        "Task 3 final -5 cm stage still requires actual tube angle settling");

  tube.theta_actual_rad = config.task3_theta_bias_rad;
  task.update(15.60, state, nullptr, &tube, true);
  check(task.static_sequence_complete() &&
            task.state() == nx_control::TaskState::HoldTarget &&
            std::abs(task.target_m() + 0.05) < 1e-12 &&
            task.target_hold_deadband_active(),
        "contest task 3 holds -5 cm immediately after all final conditions pass");

  state.position_m = -0.044;
  state.velocity_m_s = 0.050;
  task.update(2.0, state, nullptr);
  check(task.state() == nx_control::TaskState::HoldTarget &&
            task.target_hold_deadband_active(),
        "HoldTarget deadband remains active inside the 8 mm exit threshold");
  state.position_m = -0.041;
  task.update(2.1, state, nullptr);
  check(!task.target_hold_deadband_active(),
        "HoldTarget deadband exits when position error exceeds 8 mm");
  state.position_m = -0.047;
  state.velocity_m_s = 0.016;
  task.update(2.2, state, nullptr);
  check(!task.target_hold_deadband_active(),
        "HoldTarget deadband does not enter above 15 mm/s");
  state.velocity_m_s = 0.014;
  task.update(2.3, state, nullptr);
  check(task.target_hold_deadband_active(),
        "HoldTarget deadband enters inside 4 mm below 15 mm/s");

  nx_control::ChassisState chassis;
  chassis.events = 0x0001U;
  chassis.motion_phase = nx_control::MotionPhase::Accel;
  nx_control::ObserverState centered;

  task.configure(nx_control::TaskMode::Contest45, 0.08, false);
  check(task.state() == nx_control::TaskState::Idle,
        "contest task 4/5 can wait for an explicit local start");
  task.update(2.0, centered, &chassis);
  check(task.state() == nx_control::TaskState::Idle,
        "contest task 4/5 ignores chassis start events");
  task.start(2.05);
  check(task.state() == nx_control::TaskState::HoldCenter &&
            std::abs(task.target_m()) < 1e-12,
        "contest task 4/5 starts locally and always targets center");
  chassis.motion_phase = nx_control::MotionPhase::Curve;
  task.update(2.1, centered, &chassis);
  check(task.state() == nx_control::TaskState::HoldCenter,
        "contest task 4/5 ignores chassis motion phase");

  task.configure(nx_control::TaskMode::Contest6, -0.073, true);
  check(task.state() == nx_control::TaskState::HoldTarget &&
            std::abs(task.target_m() + 0.073) < 1e-12,
        "contest task 6 starts at requested target");
  chassis.events = 0U;
  chassis.motion_phase = nx_control::MotionPhase::Decel;
  task.update(3.0, centered, &chassis);
  check(task.state() == nx_control::TaskState::HoldTarget &&
            std::abs(task.target_m() + 0.073) < 1e-12,
        "contest task 6 preserves target without following chassis motion");

  task.configure(nx_control::TaskMode::Contest3, 0.0, false, false);
  chassis.events = 0x0001U;
  task.update(4.0, centered, &chassis);
  check(task.state() == nx_control::TaskState::Idle,
        "keyboard task ignores chassis start event");
  task.start(4.1);
  check(task.state() == nx_control::TaskState::StaticMove &&
            std::abs(task.target_m() - 0.05) < 1e-12,
        "keyboard start begins contest task 3");
  task.start(4.2);
  check(task.state() == nx_control::TaskState::StaticMove &&
            !task.static_sequence_complete(),
        "repeated keyboard start restarts the task");
}

void test_task3_friction_compensator() {
  constexpr double kDegrees =
      3.14159265358979323846 / 180.0;
  nx_control::ControlConfig config;
  config.task3_friction_blend_time_s = 0.04;
  config.task3_friction_breakaway_timeout_s = 0.60;
  nx_control::Task3FrictionCompensator friction(config);

  nx_control::FrictionCompensation result;
  for (int step = 0; step < 30; ++step) {
    result = friction.update(step * config.period_s, true, 0.05, 0.0,
                             0.010, 0.010);
  }
  const double positive_command =
      config.task3_theta_bias_rad + std::atan2(0.010, nx_control::kGravity) +
      result.theta_friction_rad;
  check(result.mode == nx_control::FrictionMode::BreakawayPositive &&
            result.direction == 1 && positive_command > 0.4 * kDegrees,
        "positive stationary breakaway exceeds the measured +0.4 degree threshold");
  const double static_compensation = result.theta_friction_rad;

  for (int step = 30; step < 60; ++step) {
    result = friction.update(step * config.period_s, true, 0.04, 0.011,
                             0.010, 0.010);
  }
  check(result.mode == nx_control::FrictionMode::RollingPositive &&
            result.theta_friction_rad < static_compensation &&
            std::abs(result.theta_friction_rad -
                     config.task3_rolling_compensation_rad) < 2e-5,
        "friction compensation drops from static to rolling after 10 mm/s");

  result = friction.update(1.30, true, 0.04, -0.020, 0.020, -0.010);
  check(result.mode == nx_control::FrictionMode::RollingPositive &&
            result.direction == 1,
        "friction direction follows positive PID/braking acceleration");

  friction.reset();
  result = friction.update(1.40, true, -0.008, 0.0, 0.020, 0.0);
  check(result.mode == nx_control::FrictionMode::BreakawayNegative &&
            result.direction == -1,
        "stationary overshoot follows position error instead of stale PID direction");

  friction.reset();
  result = friction.update(1.50, true, 0.003, 0.090, 0.020, 0.0);
  check(result.mode == nx_control::FrictionMode::RollingNegative &&
            result.direction == -1,
        "stopping-distance guard starts braking before a fast target crossing");

  friction.reset();
  config.task3_friction_breakaway_timeout_s = 0.10;
  nx_control::Task3FrictionCompensator limited_friction(config);
  for (int step = 0; step < 30; ++step) {
    result = limited_friction.update(1.60 + step * config.period_s, true,
                                     0.04, 0.0, 0.010, 0.0);
  }
  check(result.mode == nx_control::FrictionMode::RollingPositive &&
            result.theta_friction_rad <
                config.task3_theta_static_rad +
                    config.task3_theta_margin_rad,
        "stationary breakaway pulse falls back to rolling compensation after timeout");

  for (int step = 0; step < 40; ++step) {
    result = friction.update(2.20 + step * config.period_s, true, 0.001,
                             0.004, 0.0, 0.0);
  }
  check(result.mode == nx_control::FrictionMode::Hold &&
            result.direction == 0 &&
            std::abs(result.theta_friction_rad) < 2e-5 &&
            std::abs(config.task3_theta_bias_rad + result.theta_friction_rad +
                     0.15 * kDegrees) < 2e-5,
        "near-target deadband smoothly withdraws directional friction to the bias");

  friction.reset();
  for (int step = 0; step < 30; ++step) {
    result = friction.update(3.0 + step * config.period_s, true, -0.05,
                             0.0, -0.010, -0.010);
  }
  const double negative_command =
      config.task3_theta_bias_rad + std::atan2(-0.010, nx_control::kGravity) +
      result.theta_friction_rad;
  check(result.mode == nx_control::FrictionMode::BreakawayNegative &&
            result.direction == -1 && negative_command < -0.7 * kDegrees,
        "negative stationary breakaway crosses the measured -0.7 degree threshold");

  result = friction.update(4.0, false, -0.05, 0.0, -0.010, -0.010);
  check(result.mode == nx_control::FrictionMode::Hold &&
            result.direction == 0 &&
            std::abs(result.theta_friction_rad) < 1e-12,
        "inactive and safety paths apply no friction compensation");
}

void test_controller_safety() {
  nx_control::ControlConfig config;
  nx_control::NxController controller(config);
  controller.reset(10.0);
  controller.configure_task(nx_control::TaskMode::HoldCenter, 0.0, true);

  auto ingest_feedback = [&](std::uint32_t sequence, double now_s) {
    nx_control::TubeStatus tube;
    tube.sequence = sequence;
    tube.dmmc_time_ms = static_cast<std::uint32_t>(std::llround(now_s * 1000.0));
    tube.receive_time_s = now_s;
    controller.ingest_tube_status(tube);
    nx_control::ChassisState chassis;
    chassis.sequence = sequence;
    chassis.chassis_time_ms =
        static_cast<std::uint32_t>(std::llround(now_s * 1000.0));
    chassis.receive_time_s = now_s;
    chassis.ttl_ms = 100;
    controller.ingest_chassis_state(chassis);
  };
  auto ingest_vision = [&](std::uint32_t frame_id, double now_s,
                           double position_m = 0.0) {
    nx_control::VisionMeasurement vision;
    vision.frame_id = frame_id;
    vision.status = nx_control::VisionStatus::Measured;
    vision.position_m = position_m;
    vision.ball_confidence = vision.tube_confidence = 1.0;
    vision.receive_time_s = now_s;
    vision.capture_time_ms =
        static_cast<std::uint32_t>(std::llround(now_s * 1000.0));
    vision.has_capture_time = true;
    controller.ingest_vision(vision);
  };

  ingest_feedback(1U, 10.0);
  ingest_vision(1U, 10.0);
  auto output = controller.tick(10.02);
  check(!output.request_stop, "healthy controller does not request stop");

  ingest_feedback(2U, 10.15);
  output = controller.tick(10.15);
  check(!output.request_stop && !output.safety_latched &&
            output.command.control_state == nx_control::TaskState::StandbyHold &&
            std::abs(output.command.theta_cmd_rad) < 1e-12 &&
            output.command.flags == 0x01U &&
            nx_control::protocol::control_state_to_mc02(output.command.control_state) == 1U,
        "100-250 ms vision loss emits enabled zero-degree HOLD without SAFE");

  ingest_feedback(3U, 10.16);
  ingest_vision(2U, 10.16);
  output = controller.tick(10.16);
  check(!output.request_stop && !output.safety_latched &&
            output.command.control_state == nx_control::TaskState::HoldCenter,
        "vision recovery during soft HOLD resumes tracking");

  ingest_feedback(4U, 10.42);
  output = controller.tick(10.42);
  check(output.request_stop && output.safety_latched &&
            output.safety_event_id == 1U &&
            output.command.control_state == nx_control::TaskState::Safe &&
            (output.command.flags & 0x08U) != 0U &&
            output.last_stop_reason == "vision_stale",
        "vision loss beyond 250 ms enters latched SAFE");

  ingest_feedback(5U, 10.43);
  ingest_vision(3U, 10.43);
  output = controller.tick(10.43);
  check(output.safety_latched &&
            output.command.control_state == nx_control::TaskState::Safe,
        "fresh vision cannot automatically clear latched SAFE");

  controller.start_task(10.44);
  ingest_feedback(6U, 10.44);
  ingest_vision(4U, 10.44);
  output = controller.tick(10.44, 900U);
  check(!output.safety_latched &&
            output.command.control_state == nx_control::TaskState::HoldCenter &&
            output.command.flags == 0x03U &&
            output.command.command_id == 900U,
        "operator restart clears SAFE and sends TRACK with flags=0x03");

  ingest_feedback(7U, 10.46);
  ingest_vision(5U, 10.46);
  output = controller.tick(10.46, 901U);
  check(output.command.flags == 0x01U && output.command.command_id == 901U,
        "frames after operator restart return to flags=0x01 and preserve supplied IDs");

  nx_control::ControlConfig boundary_config = config;
  boundary_config.innovation_gate_sigma = 20.0;
  nx_control::NxController boundary_controller(boundary_config);
  boundary_controller.reset(20.0);
  boundary_controller.configure_task(nx_control::TaskMode::HoldCenter, 0.0, true);
  nx_control::TubeStatus tube;
  tube.sequence = 1U;
  tube.dmmc_time_ms = 20000U;
  tube.receive_time_s = 20.0;
  boundary_controller.ingest_tube_status(tube);
  nx_control::ChassisState chassis;
  chassis.sequence = 1U;
  chassis.chassis_time_ms = 20000U;
  chassis.receive_time_s = 20.0;
  chassis.ttl_ms = 100U;
  boundary_controller.ingest_chassis_state(chassis);
  nx_control::VisionMeasurement vision;
  vision.frame_id = 1U;
  vision.status = nx_control::VisionStatus::Measured;
  vision.position_m = boundary_config.position_soft_limit_m + 0.001;
  vision.ball_confidence = vision.tube_confidence = 1.0;
  vision.capture_time_ms = 20000U;
  vision.has_capture_time = true;
  vision.receive_time_s = 20.0;
  boundary_controller.ingest_vision(vision);
  boundary_controller.tick(20.0);
  tube.sequence = 2U;
  tube.dmmc_time_ms = 20150U;
  tube.receive_time_s = 20.15;
  boundary_controller.ingest_tube_status(tube);
  chassis.sequence = 2U;
  chassis.chassis_time_ms = 20150U;
  chassis.receive_time_s = 20.15;
  boundary_controller.ingest_chassis_state(chassis);
  output = boundary_controller.tick(20.15);
  check(output.safety_latched &&
            output.last_stop_reason == "vision_lost_outside_soft_boundary",
        "soft-boundary crossing during vision loss enters SAFE immediately");

  nx_control::ControlConfig prediction_config = config;
  prediction_config.innovation_gate_sigma = 1000.0;
  prediction_config.measurement_sigma_m = 1e-6;
  nx_control::NxController prediction_controller(prediction_config);
  prediction_controller.reset(30.0);
  prediction_controller.configure_task(nx_control::TaskMode::HoldCenter, 0.0, true);
  auto ingest_prediction_sample = [&](std::uint32_t sequence, double now_s,
                                      double position_m) {
    nx_control::TubeStatus prediction_tube;
    prediction_tube.sequence = sequence;
    prediction_tube.dmmc_time_ms =
        static_cast<std::uint32_t>(std::llround(now_s * 1000.0));
    prediction_tube.receive_time_s = now_s;
    prediction_controller.ingest_tube_status(prediction_tube);
    nx_control::ChassisState prediction_chassis;
    prediction_chassis.sequence = sequence;
    prediction_chassis.chassis_time_ms =
        static_cast<std::uint32_t>(std::llround(now_s * 1000.0));
    prediction_chassis.receive_time_s = now_s;
    prediction_chassis.ttl_ms = 100U;
    prediction_controller.ingest_chassis_state(prediction_chassis);
    nx_control::VisionMeasurement prediction_vision;
    prediction_vision.frame_id = sequence;
    prediction_vision.status = nx_control::VisionStatus::Measured;
    prediction_vision.position_m = position_m;
    prediction_vision.ball_confidence = prediction_vision.tube_confidence = 1.0;
    prediction_vision.capture_time_ms =
        static_cast<std::uint32_t>(std::llround(now_s * 1000.0));
    prediction_vision.has_capture_time = true;
    prediction_vision.receive_time_s = now_s;
    prediction_controller.ingest_vision(prediction_vision);
  };
  ingest_prediction_sample(1U, 30.0, 0.1000);
  prediction_controller.tick(30.0);
  ingest_prediction_sample(2U, 30.02, 0.1004);
  prediction_controller.tick(30.02);
  nx_control::TubeStatus prediction_tube;
  prediction_tube.sequence = 3U;
  prediction_tube.dmmc_time_ms = 30130U;
  prediction_tube.receive_time_s = 30.13;
  prediction_controller.ingest_tube_status(prediction_tube);
  nx_control::ChassisState prediction_chassis;
  prediction_chassis.sequence = 3U;
  prediction_chassis.chassis_time_ms = 30130U;
  prediction_chassis.receive_time_s = 30.13;
  prediction_chassis.ttl_ms = 100U;
  prediction_controller.ingest_chassis_state(prediction_chassis);
  output = prediction_controller.tick(30.13);
  check(output.safety_latched &&
            output.last_stop_reason == "vision_lost_predicted_soft_boundary",
        "predicted soft-boundary crossing during vision loss enters SAFE immediately");
}

void test_controller_angle_limit() {
  nx_control::ControlConfig config;
  config.pid_kp_s2 = 10000.0;
  config.innovation_gate_sigma = 1000.0;
  nx_control::NxController controller(config);
  controller.reset(20.0);
  controller.configure_task(nx_control::TaskMode::HoldCenter, 0.0, true);

  nx_control::ControlOutput output;
  double previous_theta_rad = 0.0;
  for (std::uint32_t sequence = 1; sequence <= 120; ++sequence) {
    const double now_s = 20.0 + static_cast<double>(sequence) * config.period_s;
    const auto now_ms = static_cast<std::uint32_t>(std::llround(now_s * 1000.0));

    nx_control::TubeStatus tube;
    tube.sequence = sequence;
    tube.dmmc_time_ms = now_ms;
    tube.receive_time_s = now_s;
    controller.ingest_tube_status(tube);

    nx_control::ChassisState chassis;
    chassis.sequence = sequence;
    chassis.chassis_time_ms = now_ms;
    chassis.receive_time_s = now_s;
    chassis.ttl_ms = 100;
    controller.ingest_chassis_state(chassis);

    nx_control::VisionMeasurement vision;
    vision.frame_id = sequence;
    vision.status = nx_control::VisionStatus::Measured;
    vision.position_m = -0.050;
    vision.ball_confidence = vision.tube_confidence = 1.0;
    vision.capture_time_ms = now_ms;
    vision.has_capture_time = true;
    vision.receive_time_s = now_s;
    controller.ingest_vision(vision);

    output = controller.tick(now_s);
    check(output.command.theta_cmd_rad <= config.theta_limit_rad + 1e-12,
          "controller final output does not exceed 4.0 degrees");
    check(output.command.theta_cmd_rad - previous_theta_rad <=
              config.theta_rate_limit_rad_s * config.period_s + 1e-12,
          "controller final output retains the 2 degrees per second rate limit");
    previous_theta_rad = output.command.theta_cmd_rad;
  }

  check(std::abs(output.command.theta_cmd_rad - config.theta_limit_rad) < 1e-12,
        "controller final output clamps at the 4.0 degree hardware limit");
  const auto packet = nx_control::protocol::encode_control_command(output.command);
  check(packet[16] == 0x90U && packet[17] == 0x01U,
        "tube-control-v3 still encodes 4.0 degrees as 400 cdeg");
  check(packet[18] == 0xC8U && packet[19] == 0x00U,
        "tube-control-v3 encodes the rate limit as 200 cdeg per second");
}

void test_inner_angle_watchdog() {
  nx_control::ControlConfig config;
  nx_control::NxController controller(config);
  controller.reset(65.0);
  controller.configure_task(nx_control::TaskMode::Contest45, 0.0, true);

  constexpr double kThreeDegrees =
      3.0 * 3.14159265358979323846 / 180.0;
  auto tick_with_tracking_error = [&](std::uint32_t sequence, double now_s) {
    const auto now_ms =
        static_cast<std::uint32_t>(std::llround(now_s * 1000.0));
    nx_control::TubeStatus tube;
    tube.sequence = sequence;
    tube.dmmc_time_ms = now_ms;
    tube.theta_reference_rad = kThreeDegrees;
    tube.theta_actual_rad = 0.0;
    tube.receive_time_s = now_s;
    controller.ingest_tube_status(tube);

    nx_control::VisionMeasurement vision;
    vision.frame_id = sequence;
    vision.status = nx_control::VisionStatus::Measured;
    vision.ball_confidence = vision.tube_confidence = 1.0;
    vision.capture_time_ms = now_ms;
    vision.has_capture_time = true;
    vision.receive_time_s = now_s;
    controller.ingest_vision(vision);
    return controller.tick(now_s);
  };

  tick_with_tracking_error(1U, 65.000);
  auto output = tick_with_tracking_error(2U, 65.199);
  check(!output.inner_angle_warning && !output.safety_latched,
        "inner-angle warning does not trigger before 0.2 s despite irregular samples");
  output = tick_with_tracking_error(3U, 65.201);
  check(output.inner_angle_warning && output.request_slowdown &&
            !output.safety_latched,
        "one-degree inner-loop error triggers after 0.2 s of real elapsed time");
  output = tick_with_tracking_error(4U, 65.499);
  check(!output.safety_latched,
        "inner-angle SAFE does not latch before 0.5 s despite irregular samples");
  output = tick_with_tracking_error(5U, 65.501);
  check(output.safety_latched &&
            output.last_stop_reason == "inner_angle_tracking_error" &&
            output.command.control_state == nx_control::TaskState::Safe,
        "two-degree inner-loop error latches SAFE after 0.5 s of real elapsed time");
}

void test_controller_hold_deadband() {
  nx_control::ControlConfig config;
  config.pid_kp_s2 = 10000.0;
  config.hold_enter_position_error_m = 0.10;
  config.hold_enter_velocity_m_s = 0.10;
  nx_control::NxController controller(config);
  controller.reset(70.0);
  controller.configure_task(nx_control::TaskMode::HoldCenter, 0.0, true);

  auto ingest_healthy_sample = [&](std::uint32_t sequence,
                                   double position_m) {
    const double now_s = 70.0 + static_cast<double>(sequence) * config.period_s;
    const auto now_ms = static_cast<std::uint32_t>(std::llround(now_s * 1000.0));

    nx_control::TubeStatus tube;
    tube.sequence = sequence;
    tube.dmmc_time_ms = now_ms;
    tube.receive_time_s = now_s;
    controller.ingest_tube_status(tube);

    nx_control::ChassisState chassis;
    chassis.sequence = sequence;
    chassis.chassis_time_ms = now_ms;
    chassis.receive_time_s = now_s;
    chassis.ttl_ms = 100;
    controller.ingest_chassis_state(chassis);

    nx_control::VisionMeasurement vision;
    vision.frame_id = sequence;
    vision.status = nx_control::VisionStatus::Measured;
    vision.position_m = position_m;
    vision.ball_confidence = vision.tube_confidence = 1.0;
    vision.capture_time_ms = now_ms;
    vision.has_capture_time = true;
    vision.receive_time_s = now_s;
    controller.ingest_vision(vision);
    return now_s;
  };

  nx_control::ControlOutput output;
  for (std::uint32_t sequence = 1; sequence <= 10; ++sequence) {
    output = controller.tick(ingest_healthy_sample(sequence, -0.010));
  }
  const double theta_before_hold_rad = output.command.theta_cmd_rad;
  check(theta_before_hold_rad > 0.0,
        "controller has a nonzero command before entering HoldTarget deadband");

  controller.configure_task(nx_control::TaskMode::HoldTarget, 0.0, true);
  output = controller.tick(ingest_healthy_sample(11, 0.0));
  check(controller.task_manager().target_hold_deadband_active() &&
            output.reason == "hold_deadband" &&
            output.command.control_state == nx_control::TaskState::HoldTarget,
        "controller enters HoldTarget deadband without changing the wire control state");
  check(output.command.theta_cmd_rad > 0.0 &&
            output.command.theta_cmd_rad < theta_before_hold_rad &&
            theta_before_hold_rad - output.command.theta_cmd_rad <=
                config.theta_rate_limit_rad_s * config.period_s + 1e-12,
        "HoldTarget deadband returns theta toward zero through the 2 degree per second limiter");

  for (std::uint32_t sequence = 12; sequence <= 30; ++sequence) {
    output = controller.tick(ingest_healthy_sample(sequence, 0.0));
  }
  check(std::abs(output.command.theta_cmd_rad) < 1e-12 &&
            output.reason == "hold_deadband",
        "HoldTarget deadband settles theta at zero without resuming PID corrections");
}

void test_contest_chassis_gate() {
  nx_control::ControlConfig config;
  auto tick_without_chassis = [&](nx_control::TaskMode mode, double now_s) {
    nx_control::NxController controller(config);
    controller.reset(now_s);
    controller.configure_task(mode, 0.0, true);

    nx_control::TubeStatus tube;
    tube.sequence = 1;
    tube.dmmc_time_ms = static_cast<std::uint32_t>(std::llround(now_s * 1000.0));
    tube.receive_time_s = now_s;
    controller.ingest_tube_status(tube);

    nx_control::VisionMeasurement vision;
    vision.frame_id = 1;
    vision.status = nx_control::VisionStatus::Measured;
    vision.ball_confidence = vision.tube_confidence = 1.0;
    vision.capture_time_ms = static_cast<std::uint32_t>(std::llround(now_s * 1000.0));
    vision.has_capture_time = true;
    vision.receive_time_s = now_s;
    controller.ingest_vision(vision);
    return controller.tick(now_s + config.period_s);
  };

  const auto contest3 = tick_without_chassis(nx_control::TaskMode::Contest3, 30.0);
  check(!contest3.request_stop &&
            contest3.command.control_state == nx_control::TaskState::StaticMove,
        "explicit contest task 3 bypasses chassis-fresh gate");

  const auto legacy_static =
      tick_without_chassis(nx_control::TaskMode::StaticSequence, 40.0);
  check(legacy_static.request_stop && legacy_static.reason == "chassis_stale",
        "legacy static mode still requires fresh chassis");

  const auto contest45 = tick_without_chassis(nx_control::TaskMode::Contest45, 50.0);
  check(!contest45.request_stop &&
            contest45.command.control_state == nx_control::TaskState::HoldCenter &&
            std::abs(contest45.acceleration_used_m_s2) < 1e-12,
        "contest task 4/5 runs camera feedback without a chassis state");

  const auto contest6 = tick_without_chassis(nx_control::TaskMode::Contest6, 55.0);
  check(!contest6.request_stop &&
            contest6.command.control_state == nx_control::TaskState::HoldTarget &&
            std::abs(contest6.acceleration_used_m_s2) < 1e-12,
        "contest task 6 runs camera feedback without a chassis state");
}

void test_contest456_ignores_chassis_dynamics() {
  nx_control::ControlConfig config;
  config.measurement_sigma_m = 1e-6;
  config.innovation_gate_sigma = 1e6;
  config.vision_position_filter_tau_s = 0.0;

  auto run = [&](nx_control::TaskMode mode, double target_m,
                 bool inject_chassis) {
    nx_control::NxController controller(config);
    constexpr double now_s = 80.02;
    controller.reset(80.0);
    controller.configure_task(mode, target_m, true);

    if (inject_chassis) {
      nx_control::ChassisState chassis;
      chassis.sequence = 1U;
      chassis.chassis_time_ms = 80020U;
      chassis.velocity_ref_m_s = 1.2;
      chassis.acceleration_ref_m_s2 = 0.45;
      chassis.jerk_ref_m_s3 = -0.50;
      chassis.velocity_actual_m_s = -1.0;
      chassis.acceleration_actual_m_s2 = -0.40;
      chassis.motion_phase = nx_control::MotionPhase::Accel;
      chassis.events = 0x0001U;
      chassis.ttl_ms = 100U;
      chassis.receive_time_s = now_s;
      controller.ingest_chassis_state(chassis);
    }

    nx_control::TubeStatus tube;
    tube.sequence = 1U;
    tube.dmmc_time_ms = 80020U;
    tube.receive_time_s = now_s;
    controller.ingest_tube_status(tube);

    nx_control::VisionMeasurement vision;
    vision.frame_id = 1U;
    vision.status = nx_control::VisionStatus::Measured;
    vision.position_m = 0.025;
    vision.ball_confidence = vision.tube_confidence = 1.0;
    vision.capture_time_ms = 80020U;
    vision.has_capture_time = true;
    vision.receive_time_s = now_s;
    controller.ingest_vision(vision);

    return controller.tick(now_s);
  };

  for (const auto mode_and_target :
       {std::make_pair(nx_control::TaskMode::Contest45, 0.0),
        std::make_pair(nx_control::TaskMode::Contest6, -0.073)}) {
    const auto without_chassis =
        run(mode_and_target.first, mode_and_target.second, false);
    const auto with_chassis =
        run(mode_and_target.first, mode_and_target.second, true);
    check(std::abs(without_chassis.estimate.position_m -
                   with_chassis.estimate.position_m) < 1e-12 &&
              std::abs(without_chassis.estimate.velocity_m_s -
                       with_chassis.estimate.velocity_m_s) < 1e-12 &&
              std::abs(without_chassis.pid.unsaturated_m_s2 -
                       with_chassis.pid.unsaturated_m_s2) < 1e-12 &&
              std::abs(with_chassis.acceleration_used_m_s2) < 1e-12,
          "contest task 4/5/6 PID and observer ignore chassis dynamics");
  }
}

void test_contest3_waiting_holds_beam() {
  nx_control::ControlConfig config;
  nx_control::NxController controller(config);
  controller.reset(60.0);
  controller.configure_task(nx_control::TaskMode::Contest3, 0.0, false, false);

  nx_control::TubeStatus tube;
  tube.sequence = 1;
  tube.dmmc_time_ms = 60000U;
  tube.receive_time_s = 60.0;
  controller.ingest_tube_status(tube);

  const auto waiting = controller.tick(60.0 + config.period_s);
  check(!waiting.request_stop &&
            waiting.command.control_state == nx_control::TaskState::StandbyHold &&
            std::abs(waiting.command.theta_cmd_rad) < 1e-12 &&
            (waiting.command.flags & 0x01U) != 0U,
        "contest task 3 waiting keeps motor enabled in zero-degree HOLD");
  const auto packet = nx_control::protocol::encode_control_command(waiting.command);
  check(packet[16] == 0U && packet[17] == 0U && packet[22] == 1U,
        "contest task 3 waiting emits zero-degree MC02 HOLD");

  nx_control::VisionMeasurement vision;
  vision.frame_id = 1;
  vision.status = nx_control::VisionStatus::Measured;
  vision.ball_confidence = vision.tube_confidence = 1.0;
  vision.capture_time_ms = 60040U;
  vision.has_capture_time = true;
  vision.receive_time_s = 60.04;
  controller.ingest_vision(vision);
  tube.sequence = 2;
  tube.dmmc_time_ms = 60040U;
  tube.receive_time_s = 60.04;
  controller.ingest_tube_status(tube);

  controller.start_task(60.04);
  const auto started = controller.tick(60.04);
  check(started.command.control_state == nx_control::TaskState::StaticMove,
        "contest task 3 leaves HOLD when the start key is pressed");
}

void test_controller_task3_friction_and_protocol() {
  constexpr double kDegrees =
      3.14159265358979323846 / 180.0;
  nx_control::ControlConfig config;
  config.task3_friction_blend_time_s = 0.04;
  nx_control::NxController controller(config);
  controller.reset(100.0);
  controller.configure_task(nx_control::TaskMode::Contest3, 0.0, true);

  nx_control::ControlOutput output;
  double feedback_theta_rad = 0.0;
  double previous_theta_rad = 0.0;
  bool rate_limit_ok = true;
  bool saw_positive_breakaway = false;
  for (std::uint32_t sequence = 1; sequence <= 40; ++sequence) {
    const double now_s = 100.0 + sequence * config.period_s;
    const auto now_ms =
        static_cast<std::uint32_t>(std::llround(now_s * 1000.0));
    nx_control::TubeStatus tube;
    tube.sequence = sequence;
    tube.dmmc_time_ms = now_ms;
    tube.theta_reference_rad = feedback_theta_rad;
    tube.theta_actual_rad = feedback_theta_rad;
    tube.receive_time_s = now_s;
    controller.ingest_tube_status(tube);

    nx_control::VisionMeasurement vision;
    vision.frame_id = sequence;
    vision.status = nx_control::VisionStatus::Measured;
    vision.position_m = 0.0;
    vision.ball_confidence = vision.tube_confidence = 1.0;
    vision.capture_time_ms = now_ms;
    vision.has_capture_time = true;
    vision.receive_time_s = now_s;
    controller.ingest_vision(vision);

    output = controller.tick(now_s, 1000U + sequence);
    rate_limit_ok =
        rate_limit_ok &&
        std::abs(output.command.theta_cmd_rad - previous_theta_rad) <=
            config.theta_rate_limit_rad_s * config.period_s + 1e-12 &&
        std::abs(output.command.theta_cmd_rad) <= config.theta_limit_rad + 1e-12;
    previous_theta_rad = output.command.theta_cmd_rad;
    feedback_theta_rad = output.command.theta_cmd_rad;
    saw_positive_breakaway =
        saw_positive_breakaway ||
        (output.friction_mode ==
             nx_control::FrictionMode::BreakawayPositive &&
         output.friction_direction == 1 &&
         output.command.theta_cmd_rad > 0.4 * kDegrees);
  }
  check(rate_limit_ok,
        "Task 3 total bias/friction/PID angle respects 4 degree and 2 degree/s limits");
  check(saw_positive_breakaway &&
            output.friction_mode ==
                nx_control::FrictionMode::RollingPositive &&
            output.friction_direction == 1 &&
            std::abs(output.theta_bias_rad + 0.15 * kDegrees) < 1e-12,
        "NX Task 3 bounds positive breakaway then falls back to rolling compensation");
  check(output.command.command_id == 1040U &&
            output.command.source_frame_id == 40U &&
            output.command.ttl_ms == 60U &&
            output.command.flags == 0x01U &&
            output.command.control_state == nx_control::TaskState::StaticMove,
        "Task 3 compensation preserves command IDs, frame IDs, TTL, flags, and state semantics");

  nx_control::TubeStatus tube;
  tube.sequence = 41U;
  tube.dmmc_time_ms = 100920U;
  tube.theta_reference_rad = feedback_theta_rad;
  tube.theta_actual_rad = feedback_theta_rad;
  tube.receive_time_s = 100.92;
  controller.ingest_tube_status(tube);
  output = controller.tick(100.92, 1041U);
  check(!output.safety_latched &&
            output.command.control_state ==
                nx_control::TaskState::StandbyHold &&
            std::abs(output.command.theta_cmd_rad) < 1e-12 &&
            std::abs(output.theta_bias_rad) < 1e-12 &&
            std::abs(output.theta_friction_rad) < 1e-12,
        "Task 3 vision soft HOLD applies no bias or friction compensation");

  nx_control::VisionMeasurement fresh_vision;
  fresh_vision.frame_id = 41U;
  fresh_vision.status = nx_control::VisionStatus::Measured;
  fresh_vision.position_m = 0.0;
  fresh_vision.ball_confidence = fresh_vision.tube_confidence = 1.0;
  fresh_vision.capture_time_ms = 100940U;
  fresh_vision.has_capture_time = true;
  fresh_vision.receive_time_s = 100.94;
  controller.ingest_vision(fresh_vision);
  tube.sequence = 42U;
  tube.dmmc_time_ms = 100940U;
  tube.receive_time_s = 100.94;
  controller.ingest_tube_status(tube);
  output = controller.tick(100.94, 1042U);
  check(!output.safety_latched &&
            output.command.control_state == nx_control::TaskState::StaticMove,
        "Task 3 resumes normally after soft vision HOLD");

  tube.sequence = 43U;
  tube.dmmc_time_ms = 101200U;
  tube.receive_time_s = 101.20;
  controller.ingest_tube_status(tube);
  output = controller.tick(101.20, 1043U);
  check(output.safety_latched &&
            output.command.control_state == nx_control::TaskState::Safe &&
            std::abs(output.command.theta_cmd_rad) < 1e-12 &&
            std::abs(output.theta_bias_rad) < 1e-12 &&
            std::abs(output.theta_friction_rad) < 1e-12,
        "Task 3 vision-stale SAFE path applies no bias or friction compensation");

  controller.start_task(101.22);
  tube.sequence = 44U;
  tube.dmmc_time_ms = 101220U;
  tube.receive_time_s = 101.22;
  controller.ingest_tube_status(tube);
  fresh_vision.frame_id = 42U;
  fresh_vision.capture_time_ms = 101220U;
  fresh_vision.receive_time_s = 101.22;
  controller.ingest_vision(fresh_vision);
  output = controller.tick(101.22, 1044U);
  check(!output.safety_latched,
        "operator restart clears Task 3 SAFE before stale-DMMC test");

  fresh_vision.frame_id = 43U;
  fresh_vision.capture_time_ms = 101280U;
  fresh_vision.receive_time_s = 101.28;
  controller.ingest_vision(fresh_vision);
  output = controller.tick(101.28, 1045U);
  check(output.safety_latched && output.request_stop &&
            output.command.control_state == nx_control::TaskState::Fault &&
            std::abs(output.command.theta_cmd_rad) < 1e-12 &&
            std::abs(output.theta_bias_rad) < 1e-12 &&
            std::abs(output.theta_friction_rad) < 1e-12 &&
            output.friction_mode == nx_control::FrictionMode::Hold &&
            output.friction_direction == 0,
        "DMMC stale safety path bypasses all Task 3 bias and friction compensation");
}

void test_controller_task3_position_early_braking() {
  nx_control::ControlConfig config;
  config.innovation_gate_sigma = 1000.0;
  config.measurement_sigma_m = 1e-6;
  config.vision_position_filter_tau_s = 0.0;
  nx_control::NxController controller(config);
  controller.reset(300.0);
  controller.configure_task(nx_control::TaskMode::Contest3, 0.0, true);

  nx_control::ControlOutput output;
  double feedback_theta_rad = 0.0;
  bool braked_before_threshold = false;
  bool saw_positive_braking_after_threshold = false;
  bool saw_positive_overshoot_recovery = false;
  double strongest_normal_braking_theta_rad = 0.0;
  double strongest_overshoot_theta_rad = 0.0;
  for (std::uint32_t sequence = 1; sequence <= 80; ++sequence) {
    const double now_s = 300.0 + sequence * config.period_s;
    const auto now_ms =
        static_cast<std::uint32_t>(std::llround(now_s * 1000.0));

    nx_control::TubeStatus tube;
    tube.sequence = sequence;
    tube.dmmc_time_ms = now_ms;
    tube.theta_actual_rad = feedback_theta_rad;
    tube.receive_time_s = now_s;
    controller.ingest_tube_status(tube);

    nx_control::VisionMeasurement vision;
    vision.frame_id = sequence;
    vision.status = nx_control::VisionStatus::Measured;
    vision.position_m =
        sequence <= 5U
            ? 0.0
            : static_cast<double>(sequence - 5U) * 0.001;
    vision.ball_confidence = vision.tube_confidence = 1.0;
    vision.capture_time_ms = now_ms;
    vision.has_capture_time = true;
    vision.receive_time_s = now_s;
    controller.ingest_vision(vision);

    output = controller.tick(now_s);
    feedback_theta_rad = output.command.theta_cmd_rad;
    if (output.estimate.position_m <
        config.task3_positive_early_brake_position_m - 0.001) {
      braked_before_threshold =
          braked_before_threshold || output.task3_early_braking;
    }
    if (output.estimate.position_m >=
            config.task3_positive_early_brake_position_m &&
        output.estimate.position_m <=
            config.task3_positive_overshoot_position_m &&
        output.task3_early_braking &&
        !output.task3_positive_overshoot_recovery) {
      saw_positive_braking_after_threshold = true;
      strongest_normal_braking_theta_rad =
          std::min(strongest_normal_braking_theta_rad,
                   output.theta_pid_rad);
    }
    if (output.estimate.position_m >
            config.task3_positive_overshoot_position_m &&
        output.task3_positive_overshoot_recovery) {
      saw_positive_overshoot_recovery = true;
      strongest_overshoot_theta_rad =
          std::min(strongest_overshoot_theta_rad,
                   output.theta_pid_rad);
    }
  }
  check(!braked_before_threshold &&
            saw_positive_braking_after_threshold,
        "Task 3 starts active reverse braking at measured +3.8 cm");
  check(saw_positive_overshoot_recovery &&
            strongest_overshoot_theta_rad <
                strongest_normal_braking_theta_rad -
                    0.05 * 3.14159265358979323846 / 180.0,
        "Task 3 applies a moderately stronger reverse command beyond +5.0 cm");

  bool saw_reverse_balance = false;
  bool reverse_balance_command_ok = true;
  bool saw_speed_control_after_balance = false;
  for (std::uint32_t sequence = 81; sequence <= 145; ++sequence) {
    const double now_s = 300.0 + sequence * config.period_s;
    const auto now_ms =
        static_cast<std::uint32_t>(std::llround(now_s * 1000.0));

    nx_control::TubeStatus tube;
    tube.sequence = sequence;
    tube.dmmc_time_ms = now_ms;
    tube.theta_actual_rad = feedback_theta_rad;
    tube.receive_time_s = now_s;
    controller.ingest_tube_status(tube);

    nx_control::VisionMeasurement vision;
    vision.frame_id = sequence;
    vision.status = nx_control::VisionStatus::Measured;
    vision.position_m =
        0.075 - static_cast<double>(sequence - 80U) * 0.001;
    vision.ball_confidence = vision.tube_confidence = 1.0;
    vision.capture_time_ms = now_ms;
    vision.has_capture_time = true;
    vision.receive_time_s = now_s;
    controller.ingest_vision(vision);

    output = controller.tick(now_s);
    if (output.task3_reverse_balance_active) {
      saw_reverse_balance = true;
      reverse_balance_command_ok =
          reverse_balance_command_ok &&
          std::abs(output.theta_pid_rad) < 1e-12 &&
          std::abs(output.theta_friction_rad) < 1e-12 &&
          output.pid.integrator_frozen &&
          std::abs(output.command.theta_rate_limit_rad_s -
                   config.task3_reverse_balance_rate_limit_rad_s) < 1e-12 &&
          std::abs(output.command.theta_cmd_rad -
                   config.task3_theta_bias_rad) <=
              std::abs(feedback_theta_rad -
                       config.task3_theta_bias_rad) +
                  1e-12;
    } else if (saw_reverse_balance &&
               std::abs(feedback_theta_rad -
                        config.task3_theta_bias_rad) <=
                   config.task3_settle_theta_tolerance_rad) {
      saw_speed_control_after_balance =
          std::abs(output.command.theta_rate_limit_rad_s -
                   config.theta_rate_limit_rad_s) < 1e-12;
    }
    feedback_theta_rad = output.command.theta_cmd_rad;
  }
  check(saw_reverse_balance && reverse_balance_command_ok,
        "Task 3 detects robust negative velocity and drives directly to balance at 4 degrees per second");
  check(saw_speed_control_after_balance,
        "Task 3 exits the one-shot balance phase and restores normal speed control");
}

void test_csv_task3_diagnostics() {
  char path[] = "/tmp/nx-control-csv-XXXXXX";
  const int temporary_fd = ::mkstemp(path);
  check(temporary_fd >= 0, "create temporary Task 3 CSV");
  if (temporary_fd < 0) return;
  ::close(temporary_fd);

  {
    nx_control::CsvLogger logger;
    check(logger.open(path), "open temporary Task 3 CSV");
    nx_control::ControlOutput output;
    output.task3_stage = 1;
    output.reference = nx_control::ReferencePoint{-0.01, -0.02, 0.03};
    output.settle_position_ok = true;
    output.settle_velocity_ok = false;
    output.settle_theta_ok = true;
    output.settle_elapsed_ms = 250.0;
    output.friction_mode = nx_control::FrictionMode::RollingNegative;
    output.friction_direction = -1;
    output.vision_capture_age_ms = 34.5;
    nx_control::TubeStatus tube;
    tube.theta_actual_rad = -0.01;
    logger.write(1.0, output, &tube, nullptr);
  }

  std::ifstream stream(path);
  std::string header;
  std::string row;
  std::getline(stream, header);
  std::getline(stream, row);
  const std::array<const char*, 32> required_fields{
      "task3_stage",       "planned_x_m",       "planned_v_m_s",
      "planned_a_m_s2",    "settle_position_ok", "settle_velocity_ok",
      "settle_theta_ok",    "settle_elapsed_ms", "task3_early_braking",
      "task3_positive_overshoot_recovery",
      "task3_reverse_balance_active",
      "theta_pid_deg",
      "theta_bias_deg",     "theta_friction_deg", "theta_command_deg",
      "friction_mode",      "friction_direction", "theta_actual_deg",
      "vision_capture_age_ms", "position_error_m", "velocity_error_m_s",
      "pid_p_m_s2", "pid_i_m_s2", "pid_d_m_s2", "pid_ff_m_s2",
      "pid_disturbance_m_s2", "pid_unsaturated_m_s2", "pid_applied_m_s2",
      "pid_integrator_frozen", "pid_integral_limited", "pid_saturated",
      "inner_angle_error_rad"};
  bool fields_present = true;
  for (const char* field : required_fields) {
    fields_present = fields_present &&
                     header.find(field) != std::string::npos;
  }
  check(fields_present,
        "control CSV contains every required Task 3 planning, settle, and friction field");
  check(row.find("ROLLING_NEGATIVE,-1") != std::string::npos,
        "control CSV writes human-readable friction mode and direction");
  ::unlink(path);
}

}  // namespace

int main() {
  test_protocol();
  test_delayed_observer();
  test_vision_position_filter();
  test_remote_clock_sync();
  test_runtime_command_protocol();
  test_persistent_sequence();
  test_pid_controller();
  test_deployed_timing_contract();
  test_task_manager();
  test_task3_friction_compensator();
  test_controller_safety();
  test_controller_angle_limit();
  test_inner_angle_watchdog();
  test_controller_hold_deadband();
  test_contest_chassis_gate();
  test_contest456_ignores_chassis_dynamics();
  test_contest3_waiting_holds_beam();
  test_controller_task3_friction_and_protocol();
  test_controller_task3_position_early_braking();
  test_csv_task3_diagnostics();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all nx_control tests passed\n";
  return 0;
}
