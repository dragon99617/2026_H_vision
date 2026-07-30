#include "nx_control/controller.hpp"
#include "nx_control/io.hpp"
#include "nx_control/mpc.hpp"
#include "nx_control/observer.hpp"
#include "nx_control/protocol.hpp"
#include "nx_control/task_manager.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
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

class ZeroSolver final : public nx_control::QpSolver {
 public:
  nx_control::QpResult solve(const nx_control::QpProblem& problem) override {
    nx_control::QpResult result;
    result.solved = true;
    result.iterations = 1;
    result.primal = Eigen::VectorXd::Zero(problem.hessian.rows());
    result.status = "solved";
    return result;
  }
  const char* name() const override { return "zero-test"; }
};

class LargePositiveSolver final : public nx_control::QpSolver {
 public:
  nx_control::QpResult solve(const nx_control::QpProblem& problem) override {
    nx_control::QpResult result;
    result.solved = true;
    result.iterations = 1;
    result.primal = Eigen::VectorXd::Zero(problem.hessian.rows());
    result.primal(0) = 1000.0;
    result.status = "solved";
    return result;
  }
  const char* name() const override { return "large-positive-test"; }
};

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

void test_mpc_constraints() {
  nx_control::ControlConfig config;
  check(std::abs(config.theta_limit_rad -
                 4.0 * 3.14159265358979323846 / 180.0) < 1e-12,
        "NX default angle limit matches DMMC02 4.0 degree hard limit");
  check(std::abs(config.theta_rate_limit_rad_s -
                 2.0 * 3.14159265358979323846 / 180.0) < 1e-12,
        "NX default angle rate limit is 2 degrees per second");
  check(std::abs(config.position_scale_m - 0.010) < 1e-12 &&
            std::abs(config.velocity_scale_m_s - 0.015) < 1e-12 &&
            std::abs(config.input_scale_m_s2 - 0.100) < 1e-12 &&
            std::abs(config.delta_input_scale_m_s2 - 0.015) < 1e-12,
        "MPC scales penalize velocity, angle, and angle changes more strongly");
  check(std::abs(config.hold_enter_position_error_m - 0.004) < 1e-12 &&
            std::abs(config.hold_enter_velocity_m_s - 0.015) < 1e-12 &&
            std::abs(config.hold_exit_position_error_m - 0.008) < 1e-12,
        "HoldTarget deadband defaults are 4 mm, 15 mm/s, and 8 mm");
  config.horizon = 12;
  config.solver_deadline_ms = 1000.0;
  config.qp_max_iterations = 1000;
  config.qp_eps_abs = 1e-5;
  config.qp_eps_rel = 1e-5;
  nx_control::BallMpc mpc(config);
  std::vector<nx_control::ReferencePoint> reference(12);
  std::vector<double> acceleration(12, 0.3);
  const auto result = mpc.solve({0.03, 0.0, 0.0, 0.0}, 0.0, reference, acceleration);
  const auto& problem = mpc.last_problem();
  const double expected_u_limit = nx_control::kGravity * std::tan(config.theta_limit_rad);
  check(std::abs(problem.lower(config.horizon) +
                 expected_u_limit / config.input_scale_m_s2) < 1e-12 &&
            std::abs(problem.upper(config.horizon) -
                     expected_u_limit / config.input_scale_m_s2) < 1e-12,
        "MPC input bounds are exactly plus/minus 4.0 degrees");
#ifdef NX_CONTROL_HAS_OSQP
  check(result.backend == "osqp", "production build selects the OSQP backend");
#endif
  check(result.solved, "MPC solves nominal QP");
  if (result.solved) {
    const double u_limit = nx_control::kGravity * std::tan(config.theta_limit_rad) + 1e-5;
    const double du_limit = nx_control::kGravity *
                            std::tan(config.theta_rate_limit_rad_s * config.period_s) + 1e-3;
    double previous = 0.0;
    for (double command : result.command_sequence) {
      check(std::abs(command) <= u_limit,
            "MPC command angle constraint: command=" + std::to_string(command) +
                " limit=" + std::to_string(u_limit));
      check(std::abs(command - previous) <= du_limit, "MPC command rate constraint");
      previous = command;
    }
  }
}

void test_mpc_actuator_delay() {
  nx_control::ControlConfig config;
  config.horizon = 6;
  config.actuator_delay_s = 2.0 * config.period_s;
  config.solver_deadline_ms = 1000.0;
  nx_control::BallMpc mpc(config, std::make_unique<ZeroSolver>());
  mpc.solve({0.0, 0.0, 0.0, 0.0}, 0.0,
            std::vector<nx_control::ReferencePoint>(6), std::vector<double>(6));
  const auto& constraints = mpc.last_problem().constraint;
  // Positive-position soft constraint for prediction step 1 starts at row 3*N+1.
  check(constraints.block(3 * config.horizon + 1, 0, 1, config.horizon)
                .cwiseAbs()
                .maxCoeff() < 1e-12,
        "identified pure delay postpones command influence in prediction");
}

void test_task_manager() {
  nx_control::TaskManager task;
  task.configure(nx_control::TaskMode::Contest3, 0.0, true);
  check(std::abs(task.target_m() - 0.05) < 1e-12,
        "contest task 3 starts toward +5 cm");
  nx_control::ObserverState state{0.05, 0.0, 0.0};
  task.update(1.0, state, nullptr);
  task.update(1.21, state, nullptr);
  check(std::abs(task.target_m() + 0.05) < 1e-12,
        "contest task 3 switches +5 to -5 cm");
  state.position_m = -0.05;
  task.update(1.22, state, nullptr);
  task.update(1.43, state, nullptr);
  check(task.static_sequence_complete() &&
            task.state() == nx_control::TaskState::HoldTarget &&
            std::abs(task.target_m() + 0.05) < 1e-12 &&
            task.target_hold_deadband_active(),
        "contest task 3 holds -5 cm after the second 0.2 second stable period");

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
        "contest task 4/5 can wait for chassis start");
  task.update(2.0, centered, &chassis);
  check(task.state() == nx_control::TaskState::VehicleAccel &&
            std::abs(task.target_m()) < 1e-12,
        "contest task 4/5 starts on event and always targets center");
  chassis.motion_phase = nx_control::MotionPhase::Curve;
  task.update(2.1, centered, &chassis);
  check(task.state() == nx_control::TaskState::VehicleCruise,
        "contest task 4/5 follows chassis phase");

  task.configure(nx_control::TaskMode::Contest6, -0.073, true);
  check(task.state() == nx_control::TaskState::HoldTarget &&
            std::abs(task.target_m() + 0.073) < 1e-12,
        "contest task 6 starts at requested target");
  chassis.events = 0U;
  chassis.motion_phase = nx_control::MotionPhase::Decel;
  task.update(3.0, centered, &chassis);
  check(task.state() == nx_control::TaskState::VehicleDecel &&
            std::abs(task.target_m() + 0.073) < 1e-12,
        "contest task 6 preserves target during vehicle motion");

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

void test_controller_safety() {
  nx_control::ControlConfig config;
  config.solver_deadline_ms = 1000.0;
  nx_control::NxController controller(config, std::make_unique<ZeroSolver>());
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
  nx_control::NxController boundary_controller(
      boundary_config, std::make_unique<ZeroSolver>());
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
  nx_control::NxController prediction_controller(
      prediction_config, std::make_unique<ZeroSolver>());
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
  config.solver_deadline_ms = 1000.0;
  nx_control::NxController controller(config, std::make_unique<LargePositiveSolver>());
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
        "controller final output clamps at the same 4.0 degree limit as MPC");
  const auto packet = nx_control::protocol::encode_control_command(output.command);
  check(packet[16] == 0x90U && packet[17] == 0x01U,
        "tube-control-v3 still encodes 4.0 degrees as 400 cdeg");
  check(packet[18] == 0xC8U && packet[19] == 0x00U,
        "tube-control-v3 encodes the rate limit as 200 cdeg per second");
}

void test_controller_hold_deadband() {
  nx_control::ControlConfig config;
  config.solver_deadline_ms = 1000.0;
  nx_control::NxController controller(config, std::make_unique<LargePositiveSolver>());
  controller.reset(70.0);
  controller.configure_task(nx_control::TaskMode::HoldCenter, 0.0, true);

  auto ingest_healthy_sample = [&](std::uint32_t sequence) {
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
    vision.ball_confidence = vision.tube_confidence = 1.0;
    vision.capture_time_ms = now_ms;
    vision.has_capture_time = true;
    vision.receive_time_s = now_s;
    controller.ingest_vision(vision);
    return now_s;
  };

  nx_control::ControlOutput output;
  for (std::uint32_t sequence = 1; sequence <= 10; ++sequence) {
    output = controller.tick(ingest_healthy_sample(sequence));
  }
  const double theta_before_hold_rad = output.command.theta_cmd_rad;
  check(theta_before_hold_rad > 0.0,
        "controller has a nonzero command before entering HoldTarget deadband");

  controller.configure_task(nx_control::TaskMode::HoldTarget, 0.0, true);
  output = controller.tick(ingest_healthy_sample(11));
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
    output = controller.tick(ingest_healthy_sample(sequence));
  }
  check(std::abs(output.command.theta_cmd_rad) < 1e-12 &&
            output.reason == "hold_deadband",
        "HoldTarget deadband settles theta at zero without resuming MPC corrections");
}

void test_contest3_chassis_gate() {
  nx_control::ControlConfig config;
  config.solver_deadline_ms = 1000.0;

  auto tick_without_chassis = [&](nx_control::TaskMode mode, double now_s) {
    nx_control::NxController controller(config, std::make_unique<ZeroSolver>());
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
  check(contest45.request_stop && contest45.reason == "chassis_stale",
        "contest task 4/5 still requires fresh chassis");
}

void test_contest3_waiting_holds_beam() {
  nx_control::ControlConfig config;
  config.solver_deadline_ms = 1000.0;
  nx_control::NxController controller(config, std::make_unique<ZeroSolver>());
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

}  // namespace

int main() {
  test_protocol();
  test_delayed_observer();
  test_remote_clock_sync();
  test_persistent_sequence();
  test_mpc_constraints();
  test_mpc_actuator_delay();
  test_task_manager();
  test_controller_safety();
  test_controller_angle_limit();
  test_controller_hold_deadband();
  test_contest3_chassis_gate();
  test_contest3_waiting_holds_beam();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all nx_control tests passed\n";
  return 0;
}
