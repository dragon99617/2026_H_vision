#include "nx_control/controller.hpp"
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

  nx_control::ControlCommand fault_command;
  fault_command.command_id = 1;
  fault_command.theta_rate_limit_rad_s = 50.0 * 3.14159265358979323846 / 180.0;
  fault_command.ttl_ms = 60;
  fault_command.control_state = nx_control::TaskState::Fault;
  fault_command.flags = 0x08U;
  const std::vector<std::uint8_t> expected_control{
      0xA5, 0x5A, 0x80, 0x16, 0x01, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x88, 0x13, 0x3C, 0x00, 0x04, 0x08,
      0x00, 0x00, 0x42, 0x41};
  check(nx_control::protocol::encode_control_command(fault_command) == expected_control,
        "MC02 control golden vector");

  const std::array<std::pair<nx_control::TaskState, std::uint8_t>, 9> state_mapping{{
      {nx_control::TaskState::Idle, 0U},
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

void test_mpc_constraints() {
  nx_control::ControlConfig config;
  config.horizon = 12;
  config.solver_deadline_ms = 1000.0;
  config.qp_max_iterations = 1000;
  config.qp_eps_abs = 1e-3;
  config.qp_eps_rel = 1e-3;
  nx_control::BallMpc mpc(config);
  std::vector<nx_control::ReferencePoint> reference(12);
  std::vector<double> acceleration(12, 0.3);
  const auto result = mpc.solve({0.03, 0.0, 0.0, 0.0}, 0.0, reference, acceleration);
  check(result.solved, "dense MPC solves nominal QP");
  if (result.solved) {
    const double u_limit = nx_control::kGravity * std::tan(config.theta_limit_rad) + 1e-5;
    const double du_limit = nx_control::kGravity *
                            std::tan(config.theta_rate_limit_rad_s * config.period_s) + 1e-3;
    double previous = 0.0;
    for (double command : result.command_sequence) {
      check(std::abs(command) <= u_limit, "MPC command angle constraint");
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
  task.configure(nx_control::TaskMode::StaticSequence, 0.0, true);
  nx_control::ObserverState state{0.05, 0.0, 0.0};
  task.update(1.0, state, nullptr);
  task.update(1.21, state, nullptr);
  check(std::abs(task.target_m() + 0.05) < 1e-12, "static task switches +5 to -5 cm");
  state.position_m = -0.05;
  task.update(1.22, state, nullptr);
  task.update(1.43, state, nullptr);
  check(task.static_sequence_complete(), "static task completes after stable -5 cm");
}

void test_controller_safety() {
  nx_control::ControlConfig config;
  config.solver_deadline_ms = 1000.0;
  nx_control::NxController controller(config, std::make_unique<ZeroSolver>());
  controller.reset(10.0);
  controller.configure_task(nx_control::TaskMode::HoldCenter, 0.0, true);
  nx_control::TubeStatus tube;
  tube.receive_time_s = 10.0;
  nx_control::ChassisState chassis;
  chassis.receive_time_s = 10.0;
  chassis.ttl_ms = 100;
  nx_control::VisionMeasurement vision;
  vision.status = nx_control::VisionStatus::Measured;
  vision.ball_confidence = vision.tube_confidence = 1.0;
  vision.receive_time_s = 10.0;
  vision.capture_time_ms = 10000;
  vision.has_capture_time = true;
  controller.ingest_tube_status(tube);
  controller.ingest_chassis_state(chassis);
  controller.ingest_vision(vision);
  vision.position_m = 0.12;
  controller.ingest_vision(vision);  // duplicate frame must not trip the raw safety boundary
  auto output = controller.tick(10.02);
  check(!output.request_stop, "healthy controller does not request stop");
  output = controller.tick(10.20);
  check(output.request_stop && (output.command.flags & 0x08U) != 0U,
        "stale vision/chassis requests stop");
}

}  // namespace

int main() {
  test_protocol();
  test_delayed_observer();
  test_remote_clock_sync();
  test_mpc_constraints();
  test_mpc_actuator_delay();
  test_task_manager();
  test_controller_safety();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all nx_control tests passed\n";
  return 0;
}
