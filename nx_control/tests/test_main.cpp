#include "nx_control/controller.hpp"
#include "nx_control/mpc.hpp"
#include "nx_control/observer.hpp"
#include "nx_control/protocol.hpp"
#include "nx_control/task_manager.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
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
