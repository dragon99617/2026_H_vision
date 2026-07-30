#include "nx_control/controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nx_control {
namespace {

bool sequence_is_newer(std::uint32_t value, std::uint32_t previous) {
  return static_cast<std::int32_t>(value - previous) > 0;
}

}  // namespace

NxController::NxController(const ControlConfig& config, std::unique_ptr<QpSolver> solver)
    : config_(config),
      observer_(config),
      chassis_sync_(config),
      mpc_(config, std::move(solver)),
      task_manager_(config) {}

void NxController::configure_task(TaskMode mode, double target_m, bool start_immediately,
                                  bool start_on_chassis_event) {
  task_manager_.configure(mode, target_m, start_immediately, start_on_chassis_event);
}

void NxController::start_task(double now_s) {
  safety_latched_ = false;
  safety_fault_latched_ = false;
  clear_comm_warning_pending_ = true;
  previous_command_u_ = 0.0;
  solver_failures_ = 0;
  first_solver_failure_s_ = -1.0;
  task_manager_.start(now_s);
}

void NxController::reset(double now_s) {
  observer_.reset(now_s);
  chassis_sync_.reset();
  last_tick_s_ = now_s;
  last_vision_receive_s_ = last_accepted_vision_s_ = -1.0;
  have_vision_frame_ = have_chassis_sequence_ = false;
  have_tube_status_ = false;
  vision_clock_.reset();
  dmmc_clock_.reset();
  chassis_clock_.reset();
  v2_clock_initialized_ = false;
  latest_visual_position_valid_ = false;
  previous_command_u_ = 0.0;
  solver_failures_ = 0;
  first_solver_failure_s_ = -1.0;
  safety_latched_ = false;
  safety_fault_latched_ = false;
  clear_comm_warning_pending_ = false;
  safety_event_id_ = 0;
  last_stop_reason_.clear();
}

void NxController::ingest_vision(VisionMeasurement measurement) {
  const bool had_previous_frame = have_vision_frame_;
  const std::uint32_t previous_frame_id = last_vision_frame_id_;
  if (had_previous_frame && !sequence_is_newer(measurement.frame_id, previous_frame_id)) {
    if (measurement.receive_time_s - last_vision_receive_s_ <= config_.vision_loss_safe_s) return;
    v2_clock_initialized_ = false;
  }
  have_vision_frame_ = true;
  last_vision_receive_s_ = measurement.receive_time_s;
  last_vision_frame_id_ = measurement.frame_id;
  if (measurement.has_capture_time) {
    measurement.capture_time_s =
        vision_clock_.unwrap_seconds(measurement.capture_time_ms, measurement.receive_time_s);
  } else {
    const double arrival_estimate = measurement.receive_time_s - config_.vision_v2_latency_s;
    if (!v2_clock_initialized_) {
      v2_capture_time_s_ = arrival_estimate;
      v2_clock_initialized_ = true;
    } else {
      const std::int32_t frame_delta =
          static_cast<std::int32_t>(measurement.frame_id - previous_frame_id);
      const double frame_estimate =
          v2_capture_time_s_ + static_cast<double>(std::max(1, frame_delta)) /
                                    config_.vision_frame_rate_hz;
      v2_capture_time_s_ = frame_estimate + 0.05 * (arrival_estimate - frame_estimate);
    }
    measurement.capture_time_s = v2_capture_time_s_;
  }
  // Vision and controller use CLOCK_MONOTONIC. Propagate to an inter-tick exposure time.
  measurement.capture_time_s = std::min(measurement.capture_time_s, measurement.receive_time_s);
  if (measurement.capture_time_s > observer_.time_s()) {
    const double actual_u = have_tube_status_
                                ? kGravity * std::tan(tube_status_.theta_actual_rad)
                                : previous_command_u_;
    const double acceleration =
        chassis_sync_.delay_compensated_actual_acceleration(measurement.receive_time_s);
    observer_.predict(measurement.capture_time_s, actual_u, acceleration);
  }
  if (observer_.update_position(measurement.capture_time_s, measurement.position_m,
                                measurement.ball_confidence, measurement.tube_confidence,
                                measurement.status)) {
    last_accepted_vision_s_ = measurement.receive_time_s;
  }
  latest_visual_position_valid_ = measurement.status == VisionStatus::Measured &&
                                  measurement.ball_confidence * measurement.tube_confidence >= 0.30;
  if (latest_visual_position_valid_) latest_visual_position_m_ = measurement.position_m;
}

void NxController::ingest_tube_status(const TubeStatus& status) {
  if (have_tube_status_ && !sequence_is_newer(status.sequence, last_tube_sequence_)) {
    if (status.receive_time_s - tube_status_.receive_time_s <= config_.dmmc_stale_s) return;
    dmmc_clock_.reset();
  }
  TubeStatus synchronized = status;
  synchronized.sample_time_s =
      dmmc_clock_.to_local_seconds(status.dmmc_time_ms, status.receive_time_s);
  const double previous_u = have_tube_status_
                                ? kGravity * std::tan(tube_status_.theta_actual_rad)
                                : previous_command_u_;
  observer_.predict(synchronized.sample_time_s, previous_u,
                    chassis_sync_.delay_compensated_actual_acceleration(status.receive_time_s));
  tube_status_ = synchronized;
  last_tube_sequence_ = status.sequence;
  have_tube_status_ = true;
}

void NxController::ingest_chassis_state(const ChassisState& state) {
  if (have_chassis_sequence_ && !sequence_is_newer(state.sequence, last_chassis_sequence_)) {
    if (state.receive_time_s - chassis_sync_.latest().receive_time_s <= config_.chassis_stale_s) {
      return;
    }
    chassis_clock_.reset();
  }
  ChassisState synchronized = state;
  synchronized.sample_time_s =
      chassis_clock_.to_local_seconds(state.chassis_time_ms, state.receive_time_s);
  const double actual_u = have_tube_status_ ? kGravity * std::tan(tube_status_.theta_actual_rad)
                                            : previous_command_u_;
  observer_.predict(synchronized.sample_time_s, actual_u,
                    chassis_sync_.delay_compensated_actual_acceleration(state.receive_time_s));
  last_chassis_sequence_ = state.sequence;
  have_chassis_sequence_ = true;
  chassis_sync_.ingest(synchronized);
}

double NxController::rate_limit_and_clamp(double requested_u) {
  const double hard = kGravity * std::tan(config_.theta_limit_rad);
  const double delta = kGravity * std::tan(config_.theta_rate_limit_rad_s * config_.period_s);
  requested_u = std::clamp(requested_u, previous_command_u_ - delta,
                           previous_command_u_ + delta);
  return std::clamp(requested_u, -hard, hard);
}

double NxController::fallback_command(const ObserverState& estimate,
                                      const ReferencePoint& reference,
                                      double feedforward) const {
  return feedforward - config_.fallback_kp * (estimate.position_m - reference.position_m) -
         config_.fallback_kd * (estimate.velocity_m_s - reference.velocity_m_s) -
         config_.fallback_disturbance_gain * estimate.disturbance_m_s2 /
             config_.rolling_lambda;
}

bool NxController::hold_prediction_crosses_soft_boundary(
    const ObserverState& estimate, double actuator_u_m_s2,
    double chassis_acceleration_m_s2, double horizon_s) const {
  double position = estimate.position_m;
  double velocity = estimate.velocity_m_s;
  double actuator_angle_rad = std::atan2(actuator_u_m_s2, kGravity);
  for (double t = config_.period_s; t <= horizon_s + 1e-9; t += config_.period_s) {
    const double angle_step = config_.theta_rate_limit_rad_s * config_.period_s;
    actuator_angle_rad =
        std::clamp(0.0, actuator_angle_rad - angle_step, actuator_angle_rad + angle_step);
    const double projected_u = kGravity * std::tan(actuator_angle_rad);
    const double acceleration =
        estimate.disturbance_m_s2 +
        config_.rolling_lambda * (projected_u - chassis_acceleration_m_s2);
    position += velocity * config_.period_s +
                0.5 * acceleration * config_.period_s * config_.period_s;
    velocity += acceleration * config_.period_s;
    if (std::abs(position) >= config_.position_soft_limit_m) return true;
  }
  return false;
}

void NxController::latch_safety(const std::string& reason, bool fault) {
  if (!safety_latched_) {
    safety_latched_ = true;
    safety_fault_latched_ = fault;
    ++safety_event_id_;
    last_stop_reason_ = reason.empty() ? "unspecified_safety_stop" : reason;
    clear_comm_warning_pending_ = false;
  } else if (fault) {
    safety_fault_latched_ = true;
  }
  task_manager_.force_safe(safety_fault_latched_);
}

ControlOutput NxController::tick(double now_s,
                                 std::optional<std::uint32_t> command_id) {
  if (last_tick_s_ == 0.0) reset(now_s);
  const bool chassis_valid = chassis_sync_.valid(now_s);
  const double chassis_acceleration =
      chassis_valid ? chassis_sync_.delay_compensated_actual_acceleration(now_s) : 0.0;
  const double actual_u = have_tube_status_ ? kGravity * std::tan(tube_status_.theta_actual_rad)
                                            : previous_command_u_;
  observer_.predict(now_s, actual_u, chassis_acceleration);
  last_tick_s_ = now_s;

  ControlOutput output;
  output.estimate = observer_.state();
  output.acceleration_used_m_s2 = chassis_acceleration;
  output.vision_age_ms = last_accepted_vision_s_ >= 0.0
                             ? 1000.0 * std::max(0.0, now_s - last_accepted_vision_s_)
                             : std::numeric_limits<double>::infinity();
  output.chassis_age_ms = 1000.0 * chassis_sync_.age_s(now_s);
  output.dmmc_age_ms = have_tube_status_
                           ? 1000.0 * std::max(0.0, now_s - tube_status_.receive_time_s)
                           : std::numeric_limits<double>::infinity();

  const ChassisState* chassis = chassis_valid ? &chassis_sync_.latest() : nullptr;
  output.reference = task_manager_.update(now_s, output.estimate, chassis);
  const bool contest3_waiting_for_start =
      task_manager_.mode() == TaskMode::Contest3 && task_manager_.state() == TaskState::Idle;
  const bool target_hold_deadband = task_manager_.target_hold_deadband_active();
  std::vector<double> acceleration_forecast =
      chassis_sync_.acceleration_reference_forecast(now_s, config_.horizon, config_.period_s);
  std::vector<ReferencePoint> reference = task_manager_.reference_horizon(config_.horizon);
  const double feedforward = acceleration_forecast.empty()
                                 ? 0.0
                                 : acceleration_forecast.front() +
                                       output.reference.acceleration_m_s2 / config_.rolling_lambda;

  const double vision_age_s = output.vision_age_ms / 1000.0;
  const bool vision_soft_hold =
      !contest3_waiting_for_start &&
      vision_age_s >= config_.vision_loss_hold_s &&
      vision_age_s <= config_.vision_loss_safe_s;
  const bool vision_hard_lost =
      !contest3_waiting_for_start && vision_age_s > config_.vision_loss_safe_s;
  const bool dmmc_stale = !have_tube_status_ ||
                          output.dmmc_age_ms > config_.dmmc_stale_s * 1000.0 ||
                          tube_status_.faults != 0U;
  const bool chassis_stale = !chassis_valid;
  const bool chassis_fresh_required = task_manager_.mode() != TaskMode::Contest3;
  const bool chassis_gate_failed = chassis_fresh_required && chassis_stale;
  const bool force_safe_feedforward =
      vision_soft_hold || vision_hard_lost || chassis_gate_failed || safety_latched_;
  if (vision_hard_lost) {
    output.request_stop = true;
    output.reason = "vision_stale";
  }
  if (chassis_gate_failed) {
    output.request_stop = true;
    output.reason = "chassis_stale";
  }
  if (dmmc_stale) {
    output.request_stop = true;
    output.reason = "dmmc_stale_or_fault";
  }

  double requested_u = feedforward;
  if (!force_safe_feedforward && !dmmc_stale && task_manager_.state() != TaskState::Idle &&
      !target_hold_deadband) {
    const std::array<double, 4> mpc_state{output.estimate.position_m,
                                          output.estimate.velocity_m_s,
                                          output.estimate.disturbance_m_s2, actual_u};
    const MpcResult result =
        mpc_.solve(mpc_state, previous_command_u_, reference, acceleration_forecast);
    output.mpc_solve_ms = result.solve_time_ms;
    output.max_predicted_slack_m = result.max_slack_m;
    output.solver_iterations = result.iterations;
    output.predicted_position_m = result.predicted_position;
    if (result.solved) {
      solver_failures_ = 0;
      first_solver_failure_s_ = -1.0;
      requested_u = result.command_m_s2;
    } else {
      if (solver_failures_++ == 0) first_solver_failure_s_ = now_s;
      const auto shifted = mpc_.shift_last_solution();
      requested_u = shifted.empty() ? fallback_command(output.estimate, output.reference, feedforward)
                                    : shifted.front();
      output.reason = result.timed_out ? "mpc_deadline" : "mpc_failure";
    }
    if (solver_failures_ >= config_.fallback_after_failures ||
        result.solve_time_ms > config_.solver_deadline_ms) {
      output.used_fallback = true;
      requested_u = fallback_command(output.estimate, output.reference, feedforward);
    }
    if (vision_age_s > config_.vision_decay_start_s) {
      const double correction_scale = std::clamp(
          (config_.vision_loss_hold_s - vision_age_s) /
              std::max(1e-6, config_.vision_loss_hold_s - config_.vision_decay_start_s),
          0.0, 1.0);
      requested_u = feedforward + correction_scale * (requested_u - feedforward);
    }
  }

  if (target_hold_deadband) {
    requested_u = 0.0;
    solver_failures_ = 0;
    first_solver_failure_s_ = -1.0;
    if (!output.request_stop) output.reason = "hold_deadband";
  }

  if (!contest3_waiting_for_start &&
      (solver_failures_ >= config_.stop_after_failures ||
       (first_solver_failure_s_ >= 0.0 &&
        now_s - first_solver_failure_s_ >= config_.stop_after_failure_s))) {
    output.request_stop = true;
    output.used_fallback = true;
    output.reason = "persistent_solver_failure";
  }
  if (!contest3_waiting_for_start &&
      (std::abs(output.estimate.position_m) > config_.position_soft_limit_m - 0.010 ||
       output.max_predicted_slack_m > 0.0001)) {
    output.request_slowdown = true;
  }
  const bool outside_soft_boundary =
      std::abs(output.estimate.position_m) >= config_.position_soft_limit_m ||
      (latest_visual_position_valid_ &&
       std::abs(latest_visual_position_m_) >= config_.position_soft_limit_m);
  const double prediction_horizon_s =
      std::max(0.0, config_.vision_loss_safe_s - vision_age_s);
  const bool predicted_soft_boundary =
      vision_soft_hold &&
      hold_prediction_crosses_soft_boundary(output.estimate, actual_u,
                                            chassis_acceleration, prediction_horizon_s);
  if (vision_soft_hold && (outside_soft_boundary || predicted_soft_boundary)) {
    output.request_stop = true;
    output.reason =
        outside_soft_boundary ? "vision_lost_outside_soft_boundary"
                              : "vision_lost_predicted_soft_boundary";
  }
  if (!contest3_waiting_for_start &&
      (std::abs(output.estimate.position_m) > config_.position_safe_limit_m ||
       (latest_visual_position_valid_ &&
        std::abs(latest_visual_position_m_) > config_.position_safe_limit_m))) {
    output.request_stop = true;
    output.reason = "ball_safety_boundary";
  }
  if (output.request_stop) latch_safety(output.reason, dmmc_stale);
  if (safety_latched_) {
    output.request_stop = true;
    output.reason = last_stop_reason_;
  } else if (vision_soft_hold) {
    output.reason = "vision_soft_hold";
  }

  const bool force_zero_command =
      safety_latched_ || vision_soft_hold || dmmc_stale || contest3_waiting_for_start;
  requested_u = force_zero_command ? 0.0 : rate_limit_and_clamp(requested_u);
  previous_command_u_ = requested_u;
  output.u_command_m_s2 = requested_u;
  output.solver_failures = solver_failures_;
  output.safety_latched = safety_latched_;
  output.safety_event_id = safety_event_id_;
  output.last_stop_reason = last_stop_reason_;

  ControlCommand command;
  if (command_id.has_value()) {
    command.command_id = *command_id;
    command_id_ = *command_id;
  } else {
    if (command_id_ == std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("command_id space exhausted; refusing to wrap or repeat");
    }
    command.command_id = ++command_id_;
  }
  command.source_frame_id = last_vision_frame_id_;
  command.nx_time_ms = static_cast<std::uint32_t>(std::llround(now_s * 1000.0));
  command.theta_cmd_rad = std::atan2(requested_u, kGravity);
  command.theta_rate_limit_rad_s = config_.theta_rate_limit_rad_s;
  command.ttl_ms = 60;
  command.control_state =
      safety_latched_
          ? (safety_fault_latched_ ? TaskState::Fault : TaskState::Safe)
          : ((vision_soft_hold || contest3_waiting_for_start)
                 ? TaskState::StandbyHold
                 : task_manager_.state());
  const bool restart_first_frame =
      clear_comm_warning_pending_ &&
      command.control_state != TaskState::Idle &&
      command.control_state != TaskState::Safe &&
      command.control_state != TaskState::Fault;
  if (restart_first_frame) {
    command.flags = 0x03U;
    clear_comm_warning_pending_ = false;
  } else {
    command.flags = dmmc_stale ? 0U : 0x01U;
    if (output.request_slowdown) command.flags |= 0x04U;
    if (output.request_stop) command.flags |= 0x08U;
  }
  output.command = command;
  return output;
}

}  // namespace nx_control
