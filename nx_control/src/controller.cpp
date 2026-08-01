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

NxController::NxController(const ControlConfig& config)
    : config_(config),
      observer_(config),
      chassis_sync_(config),
      pid_(config),
      task_manager_(config),
      friction_compensator_(config) {}

void NxController::configure_task(TaskMode mode, double target_m, bool start_immediately,
                                  bool start_on_chassis_event) {
  pid_.reset();
  inner_angle_warning_since_s_ = -1.0;
  task_manager_.configure(mode, target_m, start_immediately, start_on_chassis_event);
}

void NxController::start_task(double now_s) {
  safety_latched_ = false;
  safety_fault_latched_ = false;
  clear_comm_warning_pending_ = true;
  previous_applied_u_ = 0.0;
  previous_theta_command_rad_ = 0.0;
  previous_model_compensation_rad_ = 0.0;
  previous_model_compensation_active_ = false;
  task3_early_brake_stage_ = -1;
  task3_early_braking_active_ = false;
  task3_early_braking_done_ = false;
  task3_reverse_balance_active_ = false;
  task3_reverse_balance_done_ = false;
  visual_position_filter_initialized_ = false;
  friction_compensator_.reset();
  inner_angle_warning_since_s_ = -1.0;
  pid_.reset();
  task_manager_.start(now_s);
}

void NxController::stop_task() {
  task_manager_.stop();
  previous_applied_u_ = 0.0;
  previous_theta_command_rad_ = 0.0;
  previous_model_compensation_rad_ = 0.0;
  previous_model_compensation_active_ = false;
  task3_early_brake_stage_ = -1;
  task3_early_braking_active_ = false;
  task3_early_braking_done_ = false;
  task3_reverse_balance_active_ = false;
  task3_reverse_balance_done_ = false;
  friction_compensator_.reset();
  inner_angle_warning_since_s_ = -1.0;
  pid_.reset();
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
  latest_vision_capture_age_ms_ = std::numeric_limits<double>::infinity();
  filtered_visual_position_m_ = 0.0;
  visual_position_filter_time_s_ = 0.0;
  visual_position_filter_initialized_ = false;
  previous_applied_u_ = 0.0;
  previous_theta_command_rad_ = 0.0;
  previous_model_compensation_rad_ = 0.0;
  previous_model_compensation_active_ = false;
  task3_early_brake_stage_ = -1;
  task3_early_braking_active_ = false;
  task3_early_braking_done_ = false;
  task3_reverse_balance_active_ = false;
  task3_reverse_balance_done_ = false;
  friction_compensator_.reset();
  inner_angle_warning_since_s_ = -1.0;
  pid_.reset();
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
    visual_position_filter_initialized_ = false;
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
          v2_capture_time_s_ +
          static_cast<double>(std::max<std::int32_t>(1, frame_delta)) /
                                     config_.vision_frame_rate_hz;
      v2_capture_time_s_ = frame_estimate + 0.05 * (arrival_estimate - frame_estimate);
    }
    measurement.capture_time_s = v2_capture_time_s_;
  }
  // Vision and controller use CLOCK_MONOTONIC. Propagate to an inter-tick exposure time.
  measurement.capture_time_s = std::min(measurement.capture_time_s, measurement.receive_time_s);
  latest_vision_capture_age_ms_ =
      1000.0 *
      std::max(0.0, measurement.receive_time_s - measurement.capture_time_s);

  // Bypass smoothing near the soft boundary so slowdown decisions see the
  // measured position without additional filter lag.
  const double raw_position_m = measurement.position_m;
  const bool visual_position_valid =
      measurement.status == VisionStatus::Measured &&
      measurement.ball_confidence * measurement.tube_confidence >= 0.30;
  if (visual_position_valid) {
    const bool near_soft_boundary =
        std::abs(raw_position_m) >= config_.position_soft_limit_m - 0.010;
    if (config_.vision_position_filter_tau_s > 0.0 &&
        !near_soft_boundary) {
      const double dt_s =
          measurement.capture_time_s - visual_position_filter_time_s_;
      if (!visual_position_filter_initialized_ || dt_s <= 0.0 ||
          dt_s > config_.vision_loss_hold_s) {
        filtered_visual_position_m_ = raw_position_m;
      } else {
        const double alpha =
            1.0 - std::exp(-dt_s / config_.vision_position_filter_tau_s);
        filtered_visual_position_m_ +=
            alpha * (raw_position_m - filtered_visual_position_m_);
      }
      visual_position_filter_time_s_ = measurement.capture_time_s;
      visual_position_filter_initialized_ = true;
      measurement.position_m = filtered_visual_position_m_;
    } else {
      filtered_visual_position_m_ = raw_position_m;
      visual_position_filter_time_s_ = measurement.capture_time_s;
      visual_position_filter_initialized_ = true;
    }
  } else {
    visual_position_filter_initialized_ = false;
  }

  if (measurement.capture_time_s > observer_.time_s()) {
    const double actual_u =
        have_tube_status_
            ? model_u_from_actual_theta(tube_status_.theta_actual_rad)
            : previous_applied_u_;
    const double acceleration =
        chassis_acceleration_for_control(measurement.receive_time_s);
    observer_.predict(measurement.capture_time_s, actual_u, acceleration);
  }
  if (observer_.update_position(measurement.capture_time_s, measurement.position_m,
                                measurement.ball_confidence, measurement.tube_confidence,
                                measurement.status)) {
    last_accepted_vision_s_ = measurement.receive_time_s;
  }
}

void NxController::ingest_tube_status(const TubeStatus& status) {
  if (have_tube_status_ && !sequence_is_newer(status.sequence, last_tube_sequence_)) {
    if (status.receive_time_s - tube_status_.receive_time_s <= config_.dmmc_stale_s) return;
    dmmc_clock_.reset();
  }
  TubeStatus synchronized = status;
  synchronized.sample_time_s =
      dmmc_clock_.to_local_seconds(status.dmmc_time_ms, status.receive_time_s);
  const double previous_u =
      have_tube_status_
          ? model_u_from_actual_theta(tube_status_.theta_actual_rad)
          : previous_applied_u_;
  observer_.predict(synchronized.sample_time_s, previous_u,
                    chassis_acceleration_for_control(status.receive_time_s));
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
  const double actual_u =
      have_tube_status_
          ? model_u_from_actual_theta(tube_status_.theta_actual_rad)
          : previous_applied_u_;
  observer_.predict(synchronized.sample_time_s, actual_u,
                    chassis_acceleration_for_control(state.receive_time_s));
  last_chassis_sequence_ = state.sequence;
  have_chassis_sequence_ = true;
  chassis_sync_.ingest(synchronized);
}

double NxController::model_u_from_actual_theta(double theta_actual_rad) const {
  const double dynamic_theta_rad =
      previous_model_compensation_active_
          ? theta_actual_rad - previous_model_compensation_rad_
          : theta_actual_rad;
  return kGravity * std::tan(dynamic_theta_rad);
}

double NxController::rate_limit_final_angle(
    double requested_theta_rad, double rate_limit_rad_s) const {
  const double max_step_rad =
      rate_limit_rad_s * config_.period_s;
  return std::clamp(
      std::clamp(requested_theta_rad,
                 previous_theta_command_rad_ - max_step_rad,
                 previous_theta_command_rad_ + max_step_rad),
      -config_.theta_limit_rad, config_.theta_limit_rad);
}

bool NxController::contest_uses_camera_feedback_only() const {
  const TaskMode mode = task_manager_.mode();
  return mode == TaskMode::Contest45 || mode == TaskMode::Contest6;
}

double NxController::chassis_acceleration_for_control(double now_s) const {
  return contest_uses_camera_feedback_only()
             ? 0.0
             : chassis_sync_.delay_compensated_actual_acceleration(now_s);
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
  const bool camera_feedback_only = contest_uses_camera_feedback_only();
  const double chassis_acceleration =
      camera_feedback_only ? 0.0 : chassis_acceleration_for_control(now_s);
  const double actual_u =
      have_tube_status_
          ? model_u_from_actual_theta(tube_status_.theta_actual_rad)
          : previous_applied_u_;
  observer_.predict(now_s, actual_u, chassis_acceleration);
  last_tick_s_ = now_s;

  ControlOutput output;
  output.estimate = observer_.state();
  output.acceleration_used_m_s2 = chassis_acceleration;
  output.vision_age_ms = last_accepted_vision_s_ >= 0.0
                             ? 1000.0 * std::max(0.0, now_s - last_accepted_vision_s_)
                             : std::numeric_limits<double>::infinity();
  output.vision_capture_age_ms = latest_vision_capture_age_ms_;
  output.chassis_age_ms = 1000.0 * chassis_sync_.age_s(now_s);
  output.dmmc_age_ms = have_tube_status_
                           ? 1000.0 * std::max(0.0, now_s - tube_status_.receive_time_s)
                           : std::numeric_limits<double>::infinity();

  const double vision_age_s = output.vision_age_ms / 1000.0;
  const bool dmmc_stale =
      !have_tube_status_ ||
      output.dmmc_age_ms > config_.dmmc_stale_s * 1000.0 ||
      tube_status_.faults != 0U;
  const bool task_feedback_valid =
      !dmmc_stale && vision_age_s <= config_.vision_loss_hold_s;
  const ChassisState* chassis =
      !camera_feedback_only && chassis_valid ? &chassis_sync_.latest() : nullptr;
  output.reference =
      task_manager_.update(now_s, output.estimate, chassis,
                           have_tube_status_ ? &tube_status_ : nullptr,
                           task_feedback_valid);
  output.task3_stage = task_manager_.task3_stage();
  output.settle_position_ok = task_manager_.settle_position_ok();
  output.settle_velocity_ok = task_manager_.settle_velocity_ok();
  output.settle_theta_ok = task_manager_.settle_theta_ok();
  output.settle_elapsed_ms = 1000.0 * task_manager_.settle_elapsed_s();
  const bool contest3_waiting_for_start =
      task_manager_.mode() == TaskMode::Contest3 && task_manager_.state() == TaskState::Idle;
  const bool task_safety_active =
      task_manager_.state() != TaskState::Idle || contest3_waiting_for_start;
  const bool target_hold_deadband = task_manager_.target_hold_deadband_active();

  const bool vision_soft_hold =
      task_safety_active && !contest3_waiting_for_start &&
      vision_age_s >= config_.vision_loss_hold_s &&
      vision_age_s <= config_.vision_loss_safe_s;
  const bool vision_hard_lost =
      task_safety_active && !contest3_waiting_for_start &&
      vision_age_s > config_.vision_loss_safe_s;
  const bool force_safe_feedforward =
      vision_soft_hold || vision_hard_lost || safety_latched_;
  if (vision_hard_lost) {
    output.request_stop = true;
    output.reason = "vision_stale";
  }
  if (task_safety_active && dmmc_stale) {
    output.request_stop = true;
    output.reason = "dmmc_stale_or_fault";
  }

  const bool pid_active =
      !force_safe_feedforward && !dmmc_stale &&
      task_manager_.state() != TaskState::Idle;
  if (!pid_active) pid_.reset();
  output.pid = pid_.calculate(output.estimate, output.reference,
                              chassis_acceleration,
                              pid_active && !target_hold_deadband);
  double requested_u = pid_active ? output.pid.unsaturated_m_s2 : 0.0;
  if (pid_active && !target_hold_deadband) {
    if (vision_age_s > config_.vision_decay_start_s) {
      const double correction_scale = std::clamp(
          (config_.vision_loss_hold_s - vision_age_s) /
              std::max(1e-6, config_.vision_loss_hold_s - config_.vision_decay_start_s),
          0.0, 1.0);
      requested_u = output.pid.feedforward_m_s2 +
                    correction_scale *
                        (requested_u - output.pid.feedforward_m_s2);
    }
  }

  if (target_hold_deadband) {
    requested_u = 0.0;
    if (!output.request_stop) output.reason = "hold_deadband";
  }

  const bool projected_soft_boundary =
      !contest3_waiting_for_start &&
      hold_prediction_crosses_soft_boundary(
          output.estimate, actual_u, chassis_acceleration,
          config_.vision_loss_safe_s);
  if (!contest3_waiting_for_start &&
      (std::abs(output.estimate.position_m) >
           config_.position_soft_limit_m - 0.010 ||
       projected_soft_boundary)) {
    output.request_slowdown = true;
  }

  if (task_safety_active && !dmmc_stale) {
    output.inner_angle_error_rad =
        tube_status_.theta_reference_rad - tube_status_.theta_actual_rad;
    const double abs_inner_error = std::abs(output.inner_angle_error_rad);
    if (abs_inner_error > config_.inner_angle_warning_rad) {
      if (inner_angle_warning_since_s_ < 0.0) {
        inner_angle_warning_since_s_ = now_s;
      }
    } else {
      inner_angle_warning_since_s_ = -1.0;
    }
    output.inner_angle_warning =
        inner_angle_warning_since_s_ >= 0.0 &&
        now_s - inner_angle_warning_since_s_ >=
            config_.inner_angle_warning_dwell_s;
    if (output.inner_angle_warning) {
      output.request_slowdown = true;
      if (output.reason.empty()) output.reason = "inner_angle_tracking_warning";
    }
  } else {
    inner_angle_warning_since_s_ = -1.0;
  }
  if (output.request_stop) latch_safety(output.reason, dmmc_stale);
  if (safety_latched_) {
    output.request_stop = true;
    output.reason = last_stop_reason_;
  } else if (vision_soft_hold) {
    output.reason = "vision_soft_hold";
  }

  const bool force_zero_command =
      safety_latched_ || vision_soft_hold || dmmc_stale ||
      contest3_waiting_for_start || !pid_active;
  if (force_zero_command) requested_u = 0.0;

  const bool task3_compensation_active =
      !force_zero_command && !output.request_stop &&
      task_manager_.mode() == TaskMode::Contest3 &&
      (task_manager_.state() == TaskState::StaticMove ||
       task_manager_.state() == TaskState::HoldTarget);
  const int task3_stage = task_manager_.task3_stage();
  const bool task3_move_active =
      task3_compensation_active &&
      task_manager_.state() == TaskState::StaticMove &&
      (task3_stage == 0 || task3_stage == 1);
  // The positive pass is handled independently so braking can continue across
  // the stage-0 -> stage-1 transition at +4 cm. The latched one-shot remains
  // dedicated to the negative pass.
  if (!task3_move_active || task3_stage != 1) {
    task3_early_brake_stage_ = -1;
    task3_early_braking_active_ = false;
    task3_early_braking_done_ = false;
  } else {
    if (task3_stage != task3_early_brake_stage_) {
      task3_early_brake_stage_ = task3_stage;
      task3_early_braking_active_ = false;
      task3_early_braking_done_ = false;
    }
    constexpr int motion_direction = -1;
    if (!task3_early_braking_active_ &&
        !task3_early_braking_done_ &&
        motion_direction * output.estimate.position_m >=
            config_.task3_early_brake_position_m &&
        motion_direction * output.estimate.velocity_m_s >
            config_.task3_settle_velocity_m_s) {
      task3_early_braking_active_ = true;
    }
    if (task3_early_braking_active_) {
      const double forward_velocity_m_s =
          motion_direction * output.estimate.velocity_m_s;
      if (forward_velocity_m_s <= config_.task3_settle_velocity_m_s) {
        task3_early_braking_active_ = false;
        task3_early_braking_done_ = true;
      }
    }
  }
  if (!task3_move_active || task3_stage != 1) {
    task3_reverse_balance_active_ = false;
    task3_reverse_balance_done_ = false;
  } else if (task3_early_braking_active_) {
    // The one-shot -4 cm emergency brake has priority over return balancing
    // phase if the latter has not completed in time.
    task3_reverse_balance_active_ = false;
    task3_reverse_balance_done_ = true;
  } else if (!task3_reverse_balance_done_) {
    if (!task3_reverse_balance_active_) {
      if (output.estimate.velocity_m_s <=
          -config_.task3_positive_reverse_velocity_m_s) {
        task3_reverse_balance_active_ = true;
      }
    } else if (have_tube_status_ &&
               std::abs(tube_status_.theta_actual_rad -
                        config_.task3_theta_bias_rad) <=
                   config_.task3_settle_theta_tolerance_rad) {
      task3_reverse_balance_active_ = false;
      task3_reverse_balance_done_ = true;
    }
  }
  const bool task3_positive_approach_braking =
      task3_move_active && !task3_reverse_balance_active_ &&
      output.estimate.position_m >=
          config_.task3_positive_early_brake_position_m &&
      output.estimate.velocity_m_s > config_.task3_settle_velocity_m_s;
  const bool task3_positive_overshoot_recovery =
      task3_move_active && !task3_reverse_balance_active_ &&
      output.estimate.position_m >
          config_.task3_positive_overshoot_position_m;
  int task3_braking_direction = 0;
  double task3_braking_deceleration_m_s2 =
      config_.task3_braking_deceleration_m_s2;
  if (task3_positive_overshoot_recovery) {
    task3_braking_direction = 1;
    task3_braking_deceleration_m_s2 =
        config_.task3_positive_overshoot_deceleration_m_s2;
  } else if (task3_positive_approach_braking) {
    task3_braking_direction = 1;
  } else if (task3_early_braking_active_) {
    // Once the measured return crosses -4 cm, command acceleration opposite
    // to travel until the ball has slowed to the settle-speed band.
    task3_braking_direction = -1;
  }
  const bool task3_braking_active = task3_braking_direction != 0;
  if (task3_reverse_balance_active_) {
    // Zero dynamic acceleration and remove friction feedforward below so the
    // absolute command goes straight to the calibrated balance bias.
    requested_u = 0.0;
  } else if (task3_braking_active) {
    requested_u =
        -static_cast<double>(task3_braking_direction) *
        task3_braking_deceleration_m_s2 / config_.rolling_lambda;
  }
  const double target_position_error_m =
      task_manager_.target_m() - output.estimate.position_m;
  const FrictionCompensation friction = friction_compensator_.update(
      now_s,
      task3_compensation_active && !task3_reverse_balance_active_,
      target_position_error_m,
      output.estimate.velocity_m_s, requested_u,
      task3_braking_active ? 0.0 : output.reference.velocity_m_s);
  if (friction.target_deadband) requested_u = 0.0;

  const double theta_pid_rad =
      force_zero_command ? 0.0 : std::atan2(requested_u, kGravity);
  const double theta_bias_rad =
      task3_compensation_active ? config_.task3_theta_bias_rad : 0.0;
  const double theta_friction_rad =
      task3_compensation_active ? friction.theta_friction_rad : 0.0;
  const double desired_theta_rad =
      theta_pid_rad + theta_bias_rad + theta_friction_rad;
  const double theta_rate_limit_rad_s =
      task3_reverse_balance_active_
          ? config_.task3_reverse_balance_rate_limit_rad_s
          : config_.theta_rate_limit_rad_s;
  const double theta_command_rad =
      force_zero_command
          ? 0.0
          : rate_limit_final_angle(desired_theta_rad,
                                   theta_rate_limit_rad_s);

  const double applied_dynamic_theta_rad =
      theta_command_rad - theta_bias_rad - theta_friction_rad;
  const double applied_u =
      force_zero_command
          ? 0.0
          : kGravity * std::tan(applied_dynamic_theta_rad);
  const bool pid_forced_override =
      target_hold_deadband || task3_reverse_balance_active_ || task3_braking_active ||
      friction.target_deadband;
  if (force_zero_command) {
    pid_.reset();
    output.pid = pid_.last_result();
    output.pid.applied_m_s2 = 0.0;
    output.pid.integrator_frozen = true;
  } else {
    output.pid = pid_.track(applied_u, pid_forced_override);
  }

  previous_applied_u_ = applied_u;
  previous_theta_command_rad_ = theta_command_rad;
  previous_model_compensation_active_ = task3_compensation_active;
  previous_model_compensation_rad_ =
      task3_compensation_active ? theta_bias_rad + theta_friction_rad : 0.0;
  output.u_command_m_s2 = applied_u;
  output.theta_pid_rad = theta_pid_rad;
  output.theta_bias_rad = theta_bias_rad;
  output.theta_friction_rad = theta_friction_rad;
  output.friction_mode = friction.mode;
  output.friction_direction = friction.direction;
  output.task3_early_braking = task3_braking_active;
  output.task3_positive_overshoot_recovery =
      task3_positive_overshoot_recovery;
  output.task3_reverse_balance_active =
      task3_reverse_balance_active_;
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
  command.theta_cmd_rad = theta_command_rad;
  command.theta_rate_limit_rad_s = theta_rate_limit_rad_s;
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
