#include "nx_control/task_manager.hpp"

#include <algorithm>
#include <cmath>

namespace nx_control {
namespace {

std::array<double, 6> quintic_coefficients(const ReferencePoint& start,
                                           double target_position_m,
                                           double duration_s) {
  const double t = duration_s;
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double t4 = t3 * t;
  const double t5 = t4 * t;
  const double p0 = start.position_m;
  const double v0 = start.velocity_m_s;
  const double a0 = start.acceleration_m_s2;
  const double p1 = target_position_m;
  constexpr double v1 = 0.0;
  constexpr double a1 = 0.0;
  return {
      p0,
      v0,
      0.5 * a0,
      (20.0 * (p1 - p0) - (8.0 * v1 + 12.0 * v0) * t -
       (3.0 * a0 - a1) * t2) /
          (2.0 * t3),
      (30.0 * (p0 - p1) + (14.0 * v1 + 16.0 * v0) * t +
       (3.0 * a0 - 2.0 * a1) * t2) /
          (2.0 * t4),
      (12.0 * (p1 - p0) - (6.0 * v1 + 6.0 * v0) * t -
       (a0 - a1) * t2) /
          (2.0 * t5),
  };
}

ReferencePoint evaluate_quintic(const std::array<double, 6>& c, double t) {
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double t4 = t3 * t;
  const double t5 = t4 * t;
  return {
      c[0] + c[1] * t + c[2] * t2 + c[3] * t3 + c[4] * t4 + c[5] * t5,
      c[1] + 2.0 * c[2] * t + 3.0 * c[3] * t2 + 4.0 * c[4] * t3 +
          5.0 * c[5] * t4,
      2.0 * c[2] + 6.0 * c[3] * t + 12.0 * c[4] * t2 +
          20.0 * c[5] * t3,
  };
}

double evaluate_jerk(const std::array<double, 6>& c, double t) {
  return 6.0 * c[3] + 24.0 * c[4] * t + 60.0 * c[5] * t * t;
}

}  // namespace

TaskManager::TaskManager(const ControlConfig& config)
    : config_(config),
      hold_enter_position_error_m_(config.hold_enter_position_error_m),
      hold_enter_velocity_m_s_(config.hold_enter_velocity_m_s),
      hold_exit_position_error_m_(config.hold_exit_position_error_m) {}

bool TaskManager::is_static_sequence() const {
  return mode_ == TaskMode::StaticSequence || mode_ == TaskMode::Contest3;
}

bool TaskManager::is_vehicle_task() const {
  return mode_ == TaskMode::AutoVehicle || mode_ == TaskMode::Contest45 ||
         mode_ == TaskMode::Contest6;
}

void TaskManager::configure(TaskMode mode, double target_m, bool start_immediately,
                            bool start_on_chassis_event) {
  mode_ = mode;
  requested_target_m_ =
      mode == TaskMode::Contest45 || mode == TaskMode::Contest3 ? 0.0 : target_m;
  static_stage_ = 0;
  stable_since_s_ = -1.0;
  settle_elapsed_s_ = 0.0;
  settle_position_ok_ = settle_velocity_ok_ = settle_theta_ok_ = false;
  previous_events_ = 0;
  armed_ = true;
  start_on_chassis_event_ = start_on_chassis_event;
  target_hold_deadband_active_ = false;
  state_ = TaskState::Idle;
  current_reference_ = ReferencePoint{};
  static_segment_ = QuinticSegment{};
  segment_target_m_ = 0.0;
  segment_clock_started_ = false;
  if (start_immediately) start(0.0);
}

void TaskManager::start(double now_s) {
  if (!armed_ || mode_ == TaskMode::Idle) return;
  const ReferencePoint segment_start =
      state_ == TaskState::Idle ? ReferencePoint{} : current_reference_;
  task_started_s_ = now_s;
  last_update_s_ = now_s;
  stable_since_s_ = -1.0;
  settle_elapsed_s_ = 0.0;
  settle_position_ok_ = settle_velocity_ok_ = settle_theta_ok_ = false;
  static_stage_ = 0;
  target_hold_deadband_active_ = false;
  switch (mode_) {
    case TaskMode::StaticSequence:
    case TaskMode::Contest3:
      state_ = TaskState::StaticMove;
      begin_static_segment(now_s, segment_start, 0.05);
      break;
    case TaskMode::HoldCenter:
    case TaskMode::Contest45:
      state_ = TaskState::HoldCenter;
      current_reference_.position_m = 0.0;
      segment_target_m_ = 0.0;
      break;
    case TaskMode::HoldTarget:
    case TaskMode::Contest6:
      state_ = TaskState::HoldTarget;
      current_reference_.position_m = requested_target_m_;
      segment_target_m_ = requested_target_m_;
      break;
    case TaskMode::AutoVehicle:
      state_ = TaskState::HoldCenter;
      current_reference_.position_m = requested_target_m_;
      segment_target_m_ = requested_target_m_;
      break;
    case TaskMode::Idle:
      break;
  }
}

void TaskManager::stop() {
  state_ = TaskState::Idle;
  current_reference_ = ReferencePoint{};
  segment_target_m_ = 0.0;
  target_hold_deadband_active_ = false;
  stable_since_s_ = -1.0;
  settle_elapsed_s_ = 0.0;
  settle_position_ok_ = settle_velocity_ok_ = settle_theta_ok_ = false;
}

void TaskManager::force_safe(bool fault) {
  state_ = fault ? TaskState::Fault : TaskState::Safe;
  target_hold_deadband_active_ = false;
}

TaskManager::QuinticSegment TaskManager::make_quintic_segment(
    double now_s, const ReferencePoint& start, double target_position_m) const {
  const double distance = std::abs(target_position_m - start.position_m);
  double duration_s = std::max(
      config_.period_s,
      std::max({1.875 * distance / config_.task3_reference_max_velocity_m_s,
                std::sqrt(5.773502691896258 * distance /
                          config_.task3_reference_max_acceleration_m_s2),
                std::cbrt(60.0 * distance /
                          config_.task3_reference_max_jerk_m_s3)}));
  if (distance < 1e-12 && std::abs(start.velocity_m_s) < 1e-12 &&
      std::abs(start.acceleration_m_s2) < 1e-12) {
    duration_s = config_.period_s;
  }

  std::array<double, 6> coefficient{};
  for (int attempt = 0; attempt < 80; ++attempt) {
    coefficient =
        quintic_coefficients(start, target_position_m, duration_s);
    double max_velocity = 0.0;
    double max_acceleration = 0.0;
    double max_jerk = 0.0;
    for (int sample = 0; sample <= 400; ++sample) {
      const double t = duration_s * static_cast<double>(sample) / 400.0;
      const ReferencePoint point = evaluate_quintic(coefficient, t);
      max_velocity = std::max(max_velocity, std::abs(point.velocity_m_s));
      max_acceleration =
          std::max(max_acceleration, std::abs(point.acceleration_m_s2));
      max_jerk = std::max(max_jerk, std::abs(evaluate_jerk(coefficient, t)));
    }
    if (max_velocity <= config_.task3_reference_max_velocity_m_s * (1.0 + 1e-9) &&
        max_acceleration <=
            config_.task3_reference_max_acceleration_m_s2 * (1.0 + 1e-9) &&
        max_jerk <= config_.task3_reference_max_jerk_m_s3 * (1.0 + 1e-9)) {
      return QuinticSegment{coefficient, now_s, duration_s,
                            target_position_m, true};
    }
    duration_s *= 1.05;
  }
  coefficient = quintic_coefficients(start, target_position_m, duration_s);
  return QuinticSegment{coefficient, now_s, duration_s, target_position_m,
                        true};
}

void TaskManager::begin_static_segment(double now_s,
                                       const ReferencePoint& start,
                                       double target_position_m) {
  static_segment_ = make_quintic_segment(now_s, start, target_position_m);
  segment_target_m_ = target_position_m;
  current_reference_ = start;
  segment_clock_started_ = now_s > 0.0;
}

ReferencePoint TaskManager::evaluate_segment(double time_s) const {
  if (!static_segment_.valid) return current_reference_;
  const double elapsed =
      std::clamp(time_s - static_segment_.start_s, 0.0,
                 static_segment_.duration_s);
  if (elapsed >= static_segment_.duration_s) {
    return ReferencePoint{static_segment_.target_position_m, 0.0, 0.0};
  }
  return evaluate_quintic(static_segment_.coefficient, elapsed);
}

bool TaskManager::segment_complete(double now_s) const {
  return static_segment_.valid && segment_clock_started_ &&
         now_s - static_segment_.start_s >= static_segment_.duration_s;
}

ReferencePoint TaskManager::update(double now_s, const ObserverState& estimate,
                                   const ChassisState* chassis,
                                   const TubeStatus* tube_status,
                                   bool feedback_valid) {
  last_update_s_ = now_s;
  if (state_ == TaskState::Safe || state_ == TaskState::Fault) return current_reference_;
  if (chassis != nullptr) {
    const std::uint16_t rising = static_cast<std::uint16_t>(chassis->events & ~previous_events_);
    previous_events_ = chassis->events;
    if (start_on_chassis_event_ && (rising & 0x0001U) != 0U &&
        state_ == TaskState::Idle) {
      start(now_s);
    }
  }
  if (state_ != TaskState::Idle && task_started_s_ == 0.0) task_started_s_ = now_s;

  if (is_static_sequence() && state_ == TaskState::StaticMove) {
    if (!segment_clock_started_) {
      static_segment_.start_s = now_s;
      segment_clock_started_ = true;
    }
    current_reference_ = evaluate_segment(now_s);
    settle_position_ok_ =
        std::abs(estimate.position_m - segment_target_m_) <=
        config_.task3_settle_position_error_m;
    settle_velocity_ok_ =
        std::abs(estimate.velocity_m_s) <= config_.task3_settle_velocity_m_s;
    const bool require_theta_settle =
        mode_ == TaskMode::Contest3 && static_stage_ != 0;
    settle_theta_ok_ =
        !require_theta_settle ||
        (feedback_valid && tube_status != nullptr &&
         std::abs(tube_status->theta_actual_rad -
                  config_.task3_theta_bias_rad) <=
             config_.task3_settle_theta_tolerance_rad);
    const bool valid_for_settle =
        mode_ != TaskMode::Contest3 || feedback_valid;
    const bool within_band =
        segment_complete(now_s) && valid_for_settle && settle_position_ok_ &&
        settle_velocity_ok_ && settle_theta_ok_;
    if (within_band) {
      if (stable_since_s_ < 0.0) stable_since_s_ = now_s;
      settle_elapsed_s_ = std::max(0.0, now_s - stable_since_s_);
      if (settle_elapsed_s_ >= config_.task3_settle_dwell_s &&
          static_stage_ == 0) {
        static_stage_ = 1;
        begin_static_segment(now_s, current_reference_, -0.05);
        stable_since_s_ = -1.0;
      } else if (settle_elapsed_s_ >= config_.task3_settle_dwell_s &&
                 static_stage_ == 1) {
        static_stage_ = 2;
        state_ = TaskState::HoldTarget;
      }
    } else {
      stable_since_s_ = -1.0;
      settle_elapsed_s_ = 0.0;
    }
  } else if (is_vehicle_task() && chassis != nullptr && state_ != TaskState::Idle) {
    switch (chassis->motion_phase) {
      case MotionPhase::Accel:
        state_ = TaskState::VehicleAccel;
        break;
      case MotionPhase::Decel:
        state_ = TaskState::VehicleDecel;
        break;
      case MotionPhase::Cruise:
      case MotionPhase::Curve:
        state_ = TaskState::VehicleCruise;
        break;
      case MotionPhase::Stop:
        state_ = requested_target_m_ == 0.0 ? TaskState::HoldCenter : TaskState::HoldTarget;
        break;
    }
  }

  if (state_ == TaskState::HoldTarget) {
    const double position_error =
        std::abs(estimate.position_m - current_reference_.position_m);
    if (target_hold_deadband_active_) {
      if (position_error > hold_exit_position_error_m_) {
        target_hold_deadband_active_ = false;
      }
    } else if (position_error < hold_enter_position_error_m_ &&
               std::abs(estimate.velocity_m_s) < hold_enter_velocity_m_s_) {
      target_hold_deadband_active_ = true;
    }
  } else {
    target_hold_deadband_active_ = false;
  }
  return current_reference_;
}

std::vector<ReferencePoint> TaskManager::reference_horizon(int horizon) const {
  std::vector<ReferencePoint> result;
  result.reserve(static_cast<std::size_t>(std::max(0, horizon)));
  for (int step = 0; step < horizon; ++step) {
    if (is_static_sequence() && state_ == TaskState::StaticMove) {
      result.push_back(
          evaluate_segment(last_update_s_ + (step + 1) * config_.period_s));
    } else {
      result.push_back(current_reference_);
    }
  }
  return result;
}

}  // namespace nx_control
