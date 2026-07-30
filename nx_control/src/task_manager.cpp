#include "nx_control/task_manager.hpp"

#include <cmath>

namespace nx_control {

TaskManager::TaskManager(const ControlConfig& config)
    : hold_enter_position_error_m_(config.hold_enter_position_error_m),
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
  previous_events_ = 0;
  armed_ = true;
  start_on_chassis_event_ = start_on_chassis_event;
  target_hold_deadband_active_ = false;
  state_ = TaskState::Idle;
  current_reference_ = ReferencePoint{};
  if (start_immediately) start(0.0);
}

void TaskManager::start(double now_s) {
  if (!armed_ || mode_ == TaskMode::Idle) return;
  task_started_s_ = now_s;
  stable_since_s_ = -1.0;
  static_stage_ = 0;
  target_hold_deadband_active_ = false;
  switch (mode_) {
    case TaskMode::StaticSequence:
    case TaskMode::Contest3:
      state_ = TaskState::StaticMove;
      current_reference_.position_m = 0.05;
      break;
    case TaskMode::HoldCenter:
    case TaskMode::Contest45:
      state_ = TaskState::HoldCenter;
      current_reference_.position_m = 0.0;
      break;
    case TaskMode::HoldTarget:
    case TaskMode::Contest6:
      state_ = TaskState::HoldTarget;
      current_reference_.position_m = requested_target_m_;
      break;
    case TaskMode::AutoVehicle:
      state_ = TaskState::HoldCenter;
      current_reference_.position_m = requested_target_m_;
      break;
    case TaskMode::Idle:
      break;
  }
}

void TaskManager::stop() {
  state_ = TaskState::Idle;
  current_reference_ = ReferencePoint{};
  target_hold_deadband_active_ = false;
}

void TaskManager::force_safe(bool fault) {
  state_ = fault ? TaskState::Fault : TaskState::Safe;
  target_hold_deadband_active_ = false;
}

ReferencePoint TaskManager::update(double now_s, const ObserverState& estimate,
                                   const ChassisState* chassis) {
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
    const bool within_band = std::abs(estimate.position_m - current_reference_.position_m) <= 0.01 &&
                             std::abs(estimate.velocity_m_s) <= 0.03;
    if (within_band) {
      if (stable_since_s_ < 0.0) stable_since_s_ = now_s;
      if (now_s - stable_since_s_ >= 0.20 && static_stage_ == 0) {
        static_stage_ = 1;
        current_reference_.position_m = -0.05;
        stable_since_s_ = -1.0;
      } else if (now_s - stable_since_s_ >= 0.20 && static_stage_ == 1) {
        static_stage_ = 2;
        state_ = TaskState::HoldTarget;
      }
    } else {
      stable_since_s_ = -1.0;
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
  return std::vector<ReferencePoint>(static_cast<std::size_t>(horizon), current_reference_);
}

}  // namespace nx_control
