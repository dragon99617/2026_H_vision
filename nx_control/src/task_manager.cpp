#include "nx_control/task_manager.hpp"

#include <cmath>

namespace nx_control {

void TaskManager::configure(TaskMode mode, double target_m, bool start_immediately) {
  mode_ = mode;
  requested_target_m_ = target_m;
  static_stage_ = 0;
  stable_since_s_ = -1.0;
  previous_events_ = 0;
  armed_ = true;
  state_ = TaskState::Idle;
  current_reference_ = ReferencePoint{};
  if (start_immediately) start(0.0);
}

void TaskManager::start(double now_s) {
  if (!armed_ || mode_ == TaskMode::Idle) return;
  task_started_s_ = now_s;
  stable_since_s_ = -1.0;
  static_stage_ = 0;
  switch (mode_) {
    case TaskMode::StaticSequence:
      state_ = TaskState::StaticMove;
      current_reference_.position_m = 0.05;
      break;
    case TaskMode::HoldCenter:
      state_ = TaskState::HoldCenter;
      current_reference_.position_m = 0.0;
      break;
    case TaskMode::HoldTarget:
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
}

void TaskManager::force_safe(bool fault) { state_ = fault ? TaskState::Fault : TaskState::Safe; }

ReferencePoint TaskManager::update(double now_s, const ObserverState& estimate,
                                   const ChassisState* chassis) {
  if (state_ == TaskState::Safe || state_ == TaskState::Fault) return current_reference_;
  if (chassis != nullptr) {
    const std::uint16_t rising = static_cast<std::uint16_t>(chassis->events & ~previous_events_);
    previous_events_ = chassis->events;
    if ((rising & 0x0001U) != 0U && state_ == TaskState::Idle) start(now_s);
  }
  if (state_ != TaskState::Idle && task_started_s_ == 0.0) task_started_s_ = now_s;

  if (mode_ == TaskMode::StaticSequence && state_ == TaskState::StaticMove) {
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
  } else if (mode_ == TaskMode::AutoVehicle && chassis != nullptr && state_ != TaskState::Idle) {
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
  return current_reference_;
}

std::vector<ReferencePoint> TaskManager::reference_horizon(int horizon) const {
  return std::vector<ReferencePoint>(static_cast<std::size_t>(horizon), current_reference_);
}

}  // namespace nx_control
