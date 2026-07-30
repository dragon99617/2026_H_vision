#pragma once

#include "nx_control/types.hpp"

#include <vector>

namespace nx_control {

class TaskManager {
 public:
  void configure(TaskMode mode, double target_m, bool start_immediately,
                 bool start_on_chassis_event = true);
  void start(double now_s);
  void stop();
  void force_safe(bool fault);
  ReferencePoint update(double now_s, const ObserverState& estimate,
                        const ChassisState* chassis);
  std::vector<ReferencePoint> reference_horizon(int horizon) const;

  TaskState state() const { return state_; }
  TaskMode mode() const { return mode_; }
  bool static_sequence_complete() const { return static_stage_ >= 2; }
  double target_m() const { return current_reference_.position_m; }

 private:
  bool is_static_sequence() const;
  bool is_vehicle_task() const;

  TaskMode mode_ = TaskMode::Idle;
  TaskState state_ = TaskState::Idle;
  ReferencePoint current_reference_;
  double requested_target_m_ = 0.0;
  double task_started_s_ = 0.0;
  double stable_since_s_ = -1.0;
  int static_stage_ = 0;
  std::uint16_t previous_events_ = 0;
  bool armed_ = false;
  bool start_on_chassis_event_ = true;
};

}  // namespace nx_control
