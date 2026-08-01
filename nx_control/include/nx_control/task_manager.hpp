#pragma once

#include "nx_control/types.hpp"

#include <algorithm>
#include <array>

namespace nx_control {

class TaskManager {
 public:
  explicit TaskManager(const ControlConfig& config = ControlConfig{});

  void configure(TaskMode mode, double target_m, bool start_immediately,
                 bool start_on_chassis_event = true);
  void start(double now_s);
  void stop();
  void force_safe(bool fault);
  ReferencePoint update(double now_s, const ObserverState& estimate,
                        const ChassisState* chassis,
                        const TubeStatus* tube_status = nullptr,
                        bool feedback_valid = false);
  TaskState state() const { return state_; }
  TaskMode mode() const { return mode_; }
  bool static_sequence_complete() const {
    return mode_ == TaskMode::Contest3 ? static_stage_ >= 3
                                       : static_stage_ >= 2;
  }
  bool target_hold_deadband_active() const { return target_hold_deadband_active_; }
  double target_m() const { return segment_target_m_; }
  int task3_stage() const { return mode_ == TaskMode::Contest3 ? static_stage_ : -1; }
  bool task3_balance_active() const {
    return mode_ == TaskMode::Contest3 && static_stage_ == 2 &&
           state_ == TaskState::StaticMove;
  }
  double task3_balance_elapsed_s() const {
    if (task3_balance_started_s_ < 0.0) return 0.0;
    return static_stage_ >= 3
               ? settle_elapsed_s_
               : std::max(0.0, last_update_s_ - task3_balance_started_s_);
  }
  bool settle_position_ok() const { return settle_position_ok_; }
  bool settle_velocity_ok() const { return settle_velocity_ok_; }
  bool settle_theta_ok() const { return settle_theta_ok_; }
  double settle_elapsed_s() const { return settle_elapsed_s_; }

 private:
  struct QuinticSegment {
    std::array<double, 6> coefficient{};
    double start_s = 0.0;
    double duration_s = 0.0;
    double target_position_m = 0.0;
    bool valid = false;
  };

  bool is_static_sequence() const;
  bool is_vehicle_task() const;
  void begin_static_segment(double now_s, const ReferencePoint& start,
                            double target_position_m);
  QuinticSegment make_quintic_segment(double now_s, const ReferencePoint& start,
                                      double target_position_m) const;
  ReferencePoint evaluate_segment(double time_s) const;
  bool segment_complete(double now_s) const;

  ControlConfig config_;
  TaskMode mode_ = TaskMode::Idle;
  TaskState state_ = TaskState::Idle;
  ReferencePoint current_reference_;
  QuinticSegment static_segment_;
  double requested_target_m_ = 0.0;
  double segment_target_m_ = 0.0;
  double task_started_s_ = 0.0;
  double last_update_s_ = 0.0;
  double stable_since_s_ = -1.0;
  double settle_elapsed_s_ = 0.0;
  double task3_positive_reached_s_ = -1.0;
  double task3_balance_started_s_ = -1.0;
  int static_stage_ = 0;
  std::uint16_t previous_events_ = 0;
  bool armed_ = false;
  bool start_on_chassis_event_ = true;
  bool target_hold_deadband_active_ = false;
  bool segment_clock_started_ = false;
  bool settle_position_ok_ = false;
  bool settle_velocity_ok_ = false;
  bool settle_theta_ok_ = false;
  double hold_enter_position_error_m_ = 0.004;
  double hold_enter_velocity_m_s_ = 0.015;
  double hold_exit_position_error_m_ = 0.008;
};

}  // namespace nx_control
