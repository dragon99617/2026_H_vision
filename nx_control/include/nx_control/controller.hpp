#pragma once

#include "nx_control/friction_compensator.hpp"
#include "nx_control/observer.hpp"
#include "nx_control/pid.hpp"
#include "nx_control/task_manager.hpp"
#include "nx_control/types.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace nx_control {

class NxController {
 public:
  explicit NxController(const ControlConfig& config);

  void configure_task(TaskMode mode, double target_m, bool start_immediately,
                      bool start_on_chassis_event = true);
  void start_task(double now_s);
  void stop_task();
  void ingest_vision(VisionMeasurement measurement);
  void ingest_tube_status(const TubeStatus& status);
  void ingest_chassis_state(const ChassisState& state);
  ControlOutput tick(double now_s,
                     std::optional<std::uint32_t> command_id = std::nullopt);
  void reset(double now_s);

  const DelayedKalmanObserver& observer() const { return observer_; }
  const BallPid& pid() const { return pid_; }
  const TaskManager& task_manager() const { return task_manager_; }

 private:
  double model_u_from_actual_theta(double theta_actual_rad) const;
  double rate_limit_final_angle(double requested_theta_rad,
                                double rate_limit_rad_s) const;
  bool contest_uses_camera_feedback_only() const;
  double chassis_acceleration_for_control(double now_s) const;
  bool hold_prediction_crosses_soft_boundary(const ObserverState& estimate,
                                             double actuator_u_m_s2,
                                             double chassis_acceleration_m_s2,
                                             double horizon_s) const;
  void latch_safety(const std::string& reason, bool fault);

  ControlConfig config_;
  DelayedKalmanObserver observer_;
  ChassisSynchronizer chassis_sync_;
  TimestampUnwrapper vision_clock_;
  RemoteClockSynchronizer dmmc_clock_;
  RemoteClockSynchronizer chassis_clock_;
  BallPid pid_;
  TaskManager task_manager_;
  Task3FrictionCompensator friction_compensator_;
  TubeStatus tube_status_;
  bool have_tube_status_ = false;
  VisionMeasurement latest_vision_;
  double latest_raw_vision_position_m_ = 0.0;
  std::int64_t latest_vision_frame_delta_ = 0;
  bool latest_vision_measurement_accepted_ = false;
  std::string latest_vision_update_reason_ = "no_vision_packet";
  std::uint64_t vision_received_frames_ = 0;
  std::uint64_t vision_accepted_frames_ = 0;
  std::uint64_t vision_status_lost_frames_ = 0;
  std::uint64_t vision_duplicate_frames_ = 0;
  std::uint64_t vision_missing_frames_ = 0;
  std::uint64_t dmmc_status_frames_ = 0;
  std::uint64_t dmmc_duplicate_frames_ = 0;
  std::uint64_t dmmc_missing_frames_ = 0;
  std::uint64_t chassis_status_frames_ = 0;
  std::uint64_t chassis_duplicate_frames_ = 0;
  std::uint64_t chassis_missing_frames_ = 0;
  double last_tick_s_ = 0.0;
  double last_vision_receive_s_ = -1.0;
  double last_accepted_vision_s_ = -1.0;
  std::uint32_t last_vision_frame_id_ = 0;
  std::uint32_t last_tube_sequence_ = 0;
  std::uint32_t last_chassis_sequence_ = 0;
  bool have_vision_frame_ = false;
  bool have_chassis_sequence_ = false;
  bool v2_clock_initialized_ = false;
  double v2_capture_time_s_ = 0.0;
  double latest_vision_capture_age_ms_ =
      std::numeric_limits<double>::infinity();
  double filtered_visual_position_m_ = 0.0;
  double visual_position_filter_time_s_ = 0.0;
  bool visual_position_filter_initialized_ = false;
  std::uint32_t command_id_ = 0;
  double previous_applied_u_ = 0.0;
  double previous_theta_command_rad_ = 0.0;
  double previous_model_compensation_rad_ = 0.0;
  bool previous_model_compensation_active_ = false;
  int task3_early_brake_stage_ = -1;
  bool task3_early_braking_active_ = false;
  bool task3_early_braking_done_ = false;
  bool task3_reverse_balance_active_ = false;
  bool task3_reverse_balance_done_ = false;
  double inner_angle_warning_since_s_ = -1.0;
  bool safety_latched_ = false;
  bool safety_fault_latched_ = false;
  bool clear_comm_warning_pending_ = false;
  std::uint64_t safety_event_id_ = 0;
  std::string last_stop_reason_;
};

}  // namespace nx_control
