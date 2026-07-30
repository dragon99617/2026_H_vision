#pragma once

#include "nx_control/mpc.hpp"
#include "nx_control/observer.hpp"
#include "nx_control/task_manager.hpp"
#include "nx_control/types.hpp"

#include <cstdint>
#include <memory>

namespace nx_control {

class NxController {
 public:
  explicit NxController(const ControlConfig& config,
                        std::unique_ptr<QpSolver> solver = nullptr);

  void configure_task(TaskMode mode, double target_m, bool start_immediately,
                      bool start_on_chassis_event = true);
  void start_task(double now_s);
  void ingest_vision(VisionMeasurement measurement);
  void ingest_tube_status(const TubeStatus& status);
  void ingest_chassis_state(const ChassisState& state);
  ControlOutput tick(double now_s);
  void reset(double now_s);

  const DelayedKalmanObserver& observer() const { return observer_; }
  const TaskManager& task_manager() const { return task_manager_; }
  const char* mpc_backend_name() const { return mpc_.backend_name(); }

 private:
  double rate_limit_and_clamp(double requested_u);
  double fallback_command(const ObserverState& estimate, const ReferencePoint& reference,
                          double feedforward) const;

  ControlConfig config_;
  DelayedKalmanObserver observer_;
  ChassisSynchronizer chassis_sync_;
  TimestampUnwrapper vision_clock_;
  RemoteClockSynchronizer dmmc_clock_;
  RemoteClockSynchronizer chassis_clock_;
  BallMpc mpc_;
  TaskManager task_manager_;
  TubeStatus tube_status_;
  bool have_tube_status_ = false;
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
  double latest_visual_position_m_ = 0.0;
  bool latest_visual_position_valid_ = false;
  std::uint32_t command_id_ = 0;
  double previous_command_u_ = 0.0;
  int solver_failures_ = 0;
  double first_solver_failure_s_ = -1.0;
};

}  // namespace nx_control
