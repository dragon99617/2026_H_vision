#pragma once

#include "nx_control/types.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <deque>
#include <string>

namespace nx_control {

class DelayedKalmanObserver {
 public:
  explicit DelayedKalmanObserver(const ControlConfig& config);

  void reset(double time_s, double position_m = 0.0);
  void predict(double target_time_s, double actuator_u_m_s2, double chassis_acceleration_m_s2);
  bool update_position(double capture_time_s, double position_m, double ball_confidence,
                       double tube_confidence, VisionStatus status);

  ObserverState state() const;
  Eigen::Matrix3d covariance() const;
  double time_s() const { return nodes_.empty() ? 0.0 : nodes_.back().time_s; }
  std::uint64_t accepted_measurements() const { return accepted_measurements_; }
  std::uint64_t rejected_measurements() const { return rejected_measurements_; }
  std::uint64_t too_old_measurements() const { return too_old_measurements_; }
  const std::string& last_update_reason() const { return last_update_reason_; }

 private:
  struct Node {
    double time_s = 0.0;
    Eigen::Vector3d state = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
    double actuator_u_m_s2 = 0.0;
    double chassis_acceleration_m_s2 = 0.0;
  };

  void propagate(Eigen::Vector3d& state, Eigen::Matrix3d& covariance, double dt,
                 double actuator_u_m_s2, double chassis_acceleration_m_s2) const;
  bool measurement_update(Eigen::Vector3d& state, Eigen::Matrix3d& covariance,
                          double position_m, double measurement_variance);
  void trim_history();

  ControlConfig config_;
  std::deque<Node> nodes_;
  std::uint64_t accepted_measurements_ = 0;
  std::uint64_t rejected_measurements_ = 0;
  std::uint64_t too_old_measurements_ = 0;
  std::string last_update_reason_ = "no_measurement";
};

class ChassisSynchronizer {
 public:
  explicit ChassisSynchronizer(const ControlConfig& config) : config_(config) {}

  void reset();
  void ingest(const ChassisState& state);
  bool valid(double now_s) const;
  double age_s(double now_s) const;
  double filtered_actual_acceleration() const { return filtered_acceleration_; }
  double delay_compensated_actual_acceleration(double now_s) const;
  const ChassisState& latest() const { return latest_; }

 private:
  ControlConfig config_;
  ChassisState latest_;
  bool initialized_ = false;
  double filtered_acceleration_ = 0.0;
  double filter_time_s_ = 0.0;
};

class TimestampUnwrapper {
 public:
  double unwrap_seconds(std::uint32_t remote_ms, double local_now_s);
  void reset() { initialized_ = false; last_extended_ms_ = 0; }

 private:
  bool initialized_ = false;
  std::int64_t last_extended_ms_ = 0;
};

class RemoteClockSynchronizer {
 public:
  double to_local_seconds(std::uint32_t remote_ms, double receive_time_s);
  void reset() { unwrapper_.reset(); initialized_ = false; offset_s_ = 0.0; }

 private:
  TimestampUnwrapper unwrapper_;
  bool initialized_ = false;
  double offset_s_ = 0.0;
};

}  // namespace nx_control
