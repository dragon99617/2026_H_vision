#include "nx_control/observer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nx_control {

DelayedKalmanObserver::DelayedKalmanObserver(const ControlConfig& config) : config_(config) {
  reset(0.0);
}

void DelayedKalmanObserver::reset(double time_s, double position_m) {
  nodes_.clear();
  Node node;
  node.time_s = time_s;
  node.state << position_m, 0.0, 0.0;
  node.covariance.setZero();
  node.covariance.diagonal() << 0.01 * 0.01, 0.10 * 0.10, 0.40 * 0.40;
  nodes_.push_back(node);
  accepted_measurements_ = rejected_measurements_ = too_old_measurements_ = 0;
}

void DelayedKalmanObserver::propagate(Eigen::Vector3d& state, Eigen::Matrix3d& covariance,
                                      double dt, double actuator_u_m_s2,
                                      double chassis_acceleration_m_s2) const {
  if (dt <= 0.0) return;
  const double half_dt2 = 0.5 * dt * dt;
  Eigen::Matrix3d transition;
  transition << 1.0, dt, half_dt2, 0.0, 1.0, dt, 0.0, 0.0, 1.0;
  const double known_acceleration =
      config_.rolling_lambda * (actuator_u_m_s2 - chassis_acceleration_m_s2);
  state = transition * state + Eigen::Vector3d(half_dt2 * known_acceleration,
                                                dt * known_acceleration, 0.0);

  const double accel_variance = config_.process_accel_sigma_m_s2 *
                                config_.process_accel_sigma_m_s2;
  const Eigen::Vector3d accel_gain(half_dt2, dt, 0.0);
  Eigen::Matrix3d process = accel_variance * accel_gain * accel_gain.transpose();
  process(2, 2) += config_.process_disturbance_sigma_m_s3 *
                   config_.process_disturbance_sigma_m_s3 * dt;
  covariance = transition * covariance * transition.transpose() + process;
  covariance = 0.5 * (covariance + covariance.transpose());
}

void DelayedKalmanObserver::predict(double target_time_s, double actuator_u_m_s2,
                                    double chassis_acceleration_m_s2) {
  if (nodes_.empty()) reset(target_time_s);
  if (target_time_s <= nodes_.back().time_s) return;
  Node next = nodes_.back();
  const double dt = target_time_s - next.time_s;
  propagate(next.state, next.covariance, dt, actuator_u_m_s2, chassis_acceleration_m_s2);
  next.time_s = target_time_s;
  next.actuator_u_m_s2 = actuator_u_m_s2;
  next.chassis_acceleration_m_s2 = chassis_acceleration_m_s2;
  nodes_.push_back(next);
  trim_history();
}

bool DelayedKalmanObserver::measurement_update(Eigen::Vector3d& state,
                                               Eigen::Matrix3d& covariance,
                                               double position_m,
                                               double measurement_variance) {
  const double innovation = position_m - state(0);
  const double innovation_variance = covariance(0, 0) + measurement_variance;
  if (!(innovation_variance > 0.0) ||
      std::abs(innovation) > config_.innovation_gate_sigma * std::sqrt(innovation_variance)) {
    return false;
  }
  const Eigen::Vector3d gain = covariance.col(0) / innovation_variance;
  state += gain * innovation;
  const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
  Eigen::RowVector3d measurement;
  measurement << 1.0, 0.0, 0.0;
  const Eigen::Matrix3d kh = gain * measurement;
  covariance = (identity - kh) * covariance * (identity - kh).transpose() +
               gain * measurement_variance * gain.transpose();
  covariance = 0.5 * (covariance + covariance.transpose());
  return true;
}

bool DelayedKalmanObserver::update_position(double capture_time_s, double position_m,
                                            double ball_confidence,
                                            double tube_confidence,
                                            VisionStatus status) {
  if (status == VisionStatus::Lost || nodes_.empty()) return false;
  if ((status == VisionStatus::Predicted &&
       (ball_confidence < config_.predicted_min_confidence ||
        tube_confidence < config_.predicted_min_confidence)) ||
      ball_confidence <= 0.0 || tube_confidence <= 0.0) {
    ++rejected_measurements_;
    return false;
  }
  if (capture_time_s < nodes_.front().time_s - 1e-6) {
    ++too_old_measurements_;
    return false;
  }
  capture_time_s = std::clamp(capture_time_s, nodes_.front().time_s, nodes_.back().time_s);
  const double confidence = std::clamp(ball_confidence * tube_confidence, 0.05, 1.0);
  double sigma = config_.measurement_sigma_m / std::sqrt(confidence);
  if (status == VisionStatus::Predicted) sigma *= 2.5;
  const double variance = sigma * sigma;

  std::size_t lower = 0;
  while (lower + 1 < nodes_.size() && nodes_[lower + 1].time_s <= capture_time_s) ++lower;
  Eigen::Vector3d state = nodes_[lower].state;
  Eigen::Matrix3d covariance = nodes_[lower].covariance;
  std::size_t replay = lower + 1;
  const bool exact_node = std::abs(capture_time_s - nodes_[lower].time_s) <= 1e-9;
  if (!exact_node) {
    const Node interval = nodes_[lower + 1];
    propagate(state, covariance, capture_time_s - nodes_[lower].time_s,
              interval.actuator_u_m_s2, interval.chassis_acceleration_m_s2);
  }
  if (!measurement_update(state, covariance, position_m, variance)) {
    ++rejected_measurements_;
    return false;
  }

  if (exact_node) {
    nodes_[lower].state = state;
    nodes_[lower].covariance = covariance;
  } else {
    const Node& interval = nodes_[lower + 1];
    Node inserted;
    inserted.time_s = capture_time_s;
    inserted.state = state;
    inserted.covariance = covariance;
    inserted.actuator_u_m_s2 = interval.actuator_u_m_s2;
    inserted.chassis_acceleration_m_s2 = interval.chassis_acceleration_m_s2;
    nodes_.insert(nodes_.begin() + static_cast<std::ptrdiff_t>(lower + 1), inserted);
    replay = lower + 2;
  }
  double replay_time = capture_time_s;
  for (; replay < nodes_.size(); ++replay) {
    propagate(state, covariance, nodes_[replay].time_s - replay_time,
              nodes_[replay].actuator_u_m_s2, nodes_[replay].chassis_acceleration_m_s2);
    replay_time = nodes_[replay].time_s;
    nodes_[replay].state = state;
    nodes_[replay].covariance = covariance;
  }
  ++accepted_measurements_;
  return true;
}

ObserverState DelayedKalmanObserver::state() const {
  const Eigen::Vector3d value = nodes_.empty() ? Eigen::Vector3d::Zero() : nodes_.back().state;
  return ObserverState{value(0), value(1), value(2)};
}

Eigen::Matrix3d DelayedKalmanObserver::covariance() const {
  return nodes_.empty() ? Eigen::Matrix3d::Identity() : nodes_.back().covariance;
}

void DelayedKalmanObserver::trim_history() {
  if (nodes_.empty()) return;
  const double oldest = nodes_.back().time_s - config_.observer_history_s;
  while (nodes_.size() > 2 && nodes_[1].time_s < oldest) nodes_.pop_front();
}

void ChassisSynchronizer::reset() {
  latest_ = ChassisState{};
  initialized_ = false;
  filtered_acceleration_ = 0.0;
  filter_time_s_ = 0.0;
}

void ChassisSynchronizer::ingest(const ChassisState& state) {
  const double sample_time = state.sample_time_s > 0.0 ? state.sample_time_s : state.receive_time_s;
  const double dt = initialized_ ? std::max(0.0, sample_time - filter_time_s_) : 0.0;
  if (!initialized_ || dt > 0.5) {
    filtered_acceleration_ = state.acceleration_actual_m_s2;
  } else if (dt > 0.0) {
    const double alpha = 1.0 - std::exp(-dt / std::max(1e-4, config_.chassis_filter_tau_s));
    filtered_acceleration_ += alpha * (state.acceleration_actual_m_s2 - filtered_acceleration_);
  }
  latest_ = state;
  filter_time_s_ = sample_time;
  initialized_ = true;
}

bool ChassisSynchronizer::valid(double now_s) const {
  if (!initialized_ || latest_.faults != 0U) return false;
  const double allowed = latest_.ttl_ms > 0
                             ? std::min(config_.chassis_stale_s,
                                        static_cast<double>(latest_.ttl_ms) / 1000.0)
                             : config_.chassis_stale_s;
  return age_s(now_s) <= allowed;
}

double ChassisSynchronizer::age_s(double now_s) const {
  return initialized_ ? std::max(0.0, now_s - latest_.receive_time_s)
                      : std::numeric_limits<double>::infinity();
}

std::vector<double> ChassisSynchronizer::acceleration_reference_forecast(
    double now_s, int horizon, double period_s) const {
  std::vector<double> result(static_cast<std::size_t>(std::max(0, horizon)), 0.0);
  if (!valid(now_s)) return result;
  for (int index = 0; index < horizon; ++index) {
    const double lookahead = (index + 1) * period_s;
    double acceleration = latest_.acceleration_ref_m_s2 + latest_.jerk_ref_m_s3 * lookahead;
    if (latest_.motion_phase == MotionPhase::Cruise || latest_.motion_phase == MotionPhase::Curve ||
        latest_.motion_phase == MotionPhase::Stop) {
      acceleration = latest_.acceleration_ref_m_s2;
    }
    result[static_cast<std::size_t>(index)] = std::clamp(acceleration, -0.55, 0.55);
  }
  return result;
}

double ChassisSynchronizer::delay_compensated_actual_acceleration(double now_s) const {
  if (!valid(now_s)) return 0.0;
  const bool steady_phase = latest_.motion_phase == MotionPhase::Stop ||
                            latest_.motion_phase == MotionPhase::Cruise ||
                            latest_.motion_phase == MotionPhase::Curve;
  const double projection_s = steady_phase
                                  ? 0.0
                                  : std::clamp(age_s(now_s) + config_.chassis_filter_tau_s, 0.0,
                                               config_.chassis_stale_s);
  return std::clamp(filtered_acceleration_ + latest_.jerk_ref_m_s3 * projection_s, -0.55, 0.55);
}

double TimestampUnwrapper::unwrap_seconds(std::uint32_t remote_ms, double local_now_s) {
  const std::int64_t local_ms = static_cast<std::int64_t>(std::llround(local_now_s * 1000.0));
  if (!initialized_) {
    const std::int64_t epoch = local_ms & ~static_cast<std::int64_t>(0xFFFFFFFFULL);
    std::int64_t candidate = epoch | static_cast<std::int64_t>(remote_ms);
    const std::int64_t wrap = static_cast<std::int64_t>(1ULL << 32U);
    if (candidate - local_ms > wrap / 2) candidate -= wrap;
    if (local_ms - candidate > wrap / 2) candidate += wrap;
    last_extended_ms_ = candidate;
    initialized_ = true;
  } else {
    const std::int64_t wrap = static_cast<std::int64_t>(1ULL << 32U);
    const std::int64_t base = last_extended_ms_ & ~static_cast<std::int64_t>(0xFFFFFFFFULL);
    std::int64_t candidate = base | static_cast<std::int64_t>(remote_ms);
    if (candidate - last_extended_ms_ > wrap / 2) candidate -= wrap;
    if (last_extended_ms_ - candidate > wrap / 2) candidate += wrap;
    last_extended_ms_ = candidate;
  }
  return static_cast<double>(last_extended_ms_) / 1000.0;
}

double RemoteClockSynchronizer::to_local_seconds(std::uint32_t remote_ms,
                                                 double receive_time_s) {
  const double remote_s = unwrapper_.unwrap_seconds(remote_ms, receive_time_s);
  const double observed_offset = receive_time_s - remote_s;
  if (!initialized_) {
    offset_s_ = observed_offset;
    initialized_ = true;
  } else if (observed_offset < offset_s_) {
    offset_s_ = 0.90 * offset_s_ + 0.10 * observed_offset;
  } else {
    // Positive excursions are normally USB/forwarding latency; follow only clock drift.
    offset_s_ = 0.999 * offset_s_ + 0.001 * observed_offset;
  }
  return std::clamp(remote_s + offset_s_, receive_time_s - 0.5, receive_time_s);
}

}  // namespace nx_control
