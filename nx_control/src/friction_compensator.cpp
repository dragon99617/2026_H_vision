#include "nx_control/friction_compensator.hpp"

#include <algorithm>
#include <cmath>

namespace nx_control {

Task3FrictionCompensator::Task3FrictionCompensator(const ControlConfig& config)
    : config_(config) {}

void Task3FrictionCompensator::reset() {
  mode_ = FrictionMode::Hold;
  direction_ = 0;
  theta_friction_rad_ = 0.0;
  last_update_s_ = -1.0;
}

int Task3FrictionCompensator::requested_direction(
    double position_error_m, double requested_u_m_s2,
    double reference_velocity_m_s) const {
  if (std::abs(requested_u_m_s2) >=
      config_.task3_friction_request_acceleration_m_s2) {
    return requested_u_m_s2 > 0.0 ? 1 : -1;
  }
  if (std::abs(reference_velocity_m_s) >= 1e-4) {
    return reference_velocity_m_s > 0.0 ? 1 : -1;
  }
  if (std::abs(position_error_m) >=
      config_.task3_friction_disable_position_error_m) {
    return position_error_m > 0.0 ? 1 : -1;
  }
  return 0;
}

FrictionCompensation Task3FrictionCompensator::update(
    double now_s, bool active, double position_error_m, double velocity_m_s,
    double requested_u_m_s2, double reference_velocity_m_s) {
  if (!active) {
    reset();
    return {};
  }

  const double dt =
      last_update_s_ >= 0.0
          ? std::clamp(now_s - last_update_s_, 0.0, 5.0 * config_.period_s)
          : config_.period_s;
  last_update_s_ = now_s;
  const bool target_deadband =
      std::abs(position_error_m) <
          config_.task3_friction_disable_position_error_m &&
      std::abs(velocity_m_s) < config_.task3_friction_disable_velocity_m_s;

  double target_friction_rad = 0.0;
  if (target_deadband) {
    mode_ = FrictionMode::Hold;
    direction_ = 0;
  } else {
    const int request =
        requested_direction(position_error_m, requested_u_m_s2,
                            reference_velocity_m_s);
    if (request != 0) direction_ = request;

    bool rolling = mode_ == FrictionMode::RollingPositive ||
                   mode_ == FrictionMode::RollingNegative;
    if (std::abs(velocity_m_s) >=
        config_.task3_friction_rolling_enter_velocity_m_s) {
      rolling = true;
    } else if (std::abs(velocity_m_s) <=
               config_.task3_friction_stationary_enter_velocity_m_s) {
      rolling = false;
    }

    if (direction_ == 0) {
      mode_ = FrictionMode::Hold;
    } else if (rolling) {
      mode_ = direction_ > 0 ? FrictionMode::RollingPositive
                             : FrictionMode::RollingNegative;
      target_friction_rad =
          direction_ * config_.task3_rolling_compensation_rad;
    } else {
      mode_ = direction_ > 0 ? FrictionMode::BreakawayPositive
                             : FrictionMode::BreakawayNegative;
      target_friction_rad =
          direction_ *
          (config_.task3_theta_static_rad + config_.task3_theta_margin_rad);
    }
  }

  const double blend_time =
      std::max(config_.period_s, config_.task3_friction_blend_time_s);
  const double alpha = 1.0 - std::exp(-dt / blend_time);
  theta_friction_rad_ +=
      alpha * (target_friction_rad - theta_friction_rad_);
  if (std::abs(theta_friction_rad_) < 1e-12) theta_friction_rad_ = 0.0;
  return FrictionCompensation{theta_friction_rad_, mode_, direction_,
                              target_deadband};
}

const char* friction_mode_name(FrictionMode mode) {
  switch (mode) {
    case FrictionMode::Hold:
      return "HOLD";
    case FrictionMode::BreakawayPositive:
      return "BREAKAWAY_POSITIVE";
    case FrictionMode::BreakawayNegative:
      return "BREAKAWAY_NEGATIVE";
    case FrictionMode::RollingPositive:
      return "ROLLING_POSITIVE";
    case FrictionMode::RollingNegative:
      return "ROLLING_NEGATIVE";
  }
  return "HOLD";
}

}  // namespace nx_control
