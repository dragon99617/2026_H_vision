#include "nx_control/pid.hpp"

#include <algorithm>
#include <cmath>

namespace nx_control {

void BallPid::reset() {
  integral_output_m_s2_ = 0.0;
  normal_integration_allowed_ = false;
  last_result_ = PidResult{};
}

double BallPid::clamp_integral(double value, bool* limited) const {
  const double limit = config_.pid_integral_output_limit_m_s2;
  const double clamped = std::clamp(value, -limit, limit);
  if (limited != nullptr) *limited = clamped != value;
  return clamped;
}

PidResult BallPid::calculate(const ObserverState& estimate,
                             const ReferencePoint& reference,
                             double chassis_acceleration_m_s2,
                             bool allow_integrator,
                             double correction_scale,
                             double feedforward_scale) {
  PidResult result;
  result.position_error_m = reference.position_m - estimate.position_m;
  result.velocity_error_m_s = reference.velocity_m_s - estimate.velocity_m_s;
  normal_integration_allowed_ =
      allow_integrator &&
      std::abs(result.position_error_m) <= config_.pid_integral_enable_error_m;
  result.integrator_frozen = !normal_integration_allowed_;

  if (normal_integration_allowed_) {
    bool limited = false;
    integral_output_m_s2_ = clamp_integral(
        integral_output_m_s2_ +
            config_.pid_ki_s3 * result.position_error_m * config_.period_s,
        &limited);
    result.integral_limited = limited;
  }

  result.proportional_m_s2 =
      correction_scale * config_.pid_kp_s2 * result.position_error_m;
  result.integral_m_s2 = correction_scale * integral_output_m_s2_;
  result.derivative_m_s2 =
      correction_scale * config_.pid_kd_s_inv * result.velocity_error_m_s;
  result.feedforward_m_s2 =
      feedforward_scale * chassis_acceleration_m_s2 +
      reference.acceleration_m_s2 / config_.rolling_lambda;
  result.disturbance_m_s2 =
      -correction_scale * config_.pid_disturbance_gain * estimate.disturbance_m_s2 /
      config_.rolling_lambda;
  result.unsaturated_m_s2 =
      result.feedforward_m_s2 + result.proportional_m_s2 +
      result.integral_m_s2 + result.derivative_m_s2 +
      result.disturbance_m_s2;
  result.applied_m_s2 = result.unsaturated_m_s2;
  last_result_ = result;
  return last_result_;
}

PidResult BallPid::track(double applied_m_s2, bool force_freeze,
                         bool allow_back_calculation) {
  last_result_.applied_m_s2 = applied_m_s2;
  last_result_.saturated =
      std::abs(applied_m_s2 - last_result_.unsaturated_m_s2) > 1e-9;
  last_result_.integrator_frozen =
      last_result_.integrator_frozen || force_freeze;

  if ((last_result_.saturated || force_freeze) && allow_back_calculation) {
    bool limited = false;
    integral_output_m_s2_ = clamp_integral(
        integral_output_m_s2_ + config_.pid_anti_windup_gain_s_inv *
                                        (applied_m_s2 -
                                         last_result_.unsaturated_m_s2) *
                                        config_.period_s,
        &limited);
    last_result_.integral_limited = last_result_.integral_limited || limited;
  }
  return last_result_;
}

}  // namespace nx_control
