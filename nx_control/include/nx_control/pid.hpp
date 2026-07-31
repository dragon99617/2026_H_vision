#pragma once

#include "nx_control/types.hpp"

namespace nx_control {

class BallPid {
 public:
  explicit BallPid(const ControlConfig& config) : config_(config) {}

  void reset();
  PidResult calculate(const ObserverState& estimate,
                      const ReferencePoint& reference,
                      double chassis_acceleration_m_s2,
                      bool allow_integrator);
  PidResult track(double applied_m_s2, bool force_freeze);

  double integral_output_m_s2() const { return integral_output_m_s2_; }
  const PidResult& last_result() const { return last_result_; }

 private:
  double clamp_integral(double value, bool* limited) const;

  ControlConfig config_;
  double integral_output_m_s2_ = 0.0;
  bool normal_integration_allowed_ = false;
  PidResult last_result_;
};

}  // namespace nx_control
