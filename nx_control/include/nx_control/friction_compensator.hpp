#pragma once

#include "nx_control/types.hpp"

namespace nx_control {

struct FrictionCompensation {
  double theta_friction_rad = 0.0;
  FrictionMode mode = FrictionMode::Hold;
  int direction = 0;
  bool target_deadband = false;
};

class Task3FrictionCompensator {
 public:
  explicit Task3FrictionCompensator(const ControlConfig& config);

  FrictionCompensation update(double now_s, bool active, double position_error_m,
                              double velocity_m_s, double requested_u_m_s2,
                              double reference_velocity_m_s);
 void reset();

 private:
  int requested_direction(double position_error_m, double velocity_m_s,
                          double requested_u_m_s2,
                          double reference_velocity_m_s) const;

  ControlConfig config_;
  FrictionMode mode_ = FrictionMode::Hold;
  int direction_ = 0;
  double theta_friction_rad_ = 0.0;
  double last_update_s_ = -1.0;
  double breakaway_started_s_ = -1.0;
  int breakaway_direction_ = 0;
  bool breakaway_limited_ = false;
};

const char* friction_mode_name(FrictionMode mode);

}  // namespace nx_control
