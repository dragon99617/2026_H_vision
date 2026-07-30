#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace nx_control {

constexpr double kGravity = 9.80665;
constexpr double kRollingLambda = 5.0 / 7.0;

enum class VisionStatus : std::uint8_t { Lost = 0, Measured = 1, Predicted = 2 };
enum class MotionPhase : std::uint8_t {
  Stop = 0,
  Accel = 1,
  Cruise = 2,
  Curve = 3,
  Decel = 4,
};
enum class TrackSegment : std::uint8_t { Unknown = 0, AB = 1, BC = 2, CD = 3, DA = 4 };
enum class TaskState : std::uint8_t {
  Idle = 0,
  StaticMove = 1,
  HoldCenter = 2,
  HoldTarget = 3,
  VehicleAccel = 4,
  VehicleCruise = 5,
  VehicleDecel = 6,
  Safe = 7,
  Fault = 8,
  StandbyHold = 9,  // Command-only state: hold the beam angle while a task is armed.
};
enum class TaskMode : std::uint8_t {
  Idle = 0,
  StaticSequence = 1,
  HoldCenter = 2,
  HoldTarget = 3,
  AutoVehicle = 4,
  Contest3 = 5,   // H problem requirement 3: O -> +5 cm -> -5 cm.
  Contest45 = 6,  // H problem requirements 4/5: hold O while driving.
  Contest6 = 7,   // H problem requirement 6: hold a selected point while driving.
};
enum class FrictionMode : std::uint8_t {
  Hold = 0,
  BreakawayPositive = 1,
  BreakawayNegative = 2,
  RollingPositive = 3,
  RollingNegative = 4,
};

struct VisionMeasurement {
  std::uint32_t frame_id = 0;
  VisionStatus status = VisionStatus::Lost;
  double position_m = 0.0;
  double ball_confidence = 0.0;
  double tube_confidence = 0.0;
  std::uint32_t capture_time_ms = 0;
  bool has_capture_time = false;
  double capture_time_s = 0.0;
  double receive_time_s = 0.0;
};

struct TubeStatus {
  std::uint32_t sequence = 0;
  std::uint32_t dmmc_time_ms = 0;
  double theta_target_rad = 0.0;
  double theta_reference_rad = 0.0;
  double theta_actual_rad = 0.0;
  double motor_position_rad = 0.0;
  double motor_velocity_rad_s = 0.0;
  double motor_torque_nm = 0.0;
  std::uint32_t faults = 0;
  std::uint16_t can_age_ms = 0;
  std::uint16_t usb_crc_errors = 0;
  std::uint16_t control_age_ms = 0;
  std::uint8_t state = 0;
  std::uint8_t flags = 0;
  double sample_time_s = 0.0;
  double receive_time_s = 0.0;
};

struct ChassisState {
  std::uint32_t sequence = 0;
  std::uint32_t chassis_time_ms = 0;
  double velocity_ref_m_s = 0.0;
  double acceleration_ref_m_s2 = 0.0;
  double jerk_ref_m_s3 = 0.0;
  double velocity_actual_m_s = 0.0;
  double acceleration_actual_m_s2 = 0.0;
  double track_error_m = 0.0;
  double track_quality = 0.0;
  MotionPhase motion_phase = MotionPhase::Stop;
  TrackSegment track_segment = TrackSegment::Unknown;
  std::uint16_t events = 0;
  std::uint16_t faults = 0;
  std::uint16_t ttl_ms = 0;
  double yaw_rate_rad_s = 0.0;
  double sample_time_s = 0.0;
  double receive_time_s = 0.0;
};

struct ControlCommand {
  std::uint32_t command_id = 0;
  std::uint32_t source_frame_id = 0;
  std::uint32_t nx_time_ms = 0;
  double theta_cmd_rad = 0.0;
  double theta_rate_limit_rad_s = 0.0;
  std::uint16_t ttl_ms = 0;
  TaskState control_state = TaskState::Idle;
  std::uint8_t flags = 0;
};

// Wire-level tube-control-v3 fields.  This is intentionally separate from
// ControlCommand: the production controller maps its TaskState to an MC02
// state, while small communication tools sometimes need an exact MC02 state.
struct TubeControlV3Command {
  std::uint32_t command_id = 0;
  std::uint32_t source_frame_id = 0;
  std::uint32_t nx_time_ms = 0;
  std::int16_t theta_cmd_cdeg = 0;
  std::uint16_t theta_rate_limit_cdeg_s = 0;
  std::uint16_t ttl_ms = 0;
  std::uint8_t control_state = 0;
  std::uint8_t flags = 0;
  std::uint16_t reserved = 0;
};

struct ObserverState {
  double position_m = 0.0;
  double velocity_m_s = 0.0;
  double disturbance_m_s2 = 0.0;
};

struct ReferencePoint {
  double position_m = 0.0;
  double velocity_m_s = 0.0;
  double acceleration_m_s2 = 0.0;
};

struct MpcResult {
  bool solved = false;
  bool timed_out = false;
  int iterations = 0;
  double solve_time_ms = 0.0;
  double qp_build_time_ms = 0.0;
  double qp_setup_time_ms = 0.0;
  double qp_update_time_ms = 0.0;
  double qp_backend_solve_time_ms = 0.0;
  double command_m_s2 = 0.0;
  double max_slack_m = 0.0;
  std::vector<double> command_sequence;
  std::vector<double> predicted_position;
  std::string backend;
};

struct ControlOutput {
  ControlCommand command;
  ObserverState estimate;
  ReferencePoint reference;
  int task3_stage = -1;
  bool settle_position_ok = false;
  bool settle_velocity_ok = false;
  bool settle_theta_ok = false;
  double settle_elapsed_ms = 0.0;
  double u_command_m_s2 = 0.0;
  double theta_mpc_rad = 0.0;
  double theta_bias_rad = 0.0;
  double theta_friction_rad = 0.0;
  FrictionMode friction_mode = FrictionMode::Hold;
  int friction_direction = 0;
  double acceleration_used_m_s2 = 0.0;
  double vision_age_ms = std::numeric_limits<double>::infinity();
  double vision_capture_age_ms = std::numeric_limits<double>::infinity();
  double chassis_age_ms = std::numeric_limits<double>::infinity();
  double dmmc_age_ms = std::numeric_limits<double>::infinity();
  double mpc_solve_ms = 0.0;
  double qp_build_ms = 0.0;
  double qp_setup_ms = 0.0;
  double qp_update_ms = 0.0;
  double qp_backend_solve_ms = 0.0;
  double qp_iteration_us = 0.0;
  double max_predicted_slack_m = 0.0;
  int solver_iterations = 0;
  int solver_failures = 0;
  bool used_fallback = false;
  bool request_slowdown = false;
  bool request_stop = false;
  bool safety_latched = false;
  std::uint64_t safety_event_id = 0;
  std::vector<double> predicted_position_m;
  std::string reason;
  std::string last_stop_reason;
};

struct ControlConfig {
  double period_s = 0.020;
  double actuator_tau_s = 0.045;
  double actuator_delay_s = 0.0;
  double rolling_lambda = kRollingLambda;
  int horizon = 40;
  double theta_limit_rad = 4.0 * 3.14159265358979323846 / 180.0;
  double theta_rate_limit_rad_s = 2.0 * 3.14159265358979323846 / 180.0;
  double position_soft_limit_m = 0.105;
  double position_safe_limit_m = 0.115;
  double position_scale_m = 0.010;
  double velocity_scale_m_s = 0.010;
  double input_scale_m_s2 = 0.100;
  double delta_input_scale_m_s2 = 0.015;
  double hold_enter_position_error_m = 0.004;
  double hold_enter_velocity_m_s = 0.015;
  double hold_exit_position_error_m = 0.008;
  double task3_reference_max_velocity_m_s = 0.030;
  double task3_reference_max_acceleration_m_s2 = 0.060;
  double task3_reference_max_jerk_m_s3 = 0.300;
  double task3_settle_position_error_m = 0.004;
  double task3_settle_velocity_m_s = 0.005;
  double task3_settle_dwell_s = 0.500;
  double task3_settle_theta_tolerance_rad =
      0.2 * 3.14159265358979323846 / 180.0;
  double task3_theta_bias_rad = -0.15 * 3.14159265358979323846 / 180.0;
  double task3_theta_static_rad = 0.55 * 3.14159265358979323846 / 180.0;
  double task3_theta_margin_rad = 0.05 * 3.14159265358979323846 / 180.0;
  double task3_rolling_compensation_rad =
      0.30 * 3.14159265358979323846 / 180.0;
  double task3_friction_blend_time_s = 0.20;
  double task3_friction_breakaway_timeout_s = 0.75;
  double task3_braking_deceleration_m_s2 = 0.080;
  double task3_friction_rolling_enter_velocity_m_s = 0.010;
  double task3_friction_stationary_enter_velocity_m_s = 0.005;
  double task3_friction_disable_position_error_m = 0.0025;
  double task3_friction_disable_velocity_m_s = 0.005;
  double task3_friction_request_acceleration_m_s2 = 0.005;
  double slack_weight = 5000.0;
  double measurement_sigma_m = 0.003;
  double predicted_min_confidence = 0.40;
  double process_accel_sigma_m_s2 = 0.20;
  double process_disturbance_sigma_m_s3 = 0.08;
  double innovation_gate_sigma = 3.0;
  double observer_history_s = 0.250;
  double vision_v2_latency_s = 0.045;
  double vision_frame_rate_hz = 60.0;
  double vision_decay_start_s = 0.040;
  double vision_loss_hold_s = 0.100;
  double vision_loss_safe_s = 0.250;
  double chassis_filter_tau_s = 0.040;
  double chassis_stale_s = 0.100;
  double dmmc_stale_s = 0.050;
  double solver_warning_ms = 3.0;
  double solver_deadline_ms = 10.0;
  int fallback_after_failures = 3;
  int stop_after_failures = 10;
  double stop_after_failure_s = 0.200;
  double fallback_kp = 14.0;
  double fallback_kd = 4.0;
  double fallback_disturbance_gain = 0.5;
  bool require_osqp = false;
  int qp_max_iterations = 250;
  double qp_eps_abs = 1.25e-2;
  double qp_eps_rel = 1.25e-2;
  double qp_rho = 0.1;
  int qp_adaptive_rho_interval = 10;
  int qp_check_termination_interval = 10;
  bool qp_scaled_termination = true;
};

}  // namespace nx_control
