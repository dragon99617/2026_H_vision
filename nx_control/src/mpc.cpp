#include "nx_control/mpc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nx_control {
namespace {

double infinity_norm(const Eigen::VectorXd& value) {
  return value.size() == 0 ? 0.0 : value.cwiseAbs().maxCoeff();
}

Eigen::VectorXd clamp_vector(const Eigen::VectorXd& value, const Eigen::VectorXd& lower,
                             const Eigen::VectorXd& upper) {
  Eigen::VectorXd result = value;
  for (Eigen::Index index = 0; index < value.size(); ++index) {
    result(index) = std::min(upper(index), std::max(lower(index), value(index)));
  }
  return result;
}

}  // namespace

DenseAdmmSolver::DenseAdmmSolver(int max_iterations, double eps_abs, double eps_rel)
    : max_iterations_(max_iterations), eps_abs_(eps_abs), eps_rel_(eps_rel) {}

QpResult DenseAdmmSolver::solve(const QpProblem& problem) {
  const Eigen::Index variables = problem.hessian.rows();
  const Eigen::Index constraints = problem.constraint.rows();
  QpResult result;
  if (variables == 0 || problem.hessian.cols() != variables ||
      problem.gradient.size() != variables || problem.constraint.cols() != variables ||
      problem.lower.size() != constraints || problem.upper.size() != constraints) {
    result.status = "invalid_dimensions";
    return result;
  }
  if (x_.size() != variables || z_.size() != constraints || y_.size() != constraints) {
    x_ = Eigen::VectorXd::Zero(variables);
    z_ = clamp_vector(problem.constraint * x_, problem.lower, problem.upper);
    y_ = Eigen::VectorXd::Zero(constraints);
  }
  const Eigen::MatrixXd system = problem.hessian +
                                 rho_ * problem.constraint.transpose() * problem.constraint +
                                 sigma_ * Eigen::MatrixXd::Identity(variables, variables);
  Eigen::LDLT<Eigen::MatrixXd> factor(system);
  if (factor.info() != Eigen::Success) {
    result.status = "factorization_failed";
    return result;
  }

  for (int iteration = 1; iteration <= max_iterations_; ++iteration) {
    const Eigen::VectorXd rhs = sigma_ * x_ - problem.gradient +
                                problem.constraint.transpose() * (rho_ * z_ - y_);
    x_ = factor.solve(rhs);
    if (!x_.allFinite()) {
      result.status = "non_finite";
      return result;
    }
    const Eigen::VectorXd ax = problem.constraint * x_;
    const Eigen::VectorXd previous_z = z_;
    z_ = clamp_vector(ax + y_ / rho_, problem.lower, problem.upper);
    y_ += rho_ * (ax - z_);

    const double primal_residual = infinity_norm(ax - z_);
    const double dual_residual =
        infinity_norm(rho_ * problem.constraint.transpose() * (z_ - previous_z));
    const double primal_tolerance =
        eps_abs_ + eps_rel_ * std::max(infinity_norm(ax), infinity_norm(z_));
    const double dual_tolerance =
        eps_abs_ + eps_rel_ * infinity_norm(problem.constraint.transpose() * y_);
    if (primal_residual <= primal_tolerance && dual_residual <= dual_tolerance) {
      result.solved = true;
      result.iterations = iteration;
      result.primal = x_;
      result.status = "solved";
      return result;
    }
  }
  result.iterations = max_iterations_;
  result.primal = x_;
  result.status = "maximum_iterations";
  return result;
}

std::unique_ptr<QpSolver> make_default_qp_solver(const ControlConfig& config) {
#ifdef NX_CONTROL_HAS_OSQP
  return make_osqp_solver(config);
#else
  if (config.require_osqp) {
    throw std::runtime_error("require_osqp=true but this build has no OSQP backend");
  }
  return std::make_unique<DenseAdmmSolver>(config.qp_max_iterations, config.qp_eps_abs,
                                           config.qp_eps_rel);
#endif
}

BallMpc::BallMpc(const ControlConfig& config, std::unique_ptr<QpSolver> solver)
    : config_(config), solver_(solver ? std::move(solver) : make_default_qp_solver(config)) {
  if (config_.horizon <= 0) throw std::invalid_argument("MPC horizon must be positive");
  terminal_cost_ = terminal_cost();
  warm_up();
}

void BallMpc::warm_up() {
  const auto horizon = static_cast<std::size_t>(config_.horizon);
  const std::vector<ReferencePoint> reference(horizon);
  const std::vector<double> chassis_acceleration_ref(horizon, 0.0);
  std::vector<Eigen::RowVectorXd> position_sensitivity;
  std::vector<double> position_constant;
  QpProblem warmup_problem =
      build_problem({0.0, 0.0, 0.0, 0.0}, 0.0, reference,
                    chassis_acceleration_ref, position_sensitivity,
                    position_constant);
  (void)solver_->solve(warmup_problem);
}

Eigen::Matrix4d BallMpc::terminal_cost() const {
  const double dt = config_.period_s;
  const double alpha = std::exp(-dt / config_.actuator_tau_s);
  Eigen::Matrix4d a = Eigen::Matrix4d::Zero();
  a(0, 0) = 1.0;
  a(0, 1) = dt;
  a(0, 2) = 0.5 * config_.rolling_lambda * dt * dt;
  a(1, 1) = 1.0;
  a(1, 2) = config_.rolling_lambda * dt;
  a(2, 2) = alpha;
  a(2, 3) = 1.0 - alpha;
  a(3, 3) = 1.0;
  Eigen::Vector4d b;
  b << 0.0, 0.0, 1.0 - alpha, 1.0;
  Eigen::Matrix4d q = Eigen::Matrix4d::Zero();
  q.diagonal() << 1.0 / std::pow(config_.position_scale_m, 2),
      1.0 / std::pow(config_.velocity_scale_m_s, 2),
      1.0 / std::pow(config_.input_scale_m_s2, 2),
      1.0 / std::pow(config_.input_scale_m_s2, 2);
  const double r = 1.0 / std::pow(config_.delta_input_scale_m_s2, 2);
  Eigen::Matrix4d p = q;
  for (int iteration = 0; iteration < 500; ++iteration) {
    const double denominator = r + (b.transpose() * p * b)(0, 0);
    const Eigen::Matrix4d next =
        a.transpose() * p * a -
        a.transpose() * p * b * (1.0 / denominator) * b.transpose() * p * a + q;
    if ((next - p).cwiseAbs().maxCoeff() < 1e-9) {
      p = next;
      break;
    }
    p = next;
  }
  return p;
}

QpProblem BallMpc::build_problem(
    const std::array<double, 4>& state, double previous_command_m_s2,
    const std::vector<ReferencePoint>& reference,
    const std::vector<double>& chassis_acceleration_ref,
    std::vector<Eigen::RowVectorXd>& position_sensitivity,
    std::vector<double>& position_constant) {
  const int n = config_.horizon;
  const int variables = 2 * n;
  const int constraints = 5 * n;
  const double dt = config_.period_s;
  const double alpha = std::exp(-dt / config_.actuator_tau_s);
  const int actuator_delay_steps =
      std::max(0, static_cast<int>(std::lround(config_.actuator_delay_s / dt)));
  const double du_scale = config_.delta_input_scale_m_s2;
  const double x_scale = config_.position_scale_m;
  const double v_scale = config_.velocity_scale_m_s;
  const double u_scale = config_.input_scale_m_s2;

  Eigen::Matrix4d a = Eigen::Matrix4d::Identity();
  a(0, 1) = dt;
  a(0, 2) = 0.5 * config_.rolling_lambda * dt * dt;
  a(1, 2) = config_.rolling_lambda * dt;
  a(2, 2) = alpha;
  Eigen::Vector4d bu(0.0, 0.0, 1.0 - alpha, 0.0);
  Eigen::Vector4d ba(-0.5 * config_.rolling_lambda * dt * dt,
                     -config_.rolling_lambda * dt, 0.0, 0.0);
  Eigen::Vector4d bd(0.5 * dt * dt, dt, 0.0, 0.0);

  Eigen::Vector4d constant(state[0], state[1], state[3], previous_command_m_s2);
  Eigen::MatrixXd sensitivity = Eigen::MatrixXd::Zero(4, n);
  Eigen::RowVectorXd command_sensitivity = Eigen::RowVectorXd::Zero(n);
  const bool build_structure = cached_hessian_.size() == 0;
  Eigen::MatrixXd hessian =
      build_structure ? Eigen::MatrixXd::Zero(variables, variables)
                      : cached_hessian_;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(variables);
  position_sensitivity.clear();
  position_sensitivity.reserve(static_cast<std::size_t>(n));
  position_constant.clear();
  position_constant.reserve(static_cast<std::size_t>(n));

  auto add_square = [&](const Eigen::RowVectorXd& row, double error, double scale,
                        double weight = 1.0) {
    const Eigen::RowVectorXd normalized = row / scale;
    if (build_structure) {
      hessian += 2.0 * weight * normalized.transpose() * normalized;
    }
    gradient += 2.0 * weight * (error / scale) * normalized.transpose();
  };

  for (int step = 0; step < n; ++step) {
    command_sensitivity(step) = du_scale;
    Eigen::RowVectorXd applied_command_sensitivity = Eigen::RowVectorXd::Zero(n);
    if (step >= actuator_delay_steps) {
      const int applied_step = step - actuator_delay_steps;
      applied_command_sensitivity.head(applied_step + 1) =
          command_sensitivity.head(applied_step + 1);
    }
    const double acceleration =
        step < static_cast<int>(chassis_acceleration_ref.size())
            ? chassis_acceleration_ref[static_cast<std::size_t>(step)]
            : 0.0;
    constant(3) = previous_command_m_s2;
    constant = a * constant + bu * previous_command_m_s2 + ba * acceleration + bd * state[2];
    sensitivity = a * sensitivity + bu * applied_command_sensitivity;
    sensitivity.row(3) = command_sensitivity;

    const ReferencePoint target = step < static_cast<int>(reference.size())
                                      ? reference[static_cast<std::size_t>(step)]
                                      : ReferencePoint{};
    const double feedforward = acceleration + target.acceleration_m_s2 / config_.rolling_lambda;
    Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(variables);
    row.head(n) = sensitivity.row(0);
    add_square(row, constant(0) - target.position_m, x_scale);
    row.setZero();
    row.head(n) = sensitivity.row(1);
    add_square(row, constant(1) - target.velocity_m_s, v_scale);
    row.setZero();
    row.head(n) = command_sensitivity;
    add_square(row, previous_command_m_s2 - feedforward, u_scale);
    row.setZero();
    row(step) = du_scale;
    add_square(row, 0.0, config_.delta_input_scale_m_s2);

    position_sensitivity.push_back(sensitivity.row(0));
    position_constant.push_back(constant(0));
  }

  const ReferencePoint final_target = reference.empty() ? ReferencePoint{} : reference.back();
  const double final_acceleration = chassis_acceleration_ref.empty() ? 0.0 : chassis_acceleration_ref.back();
  Eigen::Vector4d terminal_error = constant;
  terminal_error(0) -= final_target.position_m;
  terminal_error(1) -= final_target.velocity_m_s;
  terminal_error(2) -= final_acceleration +
                       final_target.acceleration_m_s2 / config_.rolling_lambda;
  terminal_error(3) -= final_acceleration +
                       final_target.acceleration_m_s2 / config_.rolling_lambda;
  Eigen::MatrixXd terminal_sensitivity = Eigen::MatrixXd::Zero(4, variables);
  terminal_sensitivity.leftCols(n) = sensitivity;
  if (build_structure) {
    hessian += 2.0 * terminal_sensitivity.transpose() * terminal_cost_ *
               terminal_sensitivity;
  }
  gradient += 2.0 * terminal_sensitivity.transpose() * terminal_cost_ * terminal_error;

  if (build_structure) {
    for (int step = 0; step < n; ++step) {
      hessian(n + step, n + step) += 2.0 * config_.slack_weight;
    }
    hessian.diagonal().array() += 1e-8;
  }

  Eigen::MatrixXd constraint =
      build_structure ? Eigen::MatrixXd::Zero(constraints, variables)
                      : cached_constraint_;
  Eigen::VectorXd lower = Eigen::VectorXd::Constant(
      constraints, -std::numeric_limits<double>::infinity());
  Eigen::VectorXd upper = Eigen::VectorXd::Constant(
      constraints, std::numeric_limits<double>::infinity());
  const double u_limit = kGravity * std::tan(config_.theta_limit_rad);
  const double delta_limit = kGravity * std::tan(config_.theta_rate_limit_rad_s * dt);
  for (int step = 0; step < n; ++step) {
    if (build_structure) constraint(step, step) = 1.0;
    lower(step) = -delta_limit / du_scale;
    upper(step) = delta_limit / du_scale;

    if (build_structure) {
      for (int input = 0; input <= step; ++input) {
        constraint(n + step, input) = du_scale / u_scale;
      }
    }
    lower(n + step) = (-u_limit - previous_command_m_s2) / u_scale;
    upper(n + step) = (u_limit - previous_command_m_s2) / u_scale;

    if (build_structure) {
      constraint(2 * n + step, n + step) = 1.0;
    }
    lower(2 * n + step) = 0.0;

    if (build_structure) {
      constraint.block(3 * n + step, 0, 1, n) =
          position_sensitivity[static_cast<std::size_t>(step)] / x_scale;
      constraint(3 * n + step, n + step) = -1.0;
    }
    upper(3 * n + step) =
        (config_.position_soft_limit_m - position_constant[static_cast<std::size_t>(step)]) /
        x_scale;

    if (build_structure) {
      constraint.block(4 * n + step, 0, 1, n) =
          -position_sensitivity[static_cast<std::size_t>(step)] / x_scale;
      constraint(4 * n + step, n + step) = -1.0;
    }
    upper(4 * n + step) =
        (config_.position_soft_limit_m + position_constant[static_cast<std::size_t>(step)]) /
        x_scale;
  }
  if (build_structure) {
    cached_hessian_ = hessian;
    cached_constraint_ = constraint;
  }
  return QpProblem{std::move(hessian), std::move(gradient), std::move(constraint),
                   std::move(lower), std::move(upper)};
}

MpcResult BallMpc::solve(const std::array<double, 4>& state, double previous_command_m_s2,
                         const std::vector<ReferencePoint>& reference,
                         const std::vector<double>& chassis_acceleration_ref) {
  const auto started = std::chrono::steady_clock::now();
  std::vector<Eigen::RowVectorXd> position_sensitivity;
  std::vector<double> position_constant;
  last_problem_ = build_problem(state, previous_command_m_s2, reference,
                                chassis_acceleration_ref, position_sensitivity,
                                position_constant);
  const auto problem_built = std::chrono::steady_clock::now();
  QpResult qp = solver_->solve(last_problem_);
  const auto ended = std::chrono::steady_clock::now();

  MpcResult result;
  result.solve_time_ms = std::chrono::duration<double, std::milli>(ended - started).count();
  result.qp_build_time_ms =
      std::chrono::duration<double, std::milli>(problem_built - started)
          .count();
  result.qp_setup_time_ms = qp.setup_time_ms;
  result.qp_update_time_ms = qp.update_time_ms;
  result.qp_backend_solve_time_ms = qp.solve_time_ms;
  result.timed_out = result.solve_time_ms > config_.solver_deadline_ms;
  result.solved = qp.solved && !result.timed_out && qp.primal.size() == 2 * config_.horizon;
  result.iterations = qp.iterations;
  result.backend = solver_->name();
  if (!result.solved) return result;

  result.command_sequence.reserve(static_cast<std::size_t>(config_.horizon));
  result.predicted_position.reserve(static_cast<std::size_t>(config_.horizon));
  double command = previous_command_m_s2;
  for (int step = 0; step < config_.horizon; ++step) {
    command += config_.delta_input_scale_m_s2 * qp.primal(step);
    result.command_sequence.push_back(command);
    const double predicted = position_constant[static_cast<std::size_t>(step)] +
                             position_sensitivity[static_cast<std::size_t>(step)].dot(
                                 qp.primal.head(config_.horizon));
    result.predicted_position.push_back(predicted);
    result.max_slack_m = std::max(
        result.max_slack_m,
        config_.position_scale_m * std::max(0.0, qp.primal(config_.horizon + step)));
  }
  result.command_m_s2 = result.command_sequence.front();
  last_command_sequence_ = result.command_sequence;
  return result;
}

std::vector<double> BallMpc::shift_last_solution() const {
  if (last_command_sequence_.empty()) return {};
  std::vector<double> shifted = last_command_sequence_;
  for (std::size_t index = 1; index < shifted.size(); ++index) shifted[index - 1] = shifted[index];
  shifted.back() = shifted.size() > 1 ? shifted[shifted.size() - 2] : shifted.back();
  return shifted;
}

}  // namespace nx_control
