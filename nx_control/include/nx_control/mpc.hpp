#pragma once

#include "nx_control/types.hpp"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <memory>
#include <string>
#include <vector>

namespace nx_control {

struct QpProblem {
  Eigen::MatrixXd hessian;
  Eigen::VectorXd gradient;
  Eigen::MatrixXd constraint;
  Eigen::VectorXd lower;
  Eigen::VectorXd upper;
};

struct QpResult {
  bool solved = false;
  int iterations = 0;
  Eigen::VectorXd primal;
  std::string status;
};

class QpSolver {
 public:
  virtual ~QpSolver() = default;
  virtual QpResult solve(const QpProblem& problem) = 0;
  virtual const char* name() const = 0;
};

class DenseAdmmSolver final : public QpSolver {
 public:
  DenseAdmmSolver(int max_iterations, double eps_abs, double eps_rel);
  QpResult solve(const QpProblem& problem) override;
  const char* name() const override { return "dense-admm"; }

 private:
  int max_iterations_;
  double eps_abs_;
  double eps_rel_;
  double rho_ = 1.0;
  double sigma_ = 1e-6;
  Eigen::VectorXd x_;
  Eigen::VectorXd z_;
  Eigen::VectorXd y_;
};

#ifdef NX_CONTROL_HAS_OSQP
std::unique_ptr<QpSolver> make_osqp_solver(const ControlConfig& config);
#endif
std::unique_ptr<QpSolver> make_default_qp_solver(const ControlConfig& config);

class BallMpc {
 public:
  explicit BallMpc(const ControlConfig& config,
                   std::unique_ptr<QpSolver> solver = nullptr);

  MpcResult solve(const std::array<double, 4>& state, double previous_command_m_s2,
                  const std::vector<ReferencePoint>& reference,
                  const std::vector<double>& chassis_acceleration_ref);
  std::vector<double> shift_last_solution() const;
  const QpProblem& last_problem() const { return last_problem_; }
  const char* backend_name() const { return solver_->name(); }

 private:
  Eigen::Matrix4d terminal_cost() const;
  QpProblem build_problem(const std::array<double, 4>& state,
                          double previous_command_m_s2,
                          const std::vector<ReferencePoint>& reference,
                          const std::vector<double>& chassis_acceleration_ref,
                          std::vector<Eigen::RowVectorXd>& position_sensitivity,
                          std::vector<double>& position_constant) const;

  ControlConfig config_;
  std::unique_ptr<QpSolver> solver_;
  Eigen::Matrix4d terminal_cost_ = Eigen::Matrix4d::Identity();
  std::vector<double> last_command_sequence_;
  QpProblem last_problem_;
};

}  // namespace nx_control
