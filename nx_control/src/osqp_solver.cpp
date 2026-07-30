#include "nx_control/mpc.hpp"

#include <osqp.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nx_control {
namespace {

class OsqpSolver final : public QpSolver {
 public:
  explicit OsqpSolver(const ControlConfig& config) : config_(config) {}
  ~OsqpSolver() override { cleanup(); }

  const char* name() const override { return "osqp"; }

  QpResult solve(const QpProblem& problem) override {
    bool setup_this_call = false;
    if (workspace_ == nullptr) {
      if (!setup(problem)) {
        QpResult result;
        result.status = "setup_failed";
        return result;
      }
      setup_this_call = true;
    } else if (problem.hessian.rows() != variables_ || problem.constraint.rows() != constraints_) {
      cleanup();
      if (!setup(problem)) {
        QpResult result;
        result.status = "setup_failed";
        return result;
      }
      setup_this_call = true;
    } else {
      copy_vector(problem.gradient, q_);
      copy_vector(problem.lower, lower_);
      copy_vector(problem.upper, upper_);
      if (osqp_update_lin_cost(workspace_, q_.data()) != 0 ||
          osqp_update_bounds(workspace_, lower_.data(), upper_.data()) != 0) {
        QpResult result;
        result.status = "update_failed";
        return result;
      }
    }
    osqp_solve(workspace_);
    QpResult result;
    result.iterations = static_cast<int>(workspace_->info->iter);
#ifdef PROFILING
    result.setup_time_ms =
        setup_this_call ? 1000.0 * workspace_->info->setup_time : 0.0;
    result.update_time_ms = 1000.0 * workspace_->info->update_time;
    result.solve_time_ms = 1000.0 * workspace_->info->solve_time;
#endif
    result.solved = workspace_->info->status_val == OSQP_SOLVED ||
                    workspace_->info->status_val == OSQP_SOLVED_INACCURATE;
    result.status = workspace_->info->status;
    if (workspace_->solution != nullptr && workspace_->solution->x != nullptr) {
      result.primal.resize(variables_);
      for (c_int index = 0; index < variables_; ++index) {
        result.primal(index) = workspace_->solution->x[index];
      }
    }
    return result;
  }

 private:
  static void copy_vector(const Eigen::VectorXd& source, std::vector<c_float>& target) {
    target.resize(static_cast<std::size_t>(source.size()));
    for (Eigen::Index index = 0; index < source.size(); ++index) {
      const double value = source(index);
      target[static_cast<std::size_t>(index)] = static_cast<c_float>(
          std::isfinite(value) ? value : (value < 0.0 ? -OSQP_INFTY : OSQP_INFTY));
    }
  }

  static csc* dense_to_csc(const Eigen::MatrixXd& matrix, bool upper_triangle,
                           std::vector<c_float>& values, std::vector<c_int>& rows,
                           std::vector<c_int>& columns) {
    values.clear();
    rows.clear();
    columns.clear();
    columns.push_back(0);
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
      for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        if (upper_triangle && row > column) continue;
        const double value = matrix(row, column);
        if (std::abs(value) <= 1e-14) continue;
        values.push_back(static_cast<c_float>(value));
        rows.push_back(static_cast<c_int>(row));
      }
      columns.push_back(static_cast<c_int>(values.size()));
    }
    return csc_matrix(static_cast<c_int>(matrix.rows()), static_cast<c_int>(matrix.cols()),
                      static_cast<c_int>(values.size()), values.data(), rows.data(),
                      columns.data());
  }

  bool setup(const QpProblem& problem) {
    variables_ = static_cast<c_int>(problem.hessian.rows());
    constraints_ = static_cast<c_int>(problem.constraint.rows());
    copy_vector(problem.gradient, q_);
    copy_vector(problem.lower, lower_);
    copy_vector(problem.upper, upper_);
    p_matrix_ = dense_to_csc(problem.hessian, true, p_values_, p_rows_, p_columns_);
    a_matrix_ = dense_to_csc(problem.constraint, false, a_values_, a_rows_, a_columns_);
    data_ = static_cast<OSQPData*>(c_malloc(sizeof(OSQPData)));
    settings_ = static_cast<OSQPSettings*>(c_malloc(sizeof(OSQPSettings)));
    if (data_ == nullptr || settings_ == nullptr || p_matrix_ == nullptr || a_matrix_ == nullptr) {
      cleanup();
      return false;
    }
    data_->n = variables_;
    data_->m = constraints_;
    data_->P = p_matrix_;
    data_->A = a_matrix_;
    data_->q = q_.data();
    data_->l = lower_.data();
    data_->u = upper_.data();
    osqp_set_default_settings(settings_);
    settings_->verbose = false;
    settings_->warm_start = true;
    settings_->polish = false;
    settings_->rho = config_.qp_rho;
    settings_->adaptive_rho_interval =
        config_.qp_adaptive_rho_interval;
    settings_->scaled_termination =
        config_.qp_scaled_termination ? 1 : 0;
    settings_->max_iter = config_.qp_max_iterations;
    settings_->eps_abs = config_.qp_eps_abs;
    settings_->eps_rel = config_.qp_eps_rel;
    settings_->check_termination =
        config_.qp_check_termination_interval;
#ifdef PROFILING
    settings_->time_limit = config_.solver_deadline_ms / 1000.0;
#endif
    return osqp_setup(&workspace_, data_, settings_) == 0;
  }

  void cleanup() {
    if (workspace_ != nullptr) osqp_cleanup(workspace_);
    workspace_ = nullptr;
    if (p_matrix_ != nullptr) c_free(p_matrix_);
    if (a_matrix_ != nullptr) c_free(a_matrix_);
    if (data_ != nullptr) c_free(data_);
    if (settings_ != nullptr) c_free(settings_);
    p_matrix_ = a_matrix_ = nullptr;
    data_ = nullptr;
    settings_ = nullptr;
  }

  ControlConfig config_;
  c_int variables_ = 0;
  c_int constraints_ = 0;
  OSQPWorkspace* workspace_ = nullptr;
  OSQPData* data_ = nullptr;
  OSQPSettings* settings_ = nullptr;
  csc* p_matrix_ = nullptr;
  csc* a_matrix_ = nullptr;
  std::vector<c_float> p_values_, a_values_, q_, lower_, upper_;
  std::vector<c_int> p_rows_, p_columns_, a_rows_, a_columns_;
};

}  // namespace

std::unique_ptr<QpSolver> make_osqp_solver(const ControlConfig& config) {
  return std::make_unique<OsqpSolver>(config);
}

}  // namespace nx_control
