// SPDX-FileCopyrightText: © 2025 Xander Mensah
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

#include <complex.h>
#include <conversion.h>

#include <dual.h>
#include <variation.h>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
class CustomPDE : public PDEOperatorBase<dim, degree, number>
{
public:
  using ScalarValue     = dealii::VectorizedArray<number>;
  using ScalarGrad      = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess      = dealii::Tensor<2, dim, ScalarValue>;
  using ScalarField     = Dual<ScalarValue, dim>;
  using ScalarVariation = Variation<ScalarValue, dim>;

  using VectorValue     = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad      = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess      = dealii::Tensor<3, dim, ScalarValue>;
  using VectorField     = Dual<VectorValue, dim>;
  using VectorVariation = Variation<VectorValue, dim>;

  using ComplexValue     = Complex<ScalarValue>;
  using ComplexGrad      = dealii::Tensor<1, dim, ComplexValue>;
  using ComplexHess      = dealii::Tensor<2, dim, ComplexValue>;
  using ComplexField     = Dual<ComplexValue, dim>;
  using ComplexVariation = Variation<ComplexValue, dim>;

  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;

  number      D1            = 1.0;
  number      D2            = 1.0;
  number      deltaG        = 1.0;
  number      alpha         = 0.5;
  number      k0            = 1.0;
  number      l_int         = 4.0;
  number      epsilon_denom = 1e-5;
  number      sigma         = 2.0;
  number      dw_coeff      = sigma * 4.0 / l_int;
  number      grad_coeff    = sigma * 4.0 * l_int / (3.14159 * 3.14159);
  ScalarValue x2            = 0.05;

  /**
   * @brief Constructor.
   */
  explicit CustomPDE(const UserInputParameters<dim> &_user_inputs, PhaseFieldTools<dim> &_pf_tools)
      : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools)
  // D1(_user_inputs.user_constants.get_model_constant_double("D1")),
  // D2(_user_inputs.user_constants.get_model_constant_double("D2")),
  // epsilon_denom(_user_inputs.user_constants.get_model_constant_double("epsilon_denom")),
  // deltaG(_user_inputs.user_constants.get_model_constant_double("deltaG")),
  // dw_coeff(_user_inputs.user_constants.get_model_constant_double("dw_coeff")),
  // grad_coeff(_user_inputs.user_constants.get_model_constant_double("grad_coeff"))
  {}

private:
  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    // Custom coordinate system
    const dealii::Tensor<1, dim> &mesh_size =
        get_user_inputs().spatial_discretization.rectangular_mesh.size;
    const dealii::Point<dim>      center(mesh_size / 2.0);
    const dealii::Point<dim>      p(point - center);
    [[maybe_unused]] const double x = (dim > 0) ? p[0] : 0.;
    [[maybe_unused]] const double y = (dim > 1) ? p[1] : 0.;
    [[maybe_unused]] const double z = (dim > 2) ? p[2] : 0.;
    // ===========================================================================
    // FUNCTION FOR INITIAL CONDITIONS
    // ===========================================================================
    using std::max;
    using std::min;
    using std::sin;
    using std::numbers::pi;

    if (index == 0)
      {
        double y    = point[1];
        double x    = point[0];
        double ys   = (0.5 * (mesh_size[1] * 0.75 - y +
                            std::sin(2.0 * 3.14 * (2.845 * x + 1.0) / mesh_size[0]) * 2.0 +
                            std::sin(2.0 * 3.14 * (7.123 * x) / mesh_size[0]) * 1.0));
        double flat = 0.5 * (1.0 + sin(pi * max(-0.5, min(0.5, std::sqrt(2.0) * ys / l_int))));

        scalar_value = max(min(flat, 1.0 - 1e-4), 1e-4);
        return;
      }
    if (index == 1)
      {
        scalar_value = 0.2;
        return;
      }
    if (index == 2)
      {
        scalar_value = 0.05;
        return;
      }
    if (index == 3)
      {
        scalar_value = 0.0;
        return;
      }
  }
  void
  set_dirichlet([[maybe_unused]] const unsigned int       &index,
                [[maybe_unused]] const unsigned int       &boundary_id,
                [[maybe_unused]] const unsigned int       &component,
                [[maybe_unused]] const dealii::Point<dim> &point,
                [[maybe_unused]] number                   &scalar_value,
                [[maybe_unused]] number                   &vector_component_value) const override
  {
    scalar_value = 0.05;
  }

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int                         solve_block_id) const override
  {
    const number dt = sim_timer.get_timestep();
    if (solve_block_id == 0) // n, x
      {

        const ScalarValue n       = variable_list.template get_value<Scalar, OldOne>(0);
        const ScalarGrad  n_grad  = variable_list.template get_gradient<Scalar, OldOne>(0);
        const ScalarValue x1      = variable_list.template get_value<Scalar, OldOne>(1);
        const ScalarGrad  x1_grad = variable_list.template get_gradient<Scalar, OldOne>(1);
        const ScalarValue rxn     = variable_list.template get_value<Scalar, OldOne>(3);

        // n
        variable_list.set_value_term(0, n + dt * rxn);

        // x1
        variable_list.set_value_term(1,
                                     x1 + dt * (D1 * x1_grad * n_grad / (n + epsilon_denom) + rxn));
        variable_list.set_gradient_term(1, dt * (-D1 * x1_grad));
      }
    else if (solve_block_id == 1) // potential
      {

        const ScalarValue n      = variable_list.template get_value<Scalar, Current>(0);
        const ScalarGrad  n_grad = variable_list.template get_gradient<Scalar, Current>(0);
        const ScalarValue x1     = variable_list.template get_value<Scalar, Current>(1);
        // const ScalarGrad  x1_grad = variable_list.template get_gradient<Scalar, Current>(1);

        // rxn_mu
        const ScalarValue rxn_mu_val =
            std::log(x1) - std::log(x2) + deltaG + dw_coeff * (1.0 - 2.0 * n);
        variable_list.set_value_term(4, rxn_mu_val);
        variable_list.set_gradient_term(4, n_grad * grad_coeff);
      }
    else if (solve_block_id == 2) // rxn
      {
        constexpr double upper(1.0 - 1e-4);
        constexpr double lower(1e-4);
        ScalarValue      n      = variable_list.template get_value<Scalar, Current>(0);
        ScalarGrad       n_grad = variable_list.template get_gradient<Scalar, Current>(0);

        ScalarValue rxn_mu = variable_list.template get_value<Scalar, Current>(4);

        ScalarValue rxn_val = -n_grad.norm_square() * rxn_mu;
        constrain_dvaldt(n, rxn_val, dt, lower, upper);
        variable_list.set_value_term(3, rxn_val);
      }
  }

private:
  template <typename num>
  void
  constrain_dvaldt(const num &val, num &dvaldt, double dt, double lower = 0.0,
                   double upper = 1.0) const
  {
    using std::max;
    using std::min;
    num top = max(val + dvaldt * dt, num(upper));
    num bot = min(val + dvaldt * dt, num(lower));
    dvaldt  = (dvaldt * dt + (upper - top - (bot - lower))) / dt;
  }
};

PRISMS_PF_END_NAMESPACE
