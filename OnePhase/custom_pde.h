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

  number RT            = 1073.15*8.314; // J
  number D1            = 1.0;  // nm^2/s
  number D2            = 100.0;   // nm^2/s
  number Vm            = 7.0e-6*1e27; // nm^3/mol
  number deltaG0       = -1.0*RT;  // J
  number j0            = 0.01*1e-18; // mol/s/nm^2 // desired: 0.1
  number l_int         = 1.0;   // nm
  number gamma         = 2.0*1e-18;   // J/nm^2
  number x2            = 0.05;

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
    [[maybe_unused]] const double lx = (dim > 0) ? mesh_size[0] : 0.;
    [[maybe_unused]] const double ly = (dim > 1) ? mesh_size[1] : 0.;
    [[maybe_unused]] const double lz = (dim > 2) ? mesh_size[2] : 0.;
    // ===========================================================================
    // FUNCTION FOR INITIAL CONDITIONS
    // ===========================================================================
    using std::max;
    using std::min;
    using std::sin;
    constexpr double pi  = 3.14159265359;
    constexpr double amplitude  = 0.125;

    if (index == 0)
      {
        double y    = point[1];
        double x    = point[0];
        double y_shift =
             (  sin( 4.0*pi * (x/lx + 0.3)) * amplitude
             + sin( 9.0*pi * (x/lx + 0.)) * amplitude )
            * 0.5*(1.0 + tanh(( x - lx * 0.04) / (lx * 0.04)))
            * 0.5*(1.0 + tanh((-x + lx * 0.96) / (lx * 0.04)));
        double y_shifted = (y - ly * 0.9 - y_shift);
        double flat      = interface(-y_shifted);
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
    constexpr double pi = 3.14159265359;
    const number dt = sim_timer.get_timestep();
    if (solve_block_id == 0) // n, x
      {

        const ScalarValue n       = variable_list.template get_value<Scalar, OldOne>(0);
        const ScalarGrad  n_grad  = variable_list.template get_gradient<Scalar, OldOne>(0);
        const ScalarValue x1      = variable_list.template get_value<Scalar, OldOne>(1);
        const ScalarValue rxn     = variable_list.template get_value<Scalar, OldOne>(3);
        const ScalarValue x1_stage = variable_list.template get_value<Scalar, OldOne>(4);

        // n
        variable_list.set_value_term(0, n + dt * rxn);

        // x1
        variable_list.set_value_term(1, x1 + dt * x1_stage);
      }
    else if (solve_block_id == 1) // potential
      {

        const ScalarValue n      = variable_list.template get_value<Scalar, Current>(0);
        const ScalarGrad  n_grad = variable_list.template get_gradient<Scalar, Current>(0);
        const ScalarValue x1     = variable_list.template get_value<Scalar, Current>(1);

        // rxn_mu
        const ScalarValue deltaG_val =
            (std::log(x2/x1)) * RT + deltaG0 + 4.0*gamma*Vm/l_int * (2.0 * n - 1.0);
        variable_list.set_value_term(2, deltaG_val);
        variable_list.set_gradient_term(2, -n_grad * 8.0*gamma*Vm*l_int/(pi*pi));
      }
    else if (solve_block_id == 2) // rxn
      {
        constexpr double upper(1.0 - 1e-4);
        constexpr double lower(1e-4);
        const ScalarValue      n      = variable_list.template get_value<Scalar, Current>(0);
        const ScalarGrad       n_grad = variable_list.template get_gradient<Scalar, Current>(0);
        const ScalarValue deltaG = variable_list.template get_value<Scalar, Current>(2);

        ScalarValue rxn_val = -n_grad.norm_square() * Vm * j0 * (-deltaG/RT);
        constrain_dvaldt(n, rxn_val, dt, lower, upper);
        variable_list.set_value_term(3, rxn_val);
      }
    else if (solve_block_id == 3) // x1_stage
      {
        const ScalarValue n       = variable_list.template get_value<Scalar, Current>(0);
        const ScalarGrad  n_grad  = variable_list.template get_gradient<Scalar, Current>(0);
        const ScalarValue x1      = variable_list.template get_value<Scalar, Current>(1);
        const ScalarGrad  x1_grad = variable_list.template get_gradient<Scalar, Current>(1);
        const ScalarValue rxn = variable_list.template get_value<Scalar, Current>(3);

        // x1
        variable_list.set_value_term(4, dt * (D1 * x1_grad * n_grad / n + rxn));
        variable_list.set_gradient_term(4, dt * (-D1 * x1_grad));
      }
  }

  void
  compute_lhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    if (solve_block_id == 3) // linear lhs
      {
        const number dt = sim_timer.get_timestep();
        const ScalarValue n       = variable_list.template get_value<Scalar, Current>(0);
        const ScalarGrad  n_grad  = variable_list.template get_gradient<Scalar, Current>(0);
        const ScalarValue x1_stage      = variable_list.template get_value<Scalar, LHS>(4);
        const ScalarGrad  x1_stage_grad = variable_list.template get_gradient<Scalar, LHS>(4);
        variable_list.set_value_term(4, x1_stage - dt * (D1 * x1_stage_grad * n_grad / n));
        variable_list.set_gradient_term(4, -dt * (-D1 * x1_stage_grad));
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

  /**
   *@brief return the double obstacle interface function
   */
  template <typename real>
  const real
  interface(const real &x) const
  {
    using std::max;
    using std::min;
    using std::sin; 
    constexpr double pi = 3.14159265359;
    return 0.5 * (1.0 + sin(pi * max(-0.5, min(0.5, x / l_int))));
  }
};

PRISMS_PF_END_NAMESPACE
