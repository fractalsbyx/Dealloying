#pragma once

#include <concepts>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>


template <typename number, unsigned int dim, unsigned int order>
struct Dual;

template <typename T>
struct upgrades_to_dual : std::false_type
{};

template <std::floating_point FP>
struct upgrades_to_dual<FP> : std::true_type
{};

template <std::floating_point FP>
struct upgrades_to_dual<dealii::VectorizedArray<FP>> : std::true_type
{};

template <typename Real, unsigned int dim, unsigned int order>
struct upgrades_to_dual<Dual<Real, dim, order>> : std::true_type
{};

template <int rank, int dim, typename Number>
struct upgrades_to_dual<dealii::Tensor<rank, dim, Number>> : std::false_type
{};

template <typename T>
concept UpgradesToDual = upgrades_to_dual<T>::value;

/**
 * @brief Class for holding fields and their spatial derivatives
 * @tparam dim The dimension of the problem
 */
template <typename number, unsigned int dim, unsigned int order = 1>
struct Dual
{
  static_assert(order >= 1,
                "Dual+ must have an order greater than or equal to one.");
  static_assert(dim >= 1,
                "Tensors must have a dimension greater than or equal to zero.");
  using ValueType = number;
  using GradType =
      std::conditional_t<order == 1, dealii::Tensor<1, dim, number>,
                         dealii::Tensor<1, dim, Dual<number, dim, order - 1>>>;

  ValueType val  = number(0.);
  GradType  grad = {};

  template <typename _valuetype = ValueType, typename _gradtype = GradType>
  Dual(const _valuetype &_val = number(0.), const _gradtype &_grad = {})
      : val(_val), grad(_grad)
  {}

  Dual<number, dim>
  operator+(const Dual<number, dim> &other) const
  {
    return {val + other.val, grad + other.grad};
  }

  Dual<number, dim>
  operator+(const ValueType &constant) const
  {
    return {val + constant, grad};
  }

  Dual<number, dim>
  operator+(const double &constant) const
  {
    return {val + constant, grad};
  }

  Dual<number, dim>
  operator+() const
  {
    return *this;
  }

  template <typename other_number>
  Dual<number, dim> &
  operator+=(const other_number &other)
  {
    *this = *this + other;
    return *this;
  }

  Dual<number, dim>
  operator-(const Dual<number, dim> &other) const
  {
    return {val - other.val, grad - other.grad};
  }

  Dual<number, dim>
  operator-() const
  {
    return {-val, -grad};
  }

  Dual<number, dim>
  operator-(const ValueType &constant) const
  {
    return {val - constant, grad};
  }

  Dual<number, dim>
  operator-(const double &constant) const
  {
    return {val - constant, grad};
  }

  template <typename other_number>
  Dual<number, dim> &
  operator-=(const other_number &other)
  {
    *this = *this - other;
    return *this;
  }

  Dual<number, dim>
  operator*(const ValueType &constant) const
  {
    return {val * constant, grad * constant};
  }

  Dual<number, dim>
  operator*(const double &constant) const
  {
    return {val * constant, grad * constant};
  }

  Dual<number, dim>
  operator*(const Dual<number, dim> &other) const
  {
    return Dual<number, dim>(val * other.val,
                             (val * other.grad) + (grad * other.val));
  }

  template <typename other_number>
  Dual<number, dim> &
  operator*=(const other_number &other)
  {
    *this = *this * other;
    return *this;
  }

  Dual<number, dim>
  operator/(const ValueType &constant) const
  {
    return Dual<number, dim>{val / constant, grad / constant};
  }

  Dual<number, dim>
  operator/(const double &constant) const
  {
    return Dual<number, dim>{val / constant, grad / constant};
  }

  Dual<number, dim>
  operator/(const Dual<number, dim> &other) const
  {
    return Dual<number, dim>{val / other.val,
                             (grad * other.val - val * other.grad) /
                                 (other.val * other.val)};
  }

  template <typename other_number>
  Dual<number, dim> &
  operator/=(const other_number &other)
  {
    *this = *this / other;
    return *this;
  }
};

template <typename number, unsigned int dim, typename other_number>
Dual<number, dim>
operator+(const other_number &other, const Dual<number, dim> &field)
{
  return field + other;
}

template <typename number, unsigned int dim, typename other_number>
Dual<number, dim>
operator-(const other_number &other, const Dual<number, dim> &field)
{
  return -field + other;
}

template <typename number, unsigned int dim, UpgradesToDual other_number>
Dual<number, dim>
operator*(const other_number &other, const Dual<number, dim> &field)
{
  return field * other;
}

template <typename number, unsigned int dim, UpgradesToDual other_number>
Dual<number, dim>
operator/(const other_number &other, const Dual<number, dim> &field)
{
  static const Dual<number, dim> one = Dual<number, dim>{number(1.0), {}};
  return (one / field) * other;
}

template <typename number, unsigned int dim>
Dual<number, dim>
sqrt(const Dual<number, dim> &field)
{
  using std::sqrt;
  number sqrt_val = sqrt(field.val);
  return Dual<number, dim>{sqrt_val, field.grad / (2.0 * sqrt_val)};
}

template <typename number, unsigned int dim>
Dual<number, dim>
log(const Dual<number, dim> &field)
{
  using std::log;
  return Dual<number, dim>{log(field.val), field.grad / field.val};
}

template <typename number, unsigned int dim, unsigned int order>
struct dealii::EnableIfScalar<Dual<number, dim, order>>
{
  using type = Dual<number, dim, order>;
};
