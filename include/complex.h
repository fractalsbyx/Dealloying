#pragma once

#include <cmath>
#include <concepts>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

// sorry
#define USE_STD_MATH \
  using std::abs;    \
  using std::acosh;  \
  using std::asinh;  \
  using std::atan;   \
  using std::atan2;  \
  using std::atanh;  \
  using std::cos;    \
  using std::cosh;   \
  using std::exp;    \
  using std::log;    \
  using std::pow;    \
  using std::sin;    \
  using std::sinh;   \
  using std::sqrt;   \
  using std::tan;    \
  using std::tanh;

template <typename Real>
struct Complex;

template <typename T>
struct is_real : std::false_type
{};

template <typename Real>
struct is_real<Complex<Real>> : std::false_type
{};

template <std::floating_point FP>
struct is_real<FP> : std::true_type
{};

template <typename T>
concept RealTypeName = is_real<T>::value;

template <typename L, typename R>
struct Prod
{
  using type = decltype(L() * R());
};
template <typename L, typename R>
struct Sum
{
  using type = decltype(L() * R());
};

template <typename Real = double>
struct Complex
{

  Real x;
  Real y;
  Complex(Real x_val = 0.0, Real y_val = 0.0) : x(x_val), y(y_val) {}
  // Complex(const Complex &other) : x(other.x), y(other.y) {}

  // Binary operators of complex numbers
  template <typename RReal = Real>
  Complex<typename Sum<Real, RReal>::type>
  operator+(const Complex<RReal> &other) const
  {
    return Complex(x + other.x, y + other.y);
  }
  template <typename RReal = Real>
  Complex<typename Sum<Real, RReal>::type>
  operator-(const Complex<RReal> &other) const
  {
    return Complex(x - other.x, y - other.y);
  }
  template <typename RReal = Real>
  Complex<typename Prod<Real, RReal>::type>
  operator*(const Complex<RReal> &other) const
  {
    return Complex(x * other.x - y * other.y, x * other.y + y * other.x);
  }
  template <typename RReal = Real>
  Complex<typename Prod<Real, RReal>::type>
  operator/(const Complex<RReal> &other) const
  {
    Real denom = other.x * other.x + other.y * other.y;
    return Complex((x * other.x + y * other.y) / denom,
                   (y * other.x - x * other.y) / denom);
  }

  // Binary operators of complex numbers with scalar
  template <typename RReal = Real>
  Complex<typename Sum<Real, RReal>::type>
  operator+(const RReal &scalar) const
  {
    return Complex(x + scalar, y);
  }
  template <typename RReal = Real>
  Complex<typename Sum<Real, RReal>::type>
  operator-(const RReal &scalar) const
  {
    return Complex(x - scalar, y);
  }
  template <RealTypeName RReal = Real>
  Complex<typename Prod<Real, RReal>::type>
  operator*(const RReal &scalar) const
  {
    return Complex(x * scalar, y * scalar);
  }
  template <RealTypeName RReal = Real>
  Complex<typename Prod<Real, RReal>::type>
  operator/(const RReal &scalar) const
  {
    return Complex(x / scalar, y / scalar);
  }

  // Assignment operators with complex numbers
  template <typename RReal = Real>
  Complex &
  operator=(const Complex<RReal> &other)
  {
    if (this != &other)
      {
        x = other.x;
        y = other.y;
      }
    return *this;
  }
  template <typename RReal = Real>
  Complex &
  operator+=(const Complex<RReal> &other)
  {
    x += other.x;
    y += other.y;
    return *this;
  }
  template <typename RReal = Real>
  Complex &
  operator-=(const Complex<RReal> &other)
  {
    x -= other.x;
    y -= other.y;
    return *this;
  }
  template <typename RReal = Real>
  Complex &
  operator*=(const Complex<RReal> &other)
  {
    Real temp_x = x;
    x           = x * other.x - y * other.y;
    y           = temp_x * other.y + y * other.x;
    return *this;
  }
  template <typename RReal = Real>
  Complex &
  operator/=(const Complex<RReal> &other)
  {
    Real denom  = other.x * other.x + other.y * other.y;
    Real temp_x = x;
    x           = (x * other.x + y * other.y) / denom;
    y           = (y * other.x - temp_x * other.y) / denom;
    return *this;
  }

  // Assignment operators with scalar
  template <typename RReal = Real>
  Complex &
  operator=(const RReal &scalar)
  {
    x = scalar;
    y = Real(0);
    return *this;
  }
  template <typename RReal = Real>
  Complex &
  operator+=(const RReal &scalar)
  {
    x += scalar;
    return *this;
  }
  template <typename RReal = Real>
  Complex &
  operator-=(const RReal &scalar)
  {
    x -= scalar;
    return *this;
  }
  template <typename RReal = Real>
  Complex &
  operator*=(const RReal &scalar)
  {
    x *= scalar;
    y *= scalar;
    return *this;
  }
  template <typename RReal = Real>
  Complex &
  operator/=(const RReal &scalar)
  {
    x /= scalar;
    y /= scalar;
    return *this;
  }

  // Unary operators
  Complex
  operator-() const
  {
    return Complex(-x, -y);
  }
  Complex
  operator+() const
  {
    return *this;
  }
  Complex
  operator~() const
  {
    return Complex(x, -y);
  }

  // Other functions

  Real
  real() const
  {
    return x;
  }
  Real
  imag() const
  {
    return y;
  }
  Complex
  conjugate() const
  {
    return Complex(x, -y);
  }
  Real
  mag() const
  {
    using std::sqrt;
    return sqrt(x * x + y * y);
  }
  Real
  mag2() const
  {
    return x * x + y * y;
  }

  Real
  arg() const
  {
    using std::atan2;
    return atan2(y, x);
  }
  Complex
  exp() const
  {
    USE_STD_MATH
    return Complex(exp(x) * cos(y), exp(x) * sin(y));
  }
  Complex
  log() const
  {
    using std::log;
    return Complex(log(mag()), arg());
  }

  Complex
  sqrt() const
  {
    USE_STD_MATH
    return Complex(sqrt(mag()) * cos(arg() / 2), sqrt(mag()) * sin(arg() / 2));
  }
  Complex
  sin() const
  {
    USE_STD_MATH
    return Complex(sin(x) * cosh(y), cos(x) * sinh(y));
  }
  Complex
  cos() const
  {
    USE_STD_MATH
    return Complex(cos(x) * cosh(y), -sin(x) * sinh(y));
  }
  Complex
  tan() const
  {
    return sin() / cos();
  }
  Complex
  sinh() const
  {
    USE_STD_MATH
    return Complex(sinh(x) * cos(y), cosh(x) * sin(y));
  }
  Complex
  cosh() const
  {
    USE_STD_MATH
    return Complex(cosh(x) * cos(y), sinh(x) * sin(y));
  }
  Complex
  tanh() const
  {
    return sinh() / cosh();
  }
  Complex
  asin() const
  {
    USE_STD_MATH
    return Complex(asin(x) * cosh(y), atan2(sinh(y), x));
  }
  Complex
  acos() const
  {
    USE_STD_MATH
    return Complex(acos(x) * cosh(y), atan2(sinh(y), x));
  }
  Complex
  atan() const
  {
    USE_STD_MATH
    return Complex(atan(x) * cosh(y), atan2(sinh(y), x));
  }
  Complex
  asinh() const
  {
    USE_STD_MATH
    return Complex(asinh(x) * cos(y), atan2(sinh(y), x));
  }
  Complex
  acosh() const
  {
    USE_STD_MATH
    return Complex(acosh(x) * cos(y), atan2(sinh(y), x));
  }
  Complex
  atanh() const
  {
    USE_STD_MATH
    return Complex(atanh(x) * cos(y), atan2(sinh(y), x));
  }
  Complex
  abs() const
  {
    using std::abs;
    return Complex(abs(x), abs(y));
  }

  // Power functions
  Complex
  pow(const Complex &other) const
  {
    USE_STD_MATH
    return Complex(pow(mag(), other.x) * cos(other.y * arg()),
                   pow(mag(), other.x) * sin(other.y * arg()));
  }
  Complex
  pow(const Real &other) const
  {
    USE_STD_MATH
    return Complex(pow(mag(), other) * cos(other * arg()),
                   pow(mag(), other) * sin(other * arg()));
  }

  // Comparison operators
  // bool operator==(const Complex &other) const {
  //   return (x == other.x && y == other.y);
  // }
  // bool operator!=(const Complex &other) const { return !(*this == other); }
};

// Maybe do this without parenthesis and commas?
template <typename OStream, typename Real>
inline OStream &
operator<<(OStream &os, const Complex<Real> &c)
{
  os << "(" << c.x << ", " << c.y << ")";
  return os;
}

template <typename IStream, typename Real>
inline IStream &
operator>>(IStream &is, Complex<Real> &c)
{
  char ch;
  is >> ch;  // Read the opening parenthesis
  is >> c.x; // Read the x value
  is >> ch;  // Read the comma
  is >> c.y; // Read the y value
  is >> ch;  // Read the closing parenthesis
  return is;
}

template <RealTypeName LReal, typename RReal>
inline Complex<typename Sum<LReal, RReal>::type>
operator+(const LReal &scalar, const Complex<RReal> &c)
{
  return Complex(scalar + c.x, c.y);
}
template <RealTypeName LReal, typename RReal>
inline Complex<typename Sum<LReal, RReal>::type>
operator-(const LReal &scalar, const Complex<RReal> &c)
{
  return Complex(scalar - c.x, -c.y);
}
template <RealTypeName LReal, typename RReal>
inline Complex<typename Prod<LReal, RReal>::type>
operator*(const LReal &scalar, const Complex<RReal> &c)
{
  return Complex(scalar * c.x, scalar * c.y);
}
template <RealTypeName LReal, typename RReal>
inline Complex<typename Prod<LReal, RReal>::type>
operator/(const LReal &scalar, const Complex<RReal> &c)
{
  LReal denom = c.x * c.x + c.y * c.y;
  return Complex((scalar * c.x) / denom, (-scalar * c.y) / denom);
}
template <typename Real>
inline Real
real(const Complex<Real> &c)
{
  return c.real();
}
template <typename Real>
inline Real
imag(const Complex<Real> &c)
{
  return c.imag();
}
template <typename Real>
inline Complex<Real>
conj(const Complex<Real> &c)
{
  return c.conjugate();
}
template <typename Real>
inline Real
mag(const Complex<Real> &c)
{
  return c.mag();
}
template <typename Real>
inline Real
mag2(const Complex<Real> &c)
{
  return c.mag2();
}
template <typename Real>
inline Real
arg(const Complex<Real> &c)
{
  return c.arg();
}
template <typename Real>
inline Complex<Real>
exp(const Complex<Real> &c)
{
  return c.exp();
}
template <typename Real>
inline Complex<Real>
log(const Complex<Real> &c)
{
  return c.log();
}
template <typename Real>
inline Complex<Real>
sqrt(const Complex<Real> &c)
{
  return c.sqrt();
}
template <typename Real>
inline Complex<Real>
sin(const Complex<Real> &c)
{
  return c.sin();
}
template <typename Real>
inline Complex<Real>
cos(const Complex<Real> &c)
{
  return c.cos();
}
template <typename Real>
inline Complex<Real>
tan(const Complex<Real> &c)
{
  return c.tan();
}
template <typename Real>
inline Complex<Real>
sinh(const Complex<Real> &c)
{
  return c.sinh();
}
template <typename Real>
inline Complex<Real>
cosh(const Complex<Real> &c)
{
  return c.cosh();
}
template <typename Real>
inline Complex<Real>
tanh(const Complex<Real> &c)
{
  return c.tanh();
}
template <typename Real>
inline Complex<Real>
asin(const Complex<Real> &c)
{
  return c.asin();
}
template <typename Real>
inline Complex<Real>
acos(const Complex<Real> &c)
{
  return c.acos();
}
template <typename Real>
inline Complex<Real>
atan(const Complex<Real> &c)
{
  return c.atan();
}
template <typename Real>
inline Complex<Real>
asinh(const Complex<Real> &c)
{
  return c.asinh();
}
template <typename Real>
inline Complex<Real>
acosh(const Complex<Real> &c)
{
  return c.acosh();
}
template <typename Real>
inline Complex<Real>
atanh(const Complex<Real> &c)
{
  return c.atanh();
}
template <typename Real>
inline Complex<Real>
abs(const Complex<Real> &c)
{
  return c.abs();
}
// Coordinate conversion functions
template <typename Real>
Complex<Real>
polar(const Real &r, const Real &theta)
{
  return Complex<Real>(r * cos(theta), r * sin(theta));
}
template <typename Real>
Complex<Real>
cartesian(const Real &r, const Real &theta)
{
  return Complex<Real>(r * cos(theta), r * sin(theta));
}

#include <deal.II/base/numbers.h>
#include <deal.II/base/vectorization.h>

template <typename number>
struct dealii::EnableIfScalar<Complex<number>>
{
  using type = Complex<number>;
};

template <typename Number>
struct is_real<dealii::VectorizedArray<Number>> : std::true_type
{};