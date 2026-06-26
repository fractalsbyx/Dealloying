#pragma once

#include <cmath>

// Thermodynamic functions for the Ni-Cr FCC system.
// Data from Tang & Hallstedt 2016:
// https://www.sciencedirect.com/science/article/pii/S0364591616301638

namespace NiCrThermo
{

static constexpr double R = 8.314;

// -----------------------------------------------------------------------------
// Pure-element reference Gibbs energies  [J/mol]
// -----------------------------------------------------------------------------

inline double
G0_Ni(double T)
{
  if (T < 1728.0)
    return -5179.159 + 117.854 * T - 22.096 * T * std::log(T) - 0.0048407 * T * T;
  else
    return -27840.62 + 279.134977 * T - 43.1 * T * std::log(T) + 1.12754e31 * std::pow(T, -9.0);
}

inline double
G0_Cr(double T)
{
  if (T < 2180.0)
    return -8856.94 + 157.48 * T - 26.908 * T * std::log(T) + 0.00189435 * T * T -
           1.47721e-6 * T * T * T + 139250.0 / T;
  else
    return -34869.344 + 344.18 * T - 50.0 * T * std::log(T) - 2.88526e32 * std::pow(T, -9.0);
}

// G of Cr in the metastable FCC structure
inline double
Gfcc_Cr(double T)
{
  return G0_Cr(T) + 7284.0 + 0.163 * T;
}

// -----------------------------------------------------------------------------
// Redlich-Kister interaction parameters for the FCC phase  [J/mol]
// -----------------------------------------------------------------------------

inline double
L_fcc(int i, double T)
{
  switch (i)
    {
    case 0:
      return 4300.0 - 8.9 * T;
    case 1:
      return 27000.0 - 13.8 * T;
    default:
      return 0.0;
    }
}

// -----------------------------------------------------------------------------
// Integral molar Gibbs energy of the FCC phase  [J/mol]
// -----------------------------------------------------------------------------

template <typename Real>
Real
G_fcc(double T, Real xNi, Real xCr)
{
  using std::log;
  Real Gxs = xNi * xCr * (L_fcc(0, T) + L_fcc(1, T) * (xCr - xNi));
  return G0_Ni(T) * xNi + Gfcc_Cr(T) * xCr + R * T * (xNi * log(xNi) + xCr * log(xCr)) + Gxs;
}
/*
// -----------------------------------------------------------------------------
// Chemical potentials  [J/mol]
// -----------------------------------------------------------------------------

template <typename Real>
Real
mu_Ni(double T, Real xNi, Real xCr)
{
  using std::log;
  Real mu_Gxs = L_fcc(0, T) * xCr + L_fcc(1, T) * xCr * (xCr - 2.0 * xNi);
  return G0_Ni(T) + R * T * (log(xNi) + 1.0) + mu_Gxs;
}

template <typename Real>
Real
mu_Cr(double T, Real xNi, Real xCr)
{
  using std::log;
  Real mu_Gxs = L_fcc(0, T) * xNi + L_fcc(1, T) * xNi * (2.0 * xCr - xNi);
  return Gfcc_Cr(T) + R * T * (log(xCr) + 1.0) + mu_Gxs;
}

// -----------------------------------------------------------------------------
// Partial derivatives of chemical potentials w.r.t. composition
// -----------------------------------------------------------------------------

template <typename Real>
Real
mu_Ni_dxNi(double T, Real xNi, Real xCr)
{
  return R * T / xNi + L_fcc(1, T) * (-2.0 * xCr);
}

template <typename Real>
Real
mu_Ni_dxCr(double T, Real xNi, Real xCr)
{
  return L_fcc(0, T) + L_fcc(1, T) * (2.0 * xCr - 2.0 * xNi);
}

template <typename Real>
Real
mu_Cr_dxNi(double T, Real xNi, Real xCr)
{
  return L_fcc(0, T) + L_fcc(1, T) * (2.0 * xCr - 2.0 * xNi);
}

template <typename Real>
Real
mu_Cr_dxCr(double T, Real xNi, Real xCr)
{
  return R * T / xCr + L_fcc(1, T) * (2.0 * xNi);
} */

// -----------------------------------------------------------------------------
// Isothermal helper — call set_temperature() once, then evaluate functions
// that take only xCr (xNi = 1 - xCr internally).
// -----------------------------------------------------------------------------

class Isothermal
{
public:
  void
  set_temperature(double T)
  {
    _T    = T;
    _RT   = R * T;
    _G0Ni = G0_Ni(T);
    _GCr  = Gfcc_Cr(T);
    _L0   = L_fcc(0, T);
    _L1   = L_fcc(1, T);
  }

  double
  temperature() const
  {
    return _T;
  }

  template <typename Real>
  Real
  G_fcc(Real xCr) const
  {
    using std::log;
    Real xNi = 1.0 - xCr;
    Real Gxs = xNi * xCr * (_L0 + _L1 * (xCr - xNi));
    return _G0Ni * xNi + _GCr * xCr + _RT * (xNi * log(xNi) + xCr * log(xCr)) + Gxs;
  }
  /*
      template <typename Real>
      Real mu_Ni(Real xCr) const
      {
          using std::log;
          Real xNi = 1.0 - xCr;
          Real mu_Gxs = _L0*xCr + _L1*xCr*(xCr - 2.0*xNi);
          return _G0Ni + _RT*(log(xNi) + 1.0) + mu_Gxs;
      }

      template <typename Real>
      Real mu_Cr(Real xCr) const
      {
          using std::log;
          Real xNi = 1.0 - xCr;
          Real mu_Gxs = _L0*xNi + _L1*xNi*(2.0*xCr - xNi);
          return _GCr + _RT*(log(xCr) + 1.0) + mu_Gxs;
      }

      template <typename Real>
      Real mu_Ni_dxNi(Real xCr) const
      {
          Real xNi = 1.0 - xCr;
          return _RT/xNi + _L1*(-2.0*xCr);
      }

      template <typename Real>
      Real mu_Ni_dxCr(Real xCr) const
      {
          Real xNi = 1.0 - xCr;
          return _L0 + _L1*(2.0*xCr - 2.0*xNi);
      }

      template <typename Real>
      Real mu_Cr_dxNi(Real xCr) const
      {
          Real xNi = 1.0 - xCr;
          return _L0 + _L1*(2.0*xCr - 2.0*xNi);
      }

      template <typename Real>
      Real mu_Cr_dxCr(Real xCr) const
      {
          Real xNi = 1.0 - xCr;
          return _RT/xCr + _L1*(2.0*xNi);
      }
   */
private:
  double _T    = 0.0;
  double _RT   = 0.0;
  double _G0Ni = 0.0;
  double _GCr  = 0.0;
  double _L0   = 0.0;
  double _L1   = 0.0;
};

} // namespace NiCrThermo