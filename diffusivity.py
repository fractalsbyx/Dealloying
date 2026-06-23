"""This is vibe-coded and produces untrustworthy results. DO NOT use in production."""


"""
Cr diffusivity in Ni and Ni-Cr alloys
Based on: Gheno & Martinelli (2018), Materialia 3, 145-152
"Tracer diffusion of Cr in Ni and Ni-22Cr studied by SIMS"

Provides:
  - Tracer diffusion coefficient of Cr in Ni:       D*_Cr^Ni(T)
  - Tracer diffusion coefficient of Cr in Ni-22Cr:  D*_Cr^(Ni-22Cr)(T)
  - Tracer diffusion coefficient of Ni in Ni:        D*_Ni (self-diffusion, from Campbell & Rukhin 2011)
  - Interdiffusion coefficient D~ in Ni-Cr (Darken + Manning + thermodynamic factor)
    using Jonsson (1995) mobility assessment (extrapolated) and Lee (1992) thermodynamics.

Notes on the interdiffusion model
----------------------------------
The paper (Appendix B) uses the Darken-Manning expression:
    D~ = (x_Ni * D*_Cr + x_Cr * D*_Ni) * S * phi

where:
  S   = Manning vacancy-wind factor
  phi = thermodynamic factor  x_Cr/(RT) * d(mu_Cr)/d(x_Cr)

For phi, the paper cites Lee (1992) CALPHAD thermodynamics for the fcc Ni-Cr solution.
A simplified but representative Redlich-Kister fit of the Lee thermodynamic factor
is used here (valid ~500-1000 °C, 0-25 at.% Cr, as in the paper's Fig. B1).

Temperature range validated by experiment: 542-843 °C (815-1116 K)
Composition validated: pure Ni and Ni-22at.%Cr
"""

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
R = 8.314  # J/(mol·K)

# ── Mobility parameters from Table/Fig. 6 of Gheno & Martinelli 2018 ─────────
# Arrhenius form:  D*(T) = D0 * exp(-Q / (R*T))
# Parameters reported as ln(D0) [cm²/s] and Q [kJ/mol]

LN_D0_Cr_in_Ni       = -1.6    # cm²/s   ±0.3
Q_Cr_in_Ni           = 260e3   # J/mol   ±2 kJ/mol

LN_D0_Cr_in_Ni22Cr   = -0.3    # cm²/s   ±1.3
Q_Cr_in_Ni22Cr       = 279e3   # J/mol   ±10 kJ/mol

# Ni self-diffusion: Campbell & Rukhin (2011) recommended values
# ln(D0_Ni) = -3.02 cm²/s,  Q_Ni = 279.7 kJ/mol  (consistent with Jonsson 1995)
LN_D0_Ni_self        = -3.02   # cm²/s
Q_Ni_self            = 279.7e3 # J/mol

# f0 for fcc lattice (correlation factor)
F0_FCC = 0.7815


# ── Tracer diffusion coefficients ─────────────────────────────────────────────

def tracer_Cr_in_Ni(T_C):
    """
    Tracer diffusion coefficient of Cr in pure Ni.

    Parameters
    ----------
    T_C : float or array-like
        Temperature in °C. Valid range: ~542-843 °C (paper) / up to ~1100 °C
        (consistent with literature).

    Returns
    -------
    D : ndarray  [cm²/s]
    """
    T_C = np.asarray(T_C, dtype=float)
    T_K = T_C + 273.15
    return np.exp(LN_D0_Cr_in_Ni) * np.exp(-Q_Cr_in_Ni / (R * T_K))


def tracer_Cr_in_Ni22Cr(T_C):
    """
    Tracer diffusion coefficient of Cr in Ni-22 at.% Cr.

    Parameters
    ----------
    T_C : float or array-like
        Temperature in °C.

    Returns
    -------
    D : ndarray  [cm²/s]
    """
    T_C = np.asarray(T_C, dtype=float)
    T_K = T_C + 273.15
    return np.exp(LN_D0_Cr_in_Ni22Cr) * np.exp(-Q_Cr_in_Ni22Cr / (R * T_K))


def tracer_Ni_self(T_C):
    """
    Ni self-diffusion coefficient (Campbell & Rukhin 2011 / Jonsson 1995).

    Parameters
    ----------
    T_C : float or array-like
        Temperature in °C.

    Returns
    -------
    D : ndarray  [cm²/s]
    """
    T_C = np.asarray(T_C, dtype=float)
    T_K = T_C + 273.15
    return np.exp(LN_D0_Ni_self) * np.exp(-Q_Ni_self / (R * T_K))


def tracer_Cr(T_C, x_Cr):
    """
    Interpolated tracer diffusion coefficient of Cr in Ni-Cr fcc solid solution.

    Uses a linear interpolation in x_Cr between the two measured end-points
    (pure Ni and Ni-22at.%Cr) as a simple but physically motivated estimate.
    Valid for 0 ≤ x_Cr ≤ 0.22; extrapolation beyond 0.22 at your own risk.

    Parameters
    ----------
    T_C  : float or array-like   Temperature in °C
    x_Cr : float or array-like   Cr mole fraction (0-1)

    Returns
    -------
    D : ndarray  [cm²/s]
    """
    T_C  = np.asarray(T_C,  dtype=float)
    x_Cr = np.asarray(x_Cr, dtype=float)
    # Linear interpolation of ln(D) between x_Cr=0 and x_Cr=0.22
    ln_D_Ni     = np.log(tracer_Cr_in_Ni(T_C))
    ln_D_Ni22Cr = np.log(tracer_Cr_in_Ni22Cr(T_C))
    alpha = np.clip(x_Cr / 0.22, 0.0, 1.0)
    return np.exp(ln_D_Ni + alpha * (ln_D_Ni22Cr - ln_D_Ni))


# ── Thermodynamic factor phi (Lee 1992 via Jonsson 1995) ─────────────────────
#
# The regular-solution Gibbs excess is parameterised in Jonsson (1995) as:
#   G^ex = x_Ni * x_Cr * L(T)   with  L(T) = A + B*T  [J/mol]
#
# From Lee (1992) CALPHAD assessment for fcc Ni-Cr (Table 3 in Jonsson 1995):
#   L0 = 32600 - 8*T  [J/mol]   (0th order Redlich-Kister)
# (higher-order terms are small in the Ni-rich corner; omitted here)
#
# The thermodynamic factor is then:
#   phi = 1 + 2*x_Ni*x_Cr * L / (R*T)   for a simple regular solution
#       = 1 + 2*x_Ni*(1-x_Ni) * L / (R*T)

def thermodynamic_factor(T_C, x_Cr):
    """
    Thermodynamic factor phi = x_Cr/(RT) * d(mu_Cr)/d(x_Cr)
    using the Lee (1992) 0th-order Redlich-Kister parameter for fcc Ni-Cr.

    phi = 1 + 2*x_Ni*x_Cr * L0(T) / (R*T)

    Parameters
    ----------
    T_C  : float or array-like   Temperature in °C
    x_Cr : float or array-like   Cr mole fraction

    Returns
    -------
    phi : ndarray  (dimensionless)
    """
    T_C  = np.asarray(T_C,  dtype=float)
    x_Cr = np.asarray(x_Cr, dtype=float)
    T_K  = T_C + 273.15
    x_Ni = 1.0 - x_Cr
    L0   = 32600.0 - 8.0 * T_K   # J/mol  (Lee 1992 / Jonsson 1995)
    phi  = 1.0 + 2.0 * x_Ni * x_Cr * L0 / (R * T_K)
    return phi


# ── Manning vacancy-wind factor S ─────────────────────────────────────────────

def manning_S(D_Cr, D_Ni, x_Cr):
    """
    Manning vacancy-wind factor for a binary fcc alloy.

    S = 1 + (1-f0)/f0 * x_Ni*x_Cr*(D*_Ni - D*_Cr)^2 /
                        [(x_Ni*D*_Ni + x_Cr*D*_Cr) * (x_Ni*D*_Cr + x_Cr*D*_Ni)]

    Parameters
    ----------
    D_Cr, D_Ni : ndarray   tracer coefficients [any consistent units]
    x_Cr       : ndarray   Cr mole fraction

    Returns
    -------
    S : ndarray  (dimensionless)
    """
    x_Ni = 1.0 - x_Cr
    num  = x_Ni * x_Cr * (D_Ni - D_Cr)**2
    den  = (x_Ni * D_Ni + x_Cr * D_Cr) * (x_Ni * D_Cr + x_Cr * D_Ni)
    # Avoid division by zero at pure components
    with np.errstate(invalid='ignore', divide='ignore'):
        S = np.where(den > 0,
                     1.0 + (1.0 - F0_FCC) / F0_FCC * num / den,
                     1.0)
    return S


# ── Interdiffusion coefficient D~ ─────────────────────────────────────────────

def interdiffusion(T_C, x_Cr):
    """
    Interdiffusion coefficient D~ in Ni-Cr fcc solid solution.

    Uses the Darken-Manning expression:
        D~ = (x_Ni*D*_Cr + x_Cr*D*_Ni) * S * phi

    Parameters
    ----------
    T_C  : float or array-like   Temperature in °C
    x_Cr : float or array-like   Cr mole fraction (0-1)

    Returns
    -------
    D_tilde : ndarray  [cm²/s]
    """
    T_C  = np.asarray(T_C,  dtype=float)
    x_Cr = np.asarray(x_Cr, dtype=float)
    x_Ni = 1.0 - x_Cr

    D_Cr  = tracer_Cr(T_C, x_Cr)
    D_Ni  = tracer_Ni_self(T_C)
    S     = manning_S(D_Cr, D_Ni, x_Cr)
    phi   = thermodynamic_factor(T_C, x_Cr)

    D_tilde = (x_Ni * D_Cr + x_Cr * D_Ni) * S * phi
    return D_tilde


# ── Convenience: all quantities at once ───────────────────────────────────────

def all_diffusivities(T_C, x_Cr):
    """
    Return a dict with all diffusivities at given T and x_Cr.

    Parameters
    ----------
    T_C  : float   Temperature [°C]
    x_Cr : float   Cr mole fraction

    Returns
    -------
    dict with keys:
        D_Cr_tracer   [cm²/s]  Cr tracer diffusivity (interpolated)
        D_Ni_tracer   [cm²/s]  Ni self-diffusivity
        S             [-]      Manning vacancy-wind factor
        phi           [-]      Thermodynamic factor
        D_tilde       [cm²/s]  Interdiffusion coefficient
    """
    D_Cr   = float(tracer_Cr(T_C, x_Cr))
    D_Ni   = float(tracer_Ni_self(T_C))
    S      = float(manning_S(np.array(D_Cr), np.array(D_Ni), np.array(x_Cr)))
    phi    = float(thermodynamic_factor(T_C, x_Cr))
    D_til  = float(interdiffusion(T_C, x_Cr))
    return dict(D_Cr_tracer=D_Cr, D_Ni_tracer=D_Ni, S=S, phi=phi, D_tilde=D_til)


# ── Example / demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ── 1. Reproduce Arrhenius plot (Fig. 6 of the paper) ──────────────────
    T_exp_Ni     = np.array([843, 789, 748, 704, 668, 635, 600, 565, 542], dtype=float)
    D_exp_Ni     = np.array([1.4e-13, 3.1e-14, 9.9e-15, 1.7e-15, 8.7e-16,
                              1.9e-16, 4.7e-17, 1.5e-17, 3.7e-18])
    T_exp_22Cr   = np.array([789, 748, 704, 635, 565, 542], dtype=float)
    D_exp_22Cr   = np.array([1.6e-14, 4.0e-15, 6.8e-16, 7.7e-17, 3.9e-18, 9.3e-19])

    T_line = np.linspace(500, 900, 200)
    D_Ni_line     = tracer_Cr_in_Ni(T_line)
    D_22Cr_line   = tracer_Cr_in_Ni22Cr(T_line)
    inv_T_line    = 1e4 / (T_line + 273.15)   # 10^4/T [K^-1] for x-axis

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogy(1e4 / (T_exp_Ni   + 273.15), D_exp_Ni,   'o', color='steelblue',
                label='Cr/Ni (exp.)')
    ax.semilogy(1e4 / (T_exp_22Cr + 273.15), D_exp_22Cr, 's', color='tomato',
                label='Cr/Ni-22Cr (exp.)')
    ax.semilogy(inv_T_line, D_Ni_line,   '-', color='steelblue',
                label='Cr/Ni (fit)')
    ax.semilogy(inv_T_line, D_22Cr_line, '--', color='tomato',
                label='Cr/Ni-22Cr (fit)')
    ax.set_xlabel(r'$10^4 / T$ (K$^{-1}$)')
    ax.set_ylabel(r'$D^*_\mathrm{Cr}$ (cm² s$^{-1}$)')
    ax.set_title('Arrhenius plot - Cr tracer diffusivity\n(Gheno & Martinelli 2018)')
    ax.legend(fontsize=9)
    # x-axis: low T (high 1/T) on the left → do NOT invert; set limits explicitly
    # so that high temperature (small 1/T) is on the right
    ax.set_xlim(inv_T_line.max() * 1.02, inv_T_line.min() * 0.98)
    # secondary x-axis in °C (high T on right matches high-T side of primary axis)
    secax = ax.secondary_xaxis('top',
        functions=(lambda x: 1e4/x - 273.15, lambda T: 1e4/(T + 273.15)))
    secax.set_xlabel('Temperature (°C)')
    fig.tight_layout()
    fig.savefig('arrhenius_Cr_NiCr.png', dpi=150)
    print("Saved arrhenius_Cr_NiCr.png")

    # ── 2. Composition dependence at 500 and 1000 °C (Fig. B1 of the paper) ─
    #       Fig 2a is now split into two side-by-side subplots, one per temperature.
    x_Cr_arr = np.linspace(0, 0.22, 100)
    fig2, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    configs = [(500, 'steelblue', axes[0]), (1000, 'tomato', axes[1])]
    for T_val, color, ax2a in configs:
        D_Cr_arr  = tracer_Cr(T_val, x_Cr_arr)
        D_Ni_arr  = tracer_Ni_self(T_val) * np.ones_like(x_Cr_arr)
        D_til_arr = interdiffusion(T_val, x_Cr_arr)
        ax2a.plot(x_Cr_arr, D_Cr_arr,  '-',  color=color, label=r'$D^*_{Cr}$')
        ax2a.plot(x_Cr_arr, D_Ni_arr,  '--', color=color, label=r'$D^*_{Ni}$')
        ax2a.plot(x_Cr_arr, D_til_arr, ':',  color=color, label=r'$\tilde{D}$')
        ax2a.set_xlabel('$x_{Cr}$ (at. fraction)')
        ax2a.set_ylabel('Diffusivity (cm² s$^{-1}$)')
        ax2a.set_title(f'Tracer & interdiffusion\ncoefficients at {T_val}°C')
        ax2a.legend(fontsize=9)

    # Fig 2b: ratio D~/D*_Cr for both temperatures on a single panel
    for T_val, color in [(500, 'steelblue'), (1000, 'tomato')]:
        D_Cr_arr  = tracer_Cr(T_val, x_Cr_arr)
        D_til_arr = interdiffusion(T_val, x_Cr_arr)
        axes[2].plot(x_Cr_arr, D_til_arr / D_Cr_arr, color=color, label=f'{T_val}°C')
    axes[2].set_xlabel('$x_{Cr}$ (at. fraction)')
    axes[2].set_ylabel(r'$\tilde{D} / D^*_{Cr}$')
    axes[2].set_title(r'Ratio $\tilde{D}/D^*_{Cr}$')
    axes[2].legend(fontsize=9)

    fig2.tight_layout()
    fig2.savefig('composition_dependence_NiCr.png', dpi=150)
    print("Saved composition_dependence_NiCr.png")

    # ── 3. Tabulate key values ───────────────────────────────────────────────
    print("\n─── Single-point example ───────────────────────────────────────")
    for T_val, x_val in [(700, 0.0), (700, 0.22), (542, 0.0), (843, 0.10)]:
        res = all_diffusivities(T_val, x_val)
        print(f"T={T_val}°C  x_Cr={x_val:.2f} │ "
              f"D*_Cr={res['D_Cr_tracer']:.3e}  "
              f"D*_Ni={res['D_Ni_tracer']:.3e}  "
              f"S={res['S']:.4f}  phi={res['phi']:.4f}  "
              f"D~={res['D_tilde']:.3e}  [cm²/s]")

    plt.show()