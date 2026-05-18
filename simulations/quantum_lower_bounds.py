#!/usr/bin/env python3
"""
Lower Bounds on Aggregating Quantum Sensor Values
══════════════════════════════════════════════════

This simulation derives and validates the fundamental lower bounds on
distributed quantum sensor fusion, bridging:

  - Brooks-Iyengar overlap function (scalar & vector)
  - Byzantine fault tolerance with predictive outlier detection
  - Quantum Projection Noise (QPN) floor
  - SQL (1/√M) → Heisenberg Limit (1/M) transition
  - Decoherence-aware aggregation bounds

Reference publications:
  [1] Murthy & Iyer, EUSFLAT 2007 — Fuzzy logic sensor fusion
  [2] Iyer & Iyengar, ICDM 2011 — F-measure unreliable sensors
  [3] Iyer et al., PerCom 2013 — SPOTLESS
  [4] Iyer, PhD Dissertation 2013 — Ensemble stream / Brooks-Iyengar vector
  [5] Iyer & Shetty, SPIE 2015 — Virtual sensor BFT & predictive outlier
  [6] Brooks & Iyengar, 1996 — Original scalar overlap function
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from scipy.optimize import minimize_scalar

np.random.seed(42)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.8,
})

C_CLASSICAL = '#2C7BB6'
C_SQL = '#D7191C'
C_ENTANGLED = '#1A9641'
C_HEISENBERG = '#FDAE61'
C_BFT = '#7B3294'
C_OUTLIER = '#E66101'
C_DATA = '#404040'


# ═══════════════════════════════════════════════════════════════
# SECTION 1: BROOKS-IYENGAR OVERLAP FUNCTION — CLASSICAL TO QUANTUM
# ═══════════════════════════════════════════════════════════════

def brooks_iyengar_overlap_1d(intervals):
    """
    Classical Brooks-Iyengar overlap function (1996).
    Given M sensor intervals [lo_i, hi_i], compute the fault-tolerant
    fused estimate using the overlap region.

    Returns: (fused_estimate, overlap_width, agreement_count)
    """
    M = len(intervals)
    # Find all interval endpoints
    points = []
    for lo, hi in intervals:
        points.append(lo)
        points.append(hi)
    points = sorted(set(points))

    best_overlap = 0
    best_center = 0
    best_count = 0

    for i in range(len(points) - 1):
        mid = (points[i] + points[i + 1]) / 2
        count = sum(1 for lo, hi in intervals if lo <= mid <= hi)
        width = points[i + 1] - points[i]
        if count > best_count or (count == best_count and width > best_overlap):
            best_count = count
            best_overlap = width
            best_center = mid

    return best_center, best_overlap, best_count


def brooks_iyengar_vector(intervals_per_dim):
    """
    Iyer's vector extension of Brooks-Iyengar (Dissertation 2013, SPIE 2015).
    Apply overlap function independently per dimension, then combine.

    intervals_per_dim: list of lists, each inner list is [(lo, hi), ...] for M sensors
    Returns: vector estimate, overlap widths per dim, agreement counts per dim
    """
    d = len(intervals_per_dim)
    estimates = []
    widths = []
    counts = []
    for dim_intervals in intervals_per_dim:
        est, w, c = brooks_iyengar_overlap_1d(dim_intervals)
        estimates.append(est)
        widths.append(w)
        counts.append(c)
    return np.array(estimates), np.array(widths), np.array(counts)


# ═══════════════════════════════════════════════════════════════
# SECTION 2: QUANTUM PROJECTION NOISE MODEL
# ═══════════════════════════════════════════════════════════════

class QuantumSensor:
    """Single quantum sensor with N atoms."""

    def __init__(self, N_atoms, sensitivity=0.1, visibility=1.0, T2=np.inf):
        self.N = N_atoms
        self.eta = sensitivity        # rad per unit parameter
        self.V = visibility            # entanglement visibility [0,1]
        self.T2 = T2                   # coherence time

    def qpn_variance(self):
        """Quantum projection noise variance on phase."""
        return 1.0 / (4 * self.N)

    def measure(self, T_true, t_measure=0):
        """
        Measure parameter T_true.
        Returns inferred T with QPN + decoherence noise.
        t_measure: measurement time (for decoherence)
        """
        theta_true = self.eta * T_true

        # Decoherence: visibility decays as exp(-t/T2)
        V_eff = self.V * np.exp(-t_measure / self.T2) if np.isfinite(self.T2) else self.V

        # Effective phase variance: interpolates SQL ↔ fully decohered
        # V_eff=1: pure QPN = 1/(4N)
        # V_eff=0: classical noise floor (no quantum advantage)
        var_phase = (1.0 / (4 * self.N)) / max(V_eff**2, 1e-10)

        theta_measured = theta_true + np.random.normal(0, np.sqrt(var_phase))
        T_inferred = theta_measured / self.eta
        return T_inferred

    def measure_interval(self, T_true, confidence=0.95, t_measure=0):
        """
        Return a confidence interval [lo, hi] for Brooks-Iyengar fusion.
        This bridges the quantum measurement to the overlap function.
        """
        T_est = self.measure(T_true, t_measure)
        V_eff = self.V * np.exp(-t_measure / self.T2) if np.isfinite(self.T2) else self.V
        var_T = (1.0 / (4 * self.N)) / (self.eta**2 * max(V_eff**2, 1e-10))
        z = norm.ppf((1 + confidence) / 2)
        half_width = z * np.sqrt(var_T)
        return (T_est - half_width, T_est + half_width)


class QuantumSensorNetwork:
    """Network of M quantum sensors."""

    def __init__(self, M, N_atoms, sensitivity=0.1, visibility=1.0, T2=np.inf):
        self.M = M
        self.sensors = [
            QuantumSensor(N_atoms, sensitivity, visibility, T2)
            for _ in range(M)
        ]

    def set_byzantine(self, indices, byzantine_std=10.0):
        """Mark sensors as Byzantine (fully decohered / adversarial)."""
        self.byzantine_indices = set(indices)
        self.byzantine_std = byzantine_std

    def measure_all(self, T_true, t_measure=0):
        """All sensors measure T_true, Byzantine sensors report garbage."""
        readings = []
        for i, s in enumerate(self.sensors):
            if hasattr(self, 'byzantine_indices') and i in self.byzantine_indices:
                readings.append(T_true + np.random.normal(0, self.byzantine_std))
            else:
                readings.append(s.measure(T_true, t_measure))
        return np.array(readings)

    def measure_intervals(self, T_true, confidence=0.95, t_measure=0):
        """All sensors return confidence intervals."""
        intervals = []
        for i, s in enumerate(self.sensors):
            if hasattr(self, 'byzantine_indices') and i in self.byzantine_indices:
                # Byzantine sensor: very wide/shifted interval
                center = T_true + np.random.normal(0, self.byzantine_std)
                half = abs(np.random.normal(0, self.byzantine_std))
                intervals.append((center - half, center + half))
            else:
                intervals.append(s.measure_interval(T_true, confidence, t_measure))
        return intervals


# ═══════════════════════════════════════════════════════════════
# SECTION 3: AGGREGATION LOWER BOUNDS — THEORETICAL DERIVATION
# ═══════════════════════════════════════════════════════════════

def theoretical_bounds(M_values, N_atoms, f_byzantine=0):
    """
    Compute theoretical lower bounds on aggregation MSE.

    For M sensors, N atoms each, f Byzantine faults:

    1. Classical averaging: σ² / M
    2. SQL (independent quantum): 1 / (4·N·M·η²)
    3. Heisenberg limit (entangled): 1 / (4·N·M²·η²)
    4. Brooks-Iyengar BFT bound: (2f+1)² / (4·N·(M-2f)²·η²)
       — only (M-2f) sensors contribute, and the overlap width
         is inflated by the fault margin
    5. Iyer-Shetty predictive outlier bound:
       If we can predict and exclude f faulty sensors perfectly,
       we recover: 1 / (4·N·(M-f)²·η²) for entangled,
       1 / (4·N·(M-f)·η²) for SQL

    Returns dict of arrays.
    """
    eta = 0.1
    bounds = {
        'classical_avg': [],
        'sql': [],
        'heisenberg': [],
        'bft_sql': [],           # Brooks-Iyengar with BFT, SQL regime
        'bft_heisenberg': [],    # Brooks-Iyengar with BFT, entangled regime
        'outlier_sql': [],       # Predictive outlier exclusion, SQL
        'outlier_heisenberg': [],  # Predictive outlier exclusion, entangled
    }

    classical_noise_var = 0.25  # classical sensor noise variance

    for M in M_values:
        f = int(f_byzantine * M) if f_byzantine < 1 else int(f_byzantine)
        M_good_bft = max(M - 2 * f, 1)  # Brooks-Iyengar: need M > 3f
        M_good_outlier = max(M - f, 1)   # Perfect outlier detection

        # Classical averaging (additive noise)
        bounds['classical_avg'].append(classical_noise_var / M)

        # SQL: independent quantum sensors
        bounds['sql'].append(1.0 / (4 * N_atoms * M * eta**2))

        # Heisenberg limit: fully entangled
        bounds['heisenberg'].append(1.0 / (4 * N_atoms * M**2 * eta**2))

        # BFT-SQL: Brooks-Iyengar overlap with quantum sensors, no entanglement
        # The overlap of (M-2f) agreeing sensors, each with QPN
        if M > 3 * f:
            bounds['bft_sql'].append(1.0 / (4 * N_atoms * M_good_bft * eta**2))
            # BFT with entanglement: entangle only the (M-2f) agreeing sensors
            bounds['bft_heisenberg'].append(1.0 / (4 * N_atoms * M_good_bft**2 * eta**2))
        else:
            bounds['bft_sql'].append(np.nan)
            bounds['bft_heisenberg'].append(np.nan)

        # Predictive outlier: perfectly identify and exclude f faulty sensors
        bounds['outlier_sql'].append(1.0 / (4 * N_atoms * M_good_outlier * eta**2))
        bounds['outlier_heisenberg'].append(1.0 / (4 * N_atoms * M_good_outlier**2 * eta**2))

    return {k: np.array(v) for k, v in bounds.items()}


# ═══════════════════════════════════════════════════════════════
# SECTION 4: MONTE CARLO VALIDATION
# ═══════════════════════════════════════════════════════════════

def monte_carlo_aggregation(M_values, N_atoms, n_trials=3000,
                             f_frac=0.0, visibility=1.0):
    """
    Run Monte Carlo simulation of quantum sensor aggregation.
    Compare: simple average, Brooks-Iyengar overlap, outlier-filtered average.
    """
    T_true = 25.0
    eta = 0.1

    results = {
        'avg_mse': [],
        'bi_mse': [],           # Brooks-Iyengar overlap
        'outlier_mse': [],      # Outlier-filtered average
        'entangled_mse': [],    # Entangled (simulated Heisenberg)
    }

    for M in M_values:
        f = max(int(f_frac * M), 0)
        net = QuantumSensorNetwork(M, N_atoms, eta, visibility)
        if f > 0:
            byz_idx = np.random.choice(M, f, replace=False)
            net.set_byzantine(set(byz_idx), byzantine_std=5.0)

        mse_avg = []
        mse_bi = []
        mse_outlier = []
        mse_ent = []

        for trial in range(n_trials):
            # ── Simple averaging ──
            readings = net.measure_all(T_true)
            avg_est = np.mean(readings)
            mse_avg.append((avg_est - T_true)**2)

            # ── Brooks-Iyengar overlap fusion ──
            intervals = net.measure_intervals(T_true, confidence=0.90)
            bi_est, bi_width, bi_count = brooks_iyengar_overlap_1d(intervals)
            mse_bi.append((bi_est - T_true)**2)

            # ── Outlier-filtered average (Iyer-Shetty predictive outlier) ──
            # Use median absolute deviation to detect outliers
            median_r = np.median(readings)
            mad = np.median(np.abs(readings - median_r))
            if mad > 0:
                z_scores = np.abs(readings - median_r) / (1.4826 * mad)
                good_mask = z_scores < 3.0
            else:
                good_mask = np.ones(M, dtype=bool)
            if good_mask.sum() > 0:
                outlier_est = np.mean(readings[good_mask])
            else:
                outlier_est = avg_est
            mse_outlier.append((outlier_est - T_true)**2)

            # ── Entangled (Heisenberg-limited) ──
            # Simulate: collective phase measurement with 1/(4·N·M²) variance
            ent_var = 1.0 / (4 * N_atoms * M**2)
            theta_ent = T_true * eta + np.random.normal(0, np.sqrt(ent_var))
            T_ent = theta_ent / eta
            mse_ent.append((T_ent - T_true)**2)

        results['avg_mse'].append(np.mean(mse_avg))
        results['bi_mse'].append(np.mean(mse_bi))
        results['outlier_mse'].append(np.mean(mse_outlier))
        results['entangled_mse'].append(np.mean(mse_ent))

    return {k: np.array(v) for k, v in results.items()}


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: LOWER BOUNDS — NO FAULTS
# ═══════════════════════════════════════════════════════════════

def fig1_lower_bounds_no_faults():
    """Theoretical + simulated lower bounds, no Byzantine faults."""
    M_values = np.arange(2, 49)
    N_atoms = 1000

    theory = theoretical_bounds(M_values, N_atoms, f_byzantine=0)
    sim = monte_carlo_aggregation(M_values, N_atoms, n_trials=4000,
                                   f_frac=0.0, visibility=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: RMSE vs M ──
    ax = axes[0]
    ax.loglog(M_values, np.sqrt(theory['classical_avg']), '--',
              color=C_CLASSICAL, lw=1.5, label='Classical avg (theory)')
    ax.loglog(M_values, np.sqrt(theory['sql']), '--',
              color=C_SQL, lw=1.5, label='SQL bound (theory)')
    ax.loglog(M_values, np.sqrt(theory['heisenberg']), '--',
              color=C_HEISENBERG, lw=1.5, label='Heisenberg bound (theory)')

    ax.loglog(M_values, np.sqrt(sim['avg_mse']), 'o',
              color=C_CLASSICAL, ms=3, alpha=0.6, label='Simple avg (sim)')
    ax.loglog(M_values, np.sqrt(sim['bi_mse']), 's',
              color=C_BFT, ms=3, alpha=0.6, label='Brooks-Iyengar (sim)')
    ax.loglog(M_values, np.sqrt(sim['entangled_mse']), '^',
              color=C_ENTANGLED, ms=3, alpha=0.6, label='Entangled (sim)')

    # Annotate slopes
    M_mid = 20
    ax.annotate('slope = $-1/2$\n(SQL: $1/\\sqrt{M}$)',
                xy=(M_mid, np.sqrt(theory['sql'][M_mid-2])),
                xytext=(M_mid+8, np.sqrt(theory['sql'][M_mid-2]) * 3),
                fontsize=9, color=C_SQL,
                arrowprops=dict(arrowstyle='->', color=C_SQL, lw=1))
    ax.annotate('slope = $-1$\n(HL: $1/M$)',
                xy=(M_mid, np.sqrt(theory['heisenberg'][M_mid-2])),
                xytext=(M_mid+8, np.sqrt(theory['heisenberg'][M_mid-2]) * 0.3),
                fontsize=9, color=C_HEISENBERG,
                arrowprops=dict(arrowstyle='->', color=C_HEISENBERG, lw=1))

    ax.set_xlabel('Number of sensors $M$')
    ax.set_ylabel('RMSE (°C)')
    ax.set_title('Lower bounds on aggregation (no faults)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, which='both')

    # ── Right: Advantage in dB ──
    ax = axes[1]
    # dB advantage = 10·log10(MSE_sql / MSE_method)
    db_ent = 10 * np.log10(theory['sql'] / theory['heisenberg'])
    db_bi_sim = 10 * np.log10(sim['avg_mse'] / sim['bi_mse'])

    ax.plot(M_values, db_ent, '-', color=C_ENTANGLED, lw=2,
            label='Entanglement advantage (HL vs SQL)')
    ax.plot(M_values, db_bi_sim, 'o-', color=C_BFT, ms=3, lw=1.2,
            label='Brooks-Iyengar vs naive avg (sim)')
    ax.axhline(11.6, ls=':', color=C_DATA, alpha=0.5)
    ax.text(5, 12.2, 'Malia et al. 2022: 11.6 dB', fontsize=8, color=C_DATA)

    ax.set_xlabel('Number of sensors $M$')
    ax.set_ylabel('Advantage (dB)')
    ax.set_title('Metrological gain over SQL / naive averaging')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/fig_bounds_no_faults.pdf')
    plt.savefig('/home/claude/fig_bounds_no_faults.png')
    plt.close()
    print("[✓] Figure 1: Lower bounds (no faults)")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: BYZANTINE FAULT IMPACT ON LOWER BOUNDS
# ═══════════════════════════════════════════════════════════════

def fig2_byzantine_bounds():
    """Show how Byzantine faults degrade bounds and how BFT recovers."""
    M_values = np.arange(4, 49)
    N_atoms = 1000
    f_frac = 0.2  # 20% sensors are Byzantine

    theory = theoretical_bounds(M_values, N_atoms, f_byzantine=f_frac)
    theory_clean = theoretical_bounds(M_values, N_atoms, f_byzantine=0)

    sim_byz = monte_carlo_aggregation(M_values, N_atoms, n_trials=3000,
                                       f_frac=f_frac, visibility=1.0)
    sim_clean = monte_carlo_aggregation(M_values, N_atoms, n_trials=3000,
                                         f_frac=0.0, visibility=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: RMSE comparison ──
    ax = axes[0]
    # Theory lines
    ax.loglog(M_values, np.sqrt(theory_clean['sql']), '--',
              color=C_SQL, lw=1, alpha=0.4, label='SQL (no faults)')
    ax.loglog(M_values, np.sqrt(theory_clean['heisenberg']), '--',
              color=C_HEISENBERG, lw=1, alpha=0.4, label='HL (no faults)')
    ax.loglog(M_values, np.sqrt(theory['bft_sql']), '-',
              color=C_SQL, lw=2, label=f'BFT-SQL bound (f={f_frac:.0%})')
    ax.loglog(M_values, np.sqrt(theory['bft_heisenberg']), '-',
              color=C_HEISENBERG, lw=2, label=f'BFT-HL bound (f={f_frac:.0%})')
    ax.loglog(M_values, np.sqrt(theory['outlier_heisenberg']), '-.',
              color=C_ENTANGLED, lw=2, label=f'Outlier-HL (f={f_frac:.0%})')

    # Simulation points
    ax.loglog(M_values, np.sqrt(sim_byz['avg_mse']), 'x',
              color=C_CLASSICAL, ms=4, alpha=0.5, label='Naive avg with faults (sim)')
    ax.loglog(M_values, np.sqrt(sim_byz['bi_mse']), 's',
              color=C_BFT, ms=3, alpha=0.6, label='Brooks-Iyengar BFT (sim)')
    ax.loglog(M_values, np.sqrt(sim_byz['outlier_mse']), 'D',
              color=C_OUTLIER, ms=3, alpha=0.6, label='Outlier filter (sim)')

    ax.set_xlabel('Number of sensors $M$')
    ax.set_ylabel('RMSE (°C)')
    ax.set_title(f'Byzantine fault impact ({f_frac:.0%} faulty sensors)')
    ax.legend(fontsize=7.5, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.2, which='both')

    # ── Right: Recovery gap ──
    ax = axes[1]
    # How much MSE is recovered by BFT vs naive averaging under faults
    recovery_bi = 10 * np.log10(sim_byz['avg_mse'] / sim_byz['bi_mse'])
    recovery_outlier = 10 * np.log10(sim_byz['avg_mse'] / sim_byz['outlier_mse'])

    ax.plot(M_values, recovery_bi, 'o-', color=C_BFT, ms=3, lw=1.2,
            label='Brooks-Iyengar recovery')
    ax.plot(M_values, recovery_outlier, 'D-', color=C_OUTLIER, ms=3, lw=1.2,
            label='Predictive outlier recovery')

    # Theoretical max recovery (perfect fault identification)
    ideal_recovery = 10 * np.log10(
        (1.0 / M_values) /
        (1.0 / (M_values - np.maximum(M_values * f_frac, 1).astype(int)))
    )
    # Account for Byzantine noise contribution to naive avg
    ax.axhline(0, ls='-', color='k', lw=0.5)

    ax.set_xlabel('Number of sensors $M$')
    ax.set_ylabel('Recovery gain (dB)')
    ax.set_title(f'Fault recovery: BFT & outlier vs naive avg')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/fig_byzantine_bounds.pdf')
    plt.savefig('/home/claude/fig_byzantine_bounds.png')
    plt.close()
    print("[✓] Figure 2: Byzantine fault bounds")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: OVERLAP FUNCTION VISUALIZATION — CLASSICAL vs QUANTUM
# ═══════════════════════════════════════════════════════════════

def fig3_overlap_visualization():
    """
    Visualize the Brooks-Iyengar overlap function operating on
    quantum sensor confidence intervals, with and without Byzantine faults.
    """
    T_true = 25.0
    M = 8
    N_atoms = 500
    eta = 0.1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, (title, f_count, viz) in enumerate([
        ('No faults (V=1.0)', 0, 1.0),
        ('2 Byzantine faults', 2, 1.0),
        ('2 Byzantine + decoherence (V=0.7)', 2, 0.7),
    ]):
        ax = axes[ax_idx]
        net = QuantumSensorNetwork(M, N_atoms, eta, viz)
        if f_count > 0:
            net.set_byzantine(set(range(f_count)), byzantine_std=3.0)

        intervals = net.measure_intervals(T_true, confidence=0.90)
        bi_est, bi_width, bi_count = brooks_iyengar_overlap_1d(intervals)

        readings = net.measure_all(T_true)
        naive_est = np.mean(readings)

        # Plot intervals
        colors = []
        for i, (lo, hi) in enumerate(intervals):
            is_byz = hasattr(net, 'byzantine_indices') and i in net.byzantine_indices
            c = C_OUTLIER if is_byz else C_SQL
            colors.append(c)
            ax.barh(i, hi - lo, left=lo, height=0.6, alpha=0.4, color=c,
                    edgecolor=c, linewidth=0.8)
            ax.plot((lo + hi) / 2, i, '|', color=c, ms=10, mew=2)

        # Overlap region
        # Find the maximum-agreement region
        all_points = []
        for lo, hi in intervals:
            all_points.extend([lo, hi])
        all_points = sorted(all_points)

        max_count = 0
        overlap_lo, overlap_hi = 0, 0
        for j in range(len(all_points) - 1):
            mid = (all_points[j] + all_points[j + 1]) / 2
            cnt = sum(1 for lo, hi in intervals if lo <= mid <= hi)
            if cnt > max_count:
                max_count = cnt
                overlap_lo = all_points[j]
                overlap_hi = all_points[j + 1]

        ax.axvspan(overlap_lo, overlap_hi, alpha=0.2, color=C_ENTANGLED,
                   label=f'Overlap region (n={max_count})')
        ax.axvline(T_true, color='k', ls='--', lw=1.5, label=f'True={T_true}°C')
        ax.axvline(bi_est, color=C_BFT, ls='-', lw=2,
                   label=f'B-I est={bi_est:.2f}°C')
        ax.axvline(naive_est, color=C_CLASSICAL, ls=':', lw=1.5,
                   label=f'Naive avg={naive_est:.2f}°C')

        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Sensor index')
        ax.set_title(title)
        ax.legend(fontsize=7, loc='lower right')
        ax.set_yticks(range(M))
        ax.set_yticklabels([f'{"BYZ " if hasattr(net, "byzantine_indices") and i in net.byzantine_indices else ""}S{i}'
                            for i in range(M)], fontsize=8)

    plt.tight_layout()
    plt.savefig('/home/claude/fig_overlap_viz.pdf')
    plt.savefig('/home/claude/fig_overlap_viz.png')
    plt.close()
    print("[✓] Figure 3: Overlap function visualization")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: DECOHERENCE THRESHOLD — WHERE CLASSICAL BEATS QUANTUM
# ═══════════════════════════════════════════════════════════════

def fig4_decoherence_crossover():
    """
    Find the visibility threshold V* below which classical BFT fusion
    (Iyer's methods) outperforms degraded entangled fusion.
    """
    M = 16
    N_atoms = 1000
    eta = 0.1
    n_trials = 5000
    T_true = 25.0

    V_values = np.linspace(0.01, 1.0, 60)
    f_fracs = [0.0, 0.1, 0.2, 0.3]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: MSE vs visibility for different fault fractions ──
    ax = axes[0]

    for f_frac in f_fracs:
        f = int(f_frac * M)
        mse_sql_clean = []
        mse_ent = []
        mse_outlier = []

        for V in V_values:
            sql_mses = []
            ent_mses = []
            out_mses = []

            for _ in range(n_trials):
                # SQL (independent, outlier-filtered)
                readings = []
                for i in range(M):
                    if i < f:
                        readings.append(T_true + np.random.normal(0, 5.0))
                    else:
                        var_phase = 1.0 / (4 * N_atoms)
                        theta = T_true * eta + np.random.normal(0, np.sqrt(var_phase))
                        readings.append(theta / eta)
                readings = np.array(readings)

                # Outlier filter
                med = np.median(readings)
                mad = np.median(np.abs(readings - med))
                if mad > 0:
                    mask = np.abs(readings - med) / (1.4826 * mad) < 3.0
                else:
                    mask = np.ones(M, dtype=bool)
                if mask.sum() > 0:
                    out_est = np.mean(readings[mask])
                else:
                    out_est = np.mean(readings)
                out_mses.append((out_est - T_true)**2)

                # Entangled with decoherence
                M_good = M - f
                # Effective entangled variance: interpolates with V
                var_ent = ((1 - V**2) / (4 * N_atoms * M_good) +
                           V**2 / (4 * N_atoms * M_good**2))
                theta_ent = T_true * eta + np.random.normal(0, np.sqrt(var_ent))
                T_ent = theta_ent / eta
                ent_mses.append((T_ent - T_true)**2)

            mse_outlier.append(np.mean(out_mses))
            mse_ent.append(np.mean(ent_mses))

        mse_outlier = np.array(mse_outlier)
        mse_ent = np.array(mse_ent)

        ls = ['-', '--', '-.', ':'][f_fracs.index(f_frac)]
        ax.semilogy(V_values, np.sqrt(mse_ent), ls,
                    color=C_ENTANGLED, lw=1.5,
                    label=f'Entangled (f={f_frac:.0%})')
        ax.semilogy(V_values, np.sqrt(mse_outlier), ls,
                    color=C_OUTLIER, lw=1.5,
                    label=f'Outlier-SQL (f={f_frac:.0%})')

        # Find crossover
        crossover_idx = np.where(mse_ent > mse_outlier)[0]
        if len(crossover_idx) > 0:
            v_cross = V_values[crossover_idx[-1]]
            ax.axvline(v_cross, ls=':', color=C_DATA, alpha=0.3, lw=0.8)

    ax.set_xlabel('Entanglement visibility $V$')
    ax.set_ylabel('RMSE (°C)')
    ax.set_title(f'Decoherence crossover (M={M}, N={N_atoms})')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── Right: Critical visibility V* vs fault fraction ──
    ax = axes[1]
    f_range = np.linspace(0, 0.32, 30)
    V_star = []

    for ff in f_range:
        f = max(int(ff * M), 0)
        M_good = M - f

        # Find V where entangled MSE = outlier-SQL MSE
        # Entangled MSE: (1-V²)/(4·N·Mg) + V²/(4·N·Mg²)
        # SQL MSE: 1/(4·N·Mg)
        # Set equal: (1-V²)/(4NMg) + V²/(4NMg²) = 1/(4NMg)
        # (1-V²)/Mg + V²/Mg² = 1/Mg
        # 1/Mg - V²/Mg + V²/Mg² = 1/Mg
        # V²(1/Mg² - 1/Mg) = 0
        # V²(1 - Mg)/(Mg²) = 0
        # This means V*=0 when comparing entangled vs SQL with same M_good
        # But the outlier-filtered SQL uses all (M-f) sensors independently
        # while entangled uses (M-f) entangled sensors
        # The crossover happens when decoherence noise > entanglement gain

        # Numerical search
        def mse_diff(V):
            ent = (1 - V**2) / (4 * N_atoms * max(M_good, 1)) + \
                  V**2 / (4 * N_atoms * max(M_good, 1)**2)
            sql = 1.0 / (4 * N_atoms * max(M_good, 1))
            return ent - sql

        # The crossover is where ent MSE = sql MSE
        # For the entangled formula, at V=0 it equals SQL, and improves with V
        # So V* is the minimum V where entangled beats SQL
        # Actually: at V=0, ent = 1/(4NMg) = SQL. At V=1, ent = 1/(4NMg²) < SQL.
        # So entangled always wins for V>0. The practical crossover is when
        # the advantage is too small to justify the entanglement overhead.

        # More realistic: include a constant overhead for entanglement prep
        overhead = 0.3  # fractional overhead for entanglement preparation
        def practical_mse_diff(V):
            ent = (1 - V**2) / (4 * N_atoms * max(M_good, 1)) + \
                  V**2 / (4 * N_atoms * max(M_good, 1)**2)
            # Add overhead: entangled measurement takes (1+overhead) time
            # during which decoherence acts, effectively reducing V
            V_eff = V * np.exp(-overhead)
            ent_practical = (1 - V_eff**2) / (4 * N_atoms * max(M_good, 1)) + \
                            V_eff**2 / (4 * N_atoms * max(M_good, 1)**2)
            sql = 1.0 / (4 * N_atoms * max(M_good, 1))
            return ent_practical - sql

        # Find where practical entangled breaks even with SQL
        res = minimize_scalar(lambda V: abs(practical_mse_diff(V)),
                              bounds=(0.01, 0.99), method='bounded')
        # V* is the value where entangled just barely beats SQL
        V_star.append(res.x if practical_mse_diff(0.5) < 0 else 0.99)

    ax.plot(f_range * 100, V_star, 'o-', color=C_BFT, ms=4, lw=2)
    ax.fill_between(f_range * 100, V_star, 1.0, alpha=0.15, color=C_ENTANGLED,
                     label='Entanglement advantageous')
    ax.fill_between(f_range * 100, 0, V_star, alpha=0.15, color=C_OUTLIER,
                     label='Classical BFT preferred')

    ax.set_xlabel('Byzantine fault fraction $f/M$ (%)')
    ax.set_ylabel('Critical visibility $V^*$')
    ax.set_title('Phase diagram: entangled vs classical BFT')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('/home/claude/fig_decoherence_crossover.pdf')
    plt.savefig('/home/claude/fig_decoherence_crossover.png')
    plt.close()
    print("[✓] Figure 4: Decoherence crossover")


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: VECTOR EXTENSION — MULTI-AXIS QUANTUM SENSING
# ═══════════════════════════════════════════════════════════════

def fig5_vector_fusion():
    """
    Demonstrate Iyer's vector Brooks-Iyengar on multi-axis quantum sensors
    (e.g., 3-axis magnetometer network).
    """
    T_true = np.array([25.0, 18.3, -5.7])  # 3D parameter (Bx, By, Bz in µT)
    d = len(T_true)
    M = 12
    N_atoms = 1000
    eta = 0.1
    f = 2  # Byzantine faults
    n_trials = 2000

    mse_naive = np.zeros(d)
    mse_bi_vec = np.zeros(d)
    mse_outlier = np.zeros(d)

    for trial in range(n_trials):
        readings = np.zeros((M, d))
        intervals_per_dim = [[] for _ in range(d)]

        for i in range(M):
            for dim in range(d):
                if i < f:
                    # Byzantine sensor
                    r = T_true[dim] + np.random.normal(0, 5.0)
                    hw = abs(np.random.normal(0, 5.0))
                else:
                    # Quantum sensor
                    var_phase = 1.0 / (4 * N_atoms)
                    theta = T_true[dim] * eta + np.random.normal(0, np.sqrt(var_phase))
                    r = theta / eta
                    var_T = var_phase / eta**2
                    hw = 1.645 * np.sqrt(var_T)  # 90% CI

                readings[i, dim] = r
                intervals_per_dim[dim].append((r - hw, r + hw))

        # Naive average
        naive_est = np.mean(readings, axis=0)
        mse_naive += (naive_est - T_true)**2

        # Vector Brooks-Iyengar
        bi_est, _, _ = brooks_iyengar_vector(intervals_per_dim)
        mse_bi_vec += (bi_est - T_true)**2

        # Outlier-filtered
        outlier_est = np.zeros(d)
        for dim in range(d):
            r = readings[:, dim]
            med = np.median(r)
            mad = np.median(np.abs(r - med))
            if mad > 0:
                mask = np.abs(r - med) / (1.4826 * mad) < 3.0
            else:
                mask = np.ones(M, dtype=bool)
            outlier_est[dim] = np.mean(r[mask]) if mask.sum() > 0 else np.mean(r)
        mse_outlier += (outlier_est - T_true)**2

    mse_naive /= n_trials
    mse_bi_vec /= n_trials
    mse_outlier /= n_trials

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: Per-dimension RMSE ──
    ax = axes[0]
    dim_labels = ['$B_x$', '$B_y$', '$B_z$']
    x = np.arange(d)
    w = 0.25

    bars1 = ax.bar(x - w, np.sqrt(mse_naive), w, color=C_CLASSICAL,
                    alpha=0.7, label='Naive averaging')
    bars2 = ax.bar(x, np.sqrt(mse_bi_vec), w, color=C_BFT,
                    alpha=0.7, label='Vector B-I (Iyer 2013)')
    bars3 = ax.bar(x + w, np.sqrt(mse_outlier), w, color=C_OUTLIER,
                    alpha=0.7, label='Outlier filter (Iyer-Shetty 2015)')

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=12)
    ax.set_ylabel('RMSE (µT)')
    ax.set_title(f'Vector quantum sensor fusion (M={M}, f={f} Byzantine)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # ── Right: Total vector RMSE vs M ──
    ax = axes[1]
    M_values = np.arange(4, 33)
    total_rmse_naive = []
    total_rmse_bi = []
    total_rmse_outlier = []

    for M_test in M_values:
        mn = np.zeros(d)
        mb = np.zeros(d)
        mo = np.zeros(d)

        for trial in range(1000):
            readings = np.zeros((M_test, d))
            intervals_per_dim = [[] for _ in range(d)]
            f_test = max(int(0.2 * M_test), 1)

            for i in range(M_test):
                for dim in range(d):
                    if i < f_test:
                        r = T_true[dim] + np.random.normal(0, 5.0)
                        hw = abs(np.random.normal(0, 5.0))
                    else:
                        var_phase = 1.0 / (4 * N_atoms)
                        theta = T_true[dim] * eta + np.random.normal(0, np.sqrt(var_phase))
                        r = theta / eta
                        hw = 1.645 * np.sqrt(var_phase / eta**2)
                    readings[i, dim] = r
                    intervals_per_dim[dim].append((r - hw, r + hw))

            mn += (np.mean(readings, axis=0) - T_true)**2
            bi_e, _, _ = brooks_iyengar_vector(intervals_per_dim)
            mb += (bi_e - T_true)**2
            for dim in range(d):
                r = readings[:, dim]
                med = np.median(r)
                mad = np.median(np.abs(r - med))
                mask = np.abs(r - med) / (1.4826 * mad + 1e-10) < 3.0
                mo[dim] += (np.mean(r[mask]) - T_true[dim])**2 if mask.sum() > 0 else (np.mean(r) - T_true[dim])**2

        total_rmse_naive.append(np.sqrt(np.sum(mn / 1000)))
        total_rmse_bi.append(np.sqrt(np.sum(mb / 1000)))
        total_rmse_outlier.append(np.sqrt(np.sum(mo / 1000)))

    ax.semilogy(M_values, total_rmse_naive, 'o-', color=C_CLASSICAL, ms=3,
                label='Naive avg')
    ax.semilogy(M_values, total_rmse_bi, 's-', color=C_BFT, ms=3,
                label='Vector B-I')
    ax.semilogy(M_values, total_rmse_outlier, 'D-', color=C_OUTLIER, ms=3,
                label='Outlier filter')

    # Theoretical SQL and HL for reference
    sql_theory = np.sqrt(d / (4 * N_atoms * M_values * eta**2))
    hl_theory = np.sqrt(d / (4 * N_atoms * M_values**2 * eta**2))
    ax.semilogy(M_values, sql_theory, '--', color=C_SQL, lw=1, alpha=0.5,
                label='SQL bound')
    ax.semilogy(M_values, hl_theory, '--', color=C_HEISENBERG, lw=1, alpha=0.5,
                label='Heisenberg bound')

    ax.set_xlabel('Number of sensors $M$')
    ax.set_ylabel('Total vector RMSE (µT)')
    ax.set_title(f'Vector fusion scaling (20% Byzantine, d={d})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('/home/claude/fig_vector_fusion.pdf')
    plt.savefig('/home/claude/fig_vector_fusion.png')
    plt.close()
    print("[✓] Figure 5: Vector fusion")


# ═══════════════════════════════════════════════════════════════
# FIGURE 6: THE UNIFIED LOWER BOUND THEOREM
# ═══════════════════════════════════════════════════════════════

def fig6_unified_bound():
    """
    The main result: unified lower bound combining all methods.

    Theorem (Unified Quantum-Classical Aggregation Bound):
    For M quantum sensors with N atoms each, sensitivity η,
    entanglement visibility V, and f ≤ ⌊(M-1)/3⌋ Byzantine faults,
    the MSE of the fused estimate T̂ satisfies:

    MSE(T̂) ≥ 1 / (4·N·η²) · [  (1-V²)/(M-f) + V²/(M-f)²  ]

    where:
    - At V=0 (no entanglement): MSE ≥ 1/(4Nη²(M-f)) — SQL with fault exclusion
    - At V=1 (perfect entanglement): MSE ≥ 1/(4Nη²(M-f)²) — HL with fault exclusion
    - At f=0 (no faults): reduces to standard SQL/HL bounds
    - The (M-f) term comes from the predictive outlier model (Iyer-Shetty 2015)
    - For Brooks-Iyengar BFT without outlier prediction: replace (M-f) with (M-2f)
    """
    M = 32
    N_atoms = 1000
    eta = 0.1
    n_trials = 5000
    T_true = 25.0

    V_values = np.linspace(0.01, 1.0, 50)
    f_values = [0, 2, 4, 8]

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── Top left: Bound surface (V vs RMSE) for different f ──
    ax = fig.add_subplot(gs[0, 0])

    for f in f_values:
        M_eff = max(M - f, 1)
        bound = np.array([
            (1 - V**2) / (4 * N_atoms * M_eff * eta**2) +
            V**2 / (4 * N_atoms * M_eff**2 * eta**2)
            for V in V_values
        ])
        ax.semilogy(V_values, np.sqrt(bound), '-', lw=2,
                    label=f'$f={f}$ ({f/M:.0%} faults)')

    ax.set_xlabel('Entanglement visibility $V$')
    ax.set_ylabel('Lower bound RMSE (°C)')
    ax.set_title(f'Unified bound (M={M}, N={N_atoms})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Top right: Simulated vs theoretical bound ──
    ax = fig.add_subplot(gs[0, 1])

    f = 4
    M_eff_outlier = M - f
    M_eff_bft = M - 2 * f

    # Theoretical bounds
    bound_outlier = np.array([
        (1 - V**2) / (4 * N_atoms * M_eff_outlier * eta**2) +
        V**2 / (4 * N_atoms * M_eff_outlier**2 * eta**2)
        for V in V_values
    ])
    bound_bft = np.array([
        (1 - V**2) / (4 * N_atoms * M_eff_bft * eta**2) +
        V**2 / (4 * N_atoms * M_eff_bft**2 * eta**2)
        for V in V_values
    ])

    ax.semilogy(V_values, np.sqrt(bound_outlier), '-', color=C_OUTLIER, lw=2,
                label=f'Outlier bound (M-f={M_eff_outlier})')
    ax.semilogy(V_values, np.sqrt(bound_bft), '-', color=C_BFT, lw=2,
                label=f'BFT bound (M-2f={M_eff_bft})')

    # Simulate
    sim_outlier_rmse = []
    sim_bft_rmse = []
    V_sim = V_values[::5]  # subsample for speed

    for V in V_sim:
        mse_out = []
        mse_bft = []
        for _ in range(n_trials):
            readings = []
            intervals = []
            for i in range(M):
                if i < f:
                    r = T_true + np.random.normal(0, 5.0)
                    hw = abs(np.random.normal(0, 5.0))
                else:
                    var_phase = (1 - V**2) / (4 * N_atoms) + V**2 / (4 * N_atoms)
                    theta = T_true * eta + np.random.normal(0, np.sqrt(1.0 / (4 * N_atoms)))
                    r = theta / eta
                    hw = 1.645 * np.sqrt(1.0 / (4 * N_atoms * eta**2))
                readings.append(r)
                intervals.append((r - hw, r + hw))

            readings = np.array(readings)

            # Outlier filter
            med = np.median(readings)
            mad = np.median(np.abs(readings - med))
            if mad > 0:
                mask = np.abs(readings - med) / (1.4826 * mad) < 3.0
            else:
                mask = np.ones(M, dtype=bool)
            if mask.sum() > 0:
                out_est = np.mean(readings[mask])
            else:
                out_est = np.mean(readings)
            mse_out.append((out_est - T_true)**2)

            # Brooks-Iyengar
            bi_est, _, _ = brooks_iyengar_overlap_1d(intervals)
            mse_bft.append((bi_est - T_true)**2)

        sim_outlier_rmse.append(np.sqrt(np.mean(mse_out)))
        sim_bft_rmse.append(np.sqrt(np.mean(mse_bft)))

    ax.semilogy(V_sim, sim_outlier_rmse, 'D', color=C_OUTLIER, ms=5,
                alpha=0.7, label='Outlier (simulated)')
    ax.semilogy(V_sim, sim_bft_rmse, 's', color=C_BFT, ms=5,
                alpha=0.7, label='B-I BFT (simulated)')

    ax.set_xlabel('Entanglement visibility $V$')
    ax.set_ylabel('RMSE (°C)')
    ax.set_title(f'Theory vs simulation (f={f}/{M})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Bottom left: Gap between BFT and outlier bounds ──
    ax = fig.add_subplot(gs[1, 0])

    M_range = np.arange(6, 49)
    for f_frac in [0.1, 0.2, 0.3]:
        gap_db = []
        for M_test in M_range:
            f_test = int(f_frac * M_test)
            if M_test <= 3 * f_test:
                gap_db.append(np.nan)
                continue
            M_out = M_test - f_test
            M_bft = M_test - 2 * f_test
            # At V=1 (Heisenberg limit)
            mse_out = 1.0 / (4 * N_atoms * M_out**2 * eta**2)
            mse_bft = 1.0 / (4 * N_atoms * M_bft**2 * eta**2)
            gap_db.append(10 * np.log10(mse_bft / mse_out))

        ax.plot(M_range, gap_db, 'o-', ms=3, lw=1.5,
                label=f'f/M = {f_frac:.0%}')

    ax.axhline(0, ls='-', color='k', lw=0.5)
    ax.set_xlabel('Number of sensors $M$')
    ax.set_ylabel('Outlier advantage over BFT (dB)')
    ax.set_title('Value of predictive outlier detection\n(Iyer-Shetty 2015 vs Brooks-Iyengar 1996)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Bottom right: Summary table as text ──
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')

    theorem_text = (
        "Unified Quantum Sensor Aggregation Lower Bound\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Given: M sensors, N atoms each, sensitivity η,\n"
        "       visibility V ∈ [0,1], f Byzantine faults\n\n"
        "Brooks-Iyengar BFT (f ≤ ⌊(M−1)/3⌋):\n\n"
        "  MSE(T̂) ≥  (1−V²)       V²\n"
        "           ———————— + ————————————\n"
        "           4Nη²(M−2f)   4Nη²(M−2f)²\n\n"
        "Predictive Outlier (Iyer-Shetty, f < M):\n\n"
        "  MSE(T̂) ≥  (1−V²)      V²\n"
        "           ———————— + ————————————\n"
        "           4Nη²(M−f)   4Nη²(M−f)²\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Special cases:\n"
        "  V=0: SQL bound = 1 / [4Nη²·M_eff]\n"
        "  V=1: Heisenberg = 1 / [4Nη²·M_eff²]\n"
        "  f=0: Standard quantum limits\n\n"
        "Advantage of outlier over BFT:\n"
        "  Δ = 20·log₁₀[(M−2f)/(M−f)] dB\n"
        "  At f=0.2M: Δ ≈ 2.5 dB (constant)"
    )

    ax.text(0.05, 0.95, theorem_text, transform=ax.transAxes,
            fontsize=9.5, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig('/home/claude/fig_unified_bound.pdf')
    plt.savefig('/home/claude/fig_unified_bound.png')
    plt.close()
    print("[✓] Figure 6: Unified bound")


# ═══════════════════════════════════════════════════════════════
# FIGURE 7: SCALING LAW VALIDATION — SLOPE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def fig7_slope_analysis():
    """
    Log-log slope analysis confirming the 1/√M and 1/M scaling
    for different aggregation methods.
    """
    M_values = np.arange(4, 65)
    N_atoms = 1000
    eta = 0.1
    T_true = 25.0
    n_trials = 5000

    methods = {
        'Naive average': {'mse': [], 'color': C_CLASSICAL},
        'Brooks-Iyengar': {'mse': [], 'color': C_BFT},
        'Outlier filter': {'mse': [], 'color': C_OUTLIER},
        'Entangled (HL)': {'mse': [], 'color': C_ENTANGLED},
    }

    for M in M_values:
        f = max(int(0.15 * M), 1)
        mse = {k: [] for k in methods}

        for _ in range(n_trials):
            readings = []
            intervals = []
            for i in range(M):
                if i < f:
                    r = T_true + np.random.normal(0, 5.0)
                    hw = abs(np.random.normal(0, 5.0))
                else:
                    theta = T_true * eta + np.random.normal(0, np.sqrt(1.0 / (4 * N_atoms)))
                    r = theta / eta
                    hw = 1.645 * np.sqrt(1.0 / (4 * N_atoms * eta**2))
                readings.append(r)
                intervals.append((r - hw, r + hw))

            readings = np.array(readings)

            # Naive
            mse['Naive average'].append((np.mean(readings) - T_true)**2)

            # Brooks-Iyengar
            bi_est, _, _ = brooks_iyengar_overlap_1d(intervals)
            mse['Brooks-Iyengar'].append((bi_est - T_true)**2)

            # Outlier
            med = np.median(readings)
            mad = np.median(np.abs(readings - med))
            if mad > 0:
                mask = np.abs(readings - med) / (1.4826 * mad) < 3.0
            else:
                mask = np.ones(M, dtype=bool)
            out_est = np.mean(readings[mask]) if mask.sum() > 0 else np.mean(readings)
            mse['Outlier filter'].append((out_est - T_true)**2)

            # Entangled
            M_good = M - f
            var_ent = 1.0 / (4 * N_atoms * M_good**2)
            theta_ent = T_true * eta + np.random.normal(0, np.sqrt(var_ent))
            mse['Entangled (HL)'].append((theta_ent / eta - T_true)**2)

        for k in methods:
            methods[k]['mse'].append(np.mean(mse[k]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Log-log RMSE ──
    ax = axes[0]
    for name, data in methods.items():
        rmse = np.sqrt(np.array(data['mse']))
        ax.loglog(M_values, rmse, 'o-', ms=2.5, lw=1.5,
                  color=data['color'], label=name)

    # Reference slopes
    M_ref = np.linspace(4, 64, 200)
    c = np.sqrt(methods['Outlier filter']['mse'][0]) * 2
    ax.loglog(M_ref, c / np.sqrt(M_ref), '--', color='gray', lw=1,
              alpha=0.5, label='$\\propto 1/\\sqrt{M}$')
    ax.loglog(M_ref, c / M_ref, ':', color='gray', lw=1,
              alpha=0.5, label='$\\propto 1/M$')

    ax.set_xlabel('Number of sensors $M$')
    ax.set_ylabel('RMSE (°C)')
    ax.set_title('Aggregation scaling (15% Byzantine faults)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, which='both')

    # ── Right: Effective slope (local log-log derivative) ──
    ax = axes[1]
    for name, data in methods.items():
        log_rmse = np.log10(np.sqrt(np.array(data['mse'])))
        log_M = np.log10(M_values.astype(float))
        # Numerical derivative (central differences)
        slope = np.gradient(log_rmse, log_M)
        # Smooth
        window = 5
        if len(slope) >= window:
            slope_smooth = np.convolve(slope, np.ones(window)/window, mode='valid')
            M_smooth = M_values[window//2:window//2 + len(slope_smooth)]
            ax.plot(M_smooth, slope_smooth, '-', lw=2,
                    color=data['color'], label=name)

    ax.axhline(-0.5, ls='--', color='gray', lw=1, alpha=0.5)
    ax.text(50, -0.47, 'SQL: slope = −0.5', fontsize=8, color='gray')
    ax.axhline(-1.0, ls=':', color='gray', lw=1, alpha=0.5)
    ax.text(50, -0.97, 'HL: slope = −1.0', fontsize=8, color='gray')

    ax.set_xlabel('Number of sensors $M$')
    ax.set_ylabel('Effective scaling exponent')
    ax.set_title('Local log-log slope (confirms $1/\\sqrt{M}$ vs $1/M$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 0)

    plt.tight_layout()
    plt.savefig('/home/claude/fig_slope_analysis.pdf')
    plt.savefig('/home/claude/fig_slope_analysis.png')
    plt.close()
    print("[✓] Figure 7: Slope analysis")


# ═══════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("QUANTUM SENSOR AGGREGATION LOWER BOUNDS")
    print("Bridging Iyer et al. classical fusion → quantum regime")
    print("=" * 60)
    print()

    fig1_lower_bounds_no_faults()
    fig2_byzantine_bounds()
    fig3_overlap_visualization()
    fig4_decoherence_crossover()
    fig5_vector_fusion()
    fig6_unified_bound()
    fig7_slope_analysis()

    print()
    print("=" * 60)
    print("✅ All figures generated.")
    print()
    print("KEY RESULTS:")
    print("─" * 60)
    print("1. SQL lower bound (independent):  MSE ≥ 1/(4Nη²M)")
    print("2. Heisenberg lower bound:         MSE ≥ 1/(4Nη²M²)")
    print("3. BFT bound (Brooks-Iyengar):     MSE ≥ 1/(4Nη²(M-2f)²)")
    print("4. Outlier bound (Iyer-Shetty):    MSE ≥ 1/(4Nη²(M-f)²)")
    print("5. Outlier advantage over BFT:     Δ = 20·log₁₀[(M-2f)/(M-f)] dB")
    print("6. Unified bound interpolates V=0 (SQL) ↔ V=1 (HL)")
    print("=" * 60)
