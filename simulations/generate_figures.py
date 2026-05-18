#!/usr/bin/env python3
"""
Quantum Sensor Fusion — Simulation and Figure Generation
Generates all figures for the paper on quantum-enhanced distributed sensor networks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

np.random.seed(42)

# ── Global style ──
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.8,
})

COLORS = {
    'classical': '#2C7BB6',
    'sql': '#D7191C',
    'entangled': '#1A9641',
    'heisenberg': '#FDAE61',
    'data': '#404040',
}

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Single-sensor quantum projection noise
# ═══════════════════════════════════════════════════════════════

def fig1_single_sensor_qpn():
    """Compare classical vs quantum single-sensor noise models."""
    T_true = 25.0  # True temperature in °C
    sensitivity = 0.1  # rad/°C
    n_measurements = 500

    # Classical sensor: additive Gaussian noise
    classical_noise_std = 0.5  # °C
    T_classical = T_true + np.random.normal(0, classical_noise_std, n_measurements)

    # Quantum sensor: phase measurement with QPN
    N_atoms = 100  # atoms in sensor
    theta_true = T_true * sensitivity
    # QPN variance on phase: 1/(4N)
    qpn_std = 1.0 / (2 * np.sqrt(N_atoms))
    theta_measured = theta_true + np.random.normal(0, qpn_std, n_measurements)
    T_quantum = theta_measured / sensitivity

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Scatter: classical
    ax = axes[0]
    ax.scatter(range(n_measurements), T_classical, s=4, alpha=0.5, c=COLORS['classical'])
    ax.axhline(T_true, color='k', ls='--', lw=1, label=f'True = {T_true}°C')
    ax.set_xlabel('Measurement index')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Classical sensor (additive noise)')
    ax.legend(fontsize=9)

    # Scatter: quantum
    ax = axes[1]
    ax.scatter(range(n_measurements), T_quantum, s=4, alpha=0.5, c=COLORS['sql'])
    ax.axhline(T_true, color='k', ls='--', lw=1, label=f'True = {T_true}°C')
    ax.set_xlabel('Measurement index')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Quantum sensor (N={N_atoms} atoms, QPN)')
    ax.legend(fontsize=9)

    # Histogram comparison
    ax = axes[2]
    bins = np.linspace(23, 27, 60)
    ax.hist(T_classical, bins=bins, alpha=0.55, color=COLORS['classical'],
            label=f'Classical (σ={np.std(T_classical):.3f}°C)', density=True)
    ax.hist(T_quantum, bins=bins, alpha=0.55, color=COLORS['sql'],
            label=f'Quantum (σ={np.std(T_quantum):.3f}°C)', density=True)
    ax.axvline(T_true, color='k', ls='--', lw=1)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Density')
    ax.set_title('Noise distribution comparison')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/claude/fig1_single_sensor.pdf')
    plt.savefig('/home/claude/fig1_single_sensor.png')
    plt.close()
    print("[✓] Figure 1: Single sensor QPN")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Scaling with number of sensors M
# ═══════════════════════════════════════════════════════════════

def fig2_sensor_scaling():
    """Show 1/√M (classical/SQL) vs 1/M (Heisenberg) scaling."""
    M_values = np.arange(1, 65)
    N_atoms = 1000  # atoms per sensor
    n_trials = 2000
    T_true = 25.0
    sensitivity = 0.1

    classical_noise_std = 0.5
    qpn_var_single = 1.0 / (4 * N_atoms)  # phase variance per sensor

    mse_classical = []
    mse_sql = []
    mse_entangled = []

    for M in M_values:
        # Classical: average M noisy readings
        T_cl = T_true + np.random.normal(0, classical_noise_std / np.sqrt(M), n_trials)
        mse_classical.append(np.mean((T_cl - T_true)**2))

        # Quantum SQL (independent sensors, averaged)
        phase_noise_std = np.sqrt(qpn_var_single / M)
        theta_m = T_true * sensitivity + np.random.normal(0, phase_noise_std, n_trials)
        T_q = theta_m / sensitivity
        mse_sql.append(np.mean((T_q - T_true)**2))

        # Quantum entangled (Heisenberg-limited): variance ~ 1/(4*N*M^2)
        ent_phase_std = np.sqrt(1.0 / (4 * N_atoms * M**2))
        theta_e = T_true * sensitivity + np.random.normal(0, ent_phase_std, n_trials)
        T_e = theta_e / sensitivity
        mse_entangled.append(np.mean((T_e - T_true)**2))

    mse_classical = np.array(mse_classical)
    mse_sql = np.array(mse_sql)
    mse_entangled = np.array(mse_entangled)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Linear scale
    ax = axes[0]
    ax.plot(M_values, np.sqrt(mse_classical), 'o-', ms=3, color=COLORS['classical'],
            label='Classical (additive noise)')
    ax.plot(M_values, np.sqrt(mse_sql), 's-', ms=3, color=COLORS['sql'],
            label='Quantum SQL (independent)')
    ax.plot(M_values, np.sqrt(mse_entangled), '^-', ms=3, color=COLORS['entangled'],
            label='Quantum entangled (Heisenberg)')
    ax.set_xlabel('Number of sensors M')
    ax.set_ylabel('RMSE (°C)')
    ax.set_title('Fusion precision vs. sensor count')
    ax.legend()

    # Log-log scale
    ax = axes[1]
    ax.loglog(M_values, np.sqrt(mse_classical), 'o-', ms=3, color=COLORS['classical'],
              label='Classical ~ 1/√M')
    ax.loglog(M_values, np.sqrt(mse_sql), 's-', ms=3, color=COLORS['sql'],
              label='SQL ~ 1/√M')
    ax.loglog(M_values, np.sqrt(mse_entangled), '^-', ms=3, color=COLORS['entangled'],
              label='Entangled ~ 1/M')

    # Reference lines
    M_ref = np.linspace(1, 64, 200)
    c1 = np.sqrt(mse_classical[0])
    c2 = np.sqrt(mse_sql[0])
    c3 = np.sqrt(mse_entangled[0])
    ax.loglog(M_ref, c1 / np.sqrt(M_ref), '--', color=COLORS['classical'], alpha=0.4, lw=1)
    ax.loglog(M_ref, c2 / np.sqrt(M_ref), '--', color=COLORS['sql'], alpha=0.4, lw=1)
    ax.loglog(M_ref, c3 / M_ref, '--', color=COLORS['entangled'], alpha=0.4, lw=1)

    ax.set_xlabel('Number of sensors M')
    ax.set_ylabel('RMSE (°C)')
    ax.set_title('Log-log scaling (dashed = theoretical)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('/home/claude/fig2_scaling.pdf')
    plt.savefig('/home/claude/fig2_scaling.png')
    plt.close()
    print("[✓] Figure 2: Sensor scaling")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Effect of atom number N on single-sensor precision
# ═══════════════════════════════════════════════════════════════

def fig3_atom_scaling():
    """Show how QPN decreases with N atoms per sensor."""
    N_values = np.logspace(1, 5, 50).astype(int)
    n_trials = 5000
    T_true = 25.0
    sensitivity = 0.1

    rmse_values = []
    for N in N_values:
        qpn_std = 1.0 / (2 * np.sqrt(N))
        theta = T_true * sensitivity + np.random.normal(0, qpn_std, n_trials)
        T_inferred = theta / sensitivity
        rmse_values.append(np.sqrt(np.mean((T_inferred - T_true)**2)))

    rmse_values = np.array(rmse_values)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N_values, rmse_values, 'o-', ms=4, color=COLORS['sql'], label='Simulated RMSE')
    # Theoretical
    N_th = np.logspace(1, 5, 200)
    theoretical = 1.0 / (2 * np.sqrt(N_th) * sensitivity)
    ax.loglog(N_th, theoretical, '--', color=COLORS['heisenberg'], lw=2, label=r'Theoretical $\frac{1}{2\sqrt{N}\cdot\eta}$')

    ax.set_xlabel('Number of atoms N per sensor')
    ax.set_ylabel('RMSE (°C)')
    ax.set_title('Single-sensor precision vs. atom count')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('/home/claude/fig3_atom_scaling.pdf')
    plt.savefig('/home/claude/fig3_atom_scaling.png')
    plt.close()
    print("[✓] Figure 3: Atom scaling")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Decoherence impact on quantum advantage
# ═══════════════════════════════════════════════════════════════

def fig4_decoherence():
    """Show how decoherence degrades quantum advantage."""
    M = 16
    N_atoms = 1000
    n_trials = 3000
    T_true = 25.0
    sensitivity = 0.1

    # Decoherence parameter: visibility V in [0, 1]
    # V=1 is perfect, V=0 is fully decohered
    V_values = np.linspace(0.05, 1.0, 40)

    rmse_sql = []
    rmse_ent = []

    for V in V_values:
        # SQL bound doesn't depend on entanglement visibility
        qpn_sql_std = np.sqrt(1.0 / (4 * N_atoms * M))
        theta_sql = T_true * sensitivity + np.random.normal(0, qpn_sql_std, n_trials)
        T_sql = theta_sql / sensitivity
        rmse_sql.append(np.sqrt(np.mean((T_sql - T_true)**2)))

        # Entangled: effective variance interpolates between SQL and Heisenberg
        # var_ent = (1-V^2)/(4*N*M) + V^2/(4*N*M^2)
        var_ent = (1 - V**2) / (4 * N_atoms * M) + V**2 / (4 * N_atoms * M**2)
        theta_ent = T_true * sensitivity + np.random.normal(0, np.sqrt(var_ent), n_trials)
        T_ent = theta_ent / sensitivity
        rmse_ent.append(np.sqrt(np.mean((T_ent - T_true)**2)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(V_values, rmse_sql, '-', color=COLORS['sql'], lw=2, label='SQL (independent sensors)')
    ax.plot(V_values, rmse_ent, '-', color=COLORS['entangled'], lw=2, label='Entangled sensors')
    ax.fill_between(V_values, rmse_ent, rmse_sql, alpha=0.15, color=COLORS['entangled'],
                     label='Quantum advantage region')
    ax.axhline(rmse_sql[0], ls=':', color=COLORS['sql'], alpha=0.4)

    ax.set_xlabel('Entanglement visibility V')
    ax.set_ylabel('RMSE (°C)')
    ax.set_title(f'Decoherence impact on quantum advantage (M={M}, N={N_atoms})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/fig4_decoherence.pdf')
    plt.savefig('/home/claude/fig4_decoherence.png')
    plt.close()
    print("[✓] Figure 4: Decoherence impact")


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Fusion algorithm comparison
# ═══════════════════════════════════════════════════════════════

def fig5_fusion_algorithms():
    """Compare simple averaging vs Kalman vs Bayesian fusion with quantum sensors."""
    M = 8
    N_atoms = 500
    T_true_trajectory = 25.0 + 2.0 * np.sin(np.linspace(0, 4*np.pi, 200))  # time-varying
    n_steps = len(T_true_trajectory)
    sensitivity = 0.1

    # Per-sensor QPN std
    qpn_std = 1.0 / (2 * np.sqrt(N_atoms) * sensitivity)

    # Generate M sensor readings at each timestep
    readings = np.zeros((n_steps, M))
    for t in range(n_steps):
        readings[t] = T_true_trajectory[t] + np.random.normal(0, qpn_std, M)

    # 1. Simple averaging
    avg_est = np.mean(readings, axis=1)

    # 2. Kalman filter
    kalman_est = np.zeros(n_steps)
    x_hat = readings[0].mean()
    P = qpn_std**2
    Q = 0.01  # process noise
    R = (qpn_std**2) / M  # measurement noise (averaged)
    for t in range(n_steps):
        # Predict
        x_pred = x_hat
        P_pred = P + Q
        # Update
        z = np.mean(readings[t])
        K = P_pred / (P_pred + R)
        x_hat = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        kalman_est[t] = x_hat

    # 3. Weighted Bayesian (reliability-weighted)
    # Simulate some sensors being noisier (decoherence)
    sensor_weights = np.array([1.0, 1.0, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4])
    sensor_weights /= sensor_weights.sum()
    bayesian_est = np.average(readings, axis=1, weights=sensor_weights)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Trajectory comparison
    ax = axes[0]
    ax.plot(T_true_trajectory, 'k-', lw=2, label='True temperature', zorder=5)
    ax.plot(avg_est, '-', color=COLORS['classical'], alpha=0.7, label='Simple averaging')
    ax.plot(kalman_est, '-', color=COLORS['entangled'], alpha=0.8, label='Kalman filter')
    ax.plot(bayesian_est, '-', color=COLORS['heisenberg'], alpha=0.7, label='Weighted Bayesian')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Fusion algorithm comparison (M={M} quantum sensors, N={N_atoms})')
    ax.legend(loc='upper right')

    # Running RMSE
    ax = axes[1]
    window = 20
    for est, name, color in [(avg_est, 'Averaging', COLORS['classical']),
                               (kalman_est, 'Kalman', COLORS['entangled']),
                               (bayesian_est, 'Bayesian', COLORS['heisenberg'])]:
        err = (est - T_true_trajectory)**2
        running_rmse = np.sqrt(np.convolve(err, np.ones(window)/window, mode='valid'))
        ax.plot(running_rmse, '-', color=color, label=name)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Running RMSE (°C)')
    ax.set_title(f'Running RMSE (window={window})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/fig5_fusion_algorithms.pdf')
    plt.savefig('/home/claude/fig5_fusion_algorithms.png')
    plt.close()
    print("[✓] Figure 5: Fusion algorithms")


# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Entanglement advantage heatmap (M × N)
# ═══════════════════════════════════════════════════════════════

def fig6_advantage_heatmap():
    """Heatmap of dB advantage of entangled over SQL as function of M and N."""
    M_range = np.arange(2, 33)
    N_range = np.logspace(1, 4, 30).astype(int)
    n_trials = 2000
    T_true = 25.0
    sensitivity = 0.1

    advantage_db = np.zeros((len(N_range), len(M_range)))

    for i, N in enumerate(N_range):
        for j, M in enumerate(M_range):
            # SQL variance
            var_sql = 1.0 / (4 * N * M * sensitivity**2)
            # Heisenberg variance
            var_hl = 1.0 / (4 * N * M**2 * sensitivity**2)
            # Advantage in dB
            advantage_db[i, j] = 10 * np.log10(var_sql / var_hl)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(advantage_db, aspect='auto', origin='lower',
                    extent=[M_range[0], M_range[-1], np.log10(N_range[0]), np.log10(N_range[-1])],
                    cmap='YlGnBu', vmin=0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Entanglement advantage (dB)')

    ax.set_xlabel('Number of sensors M')
    ax.set_ylabel('Atoms per sensor N (log₁₀)')
    ax.set_title('Theoretical quantum advantage: Heisenberg vs SQL')

    # Add contour lines
    M_grid, N_grid = np.meshgrid(M_range, np.log10(N_range))
    cs = ax.contour(M_grid, N_grid, advantage_db, levels=[3, 6, 9, 12, 15],
                     colors='white', linewidths=1, linestyles='--')
    ax.clabel(cs, fmt='%.0f dB', fontsize=9, colors='white')

    plt.tight_layout()
    plt.savefig('/home/claude/fig6_heatmap.pdf')
    plt.savefig('/home/claude/fig6_heatmap.png')
    plt.close()
    print("[✓] Figure 6: Advantage heatmap")


# ═══════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating figures for Quantum Sensor Fusion paper...\n")
    fig1_single_sensor_qpn()
    fig2_sensor_scaling()
    fig3_atom_scaling()
    fig4_decoherence()
    fig5_fusion_algorithms()
    fig6_advantage_heatmap()
    print("\n✅ All figures generated successfully.")
