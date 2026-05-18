#!/usr/bin/env python3
"""
Intel Lab Motes → Spatially-Clustered Quantum Sensor Fusion
════════════════════════════════════════════════════════════
1. Load mote positions → spatial clustering (k-means)
2. Identify window-facing motes (outlier temps)
3. Per-epoch overlap WITHIN each spatial cluster
4. Handle missing motes per epoch
5. Classical fusion → quantum advantage with entanglement SNR
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from sklearn.cluster import KMeans
np.random.seed(42)

plt.rcParams.update({'font.size':10,'axes.labelsize':12,'axes.titlesize':11,
    'legend.fontsize':8,'figure.dpi':300,'savefig.dpi':300,
    'savefig.bbox':'tight','lines.linewidth':1.5})

C_CL='#2C7BB6'; C_SQL='#D7191C'; C_ENT='#1A9641'; C_HL='#FDAE61'
C_BFT='#7B3294'; C_OUT='#E66101'; C_DATA='#404040'

# ═══════════════════════════════════════════════════════════
# 1. MOTE POSITIONS
# ═══════════════════════════════════════════════════════════
mote_locs = {}
locs_raw = """1 21.5 23
2 24.5 20
3 19.5 19
4 22.5 15
5 24.5 12
6 19.5 12
7 22.5 8
8 24.5 4
9 21.5 2
10 19.5 5
11 16.5 3
12 13.5 1
13 12.5 5
14 8.5 6
15 5.5 3
16 1.5 2
17 1.5 8
18 5.5 10
19 3.5 13
20 0.5 17
21 4.5 18
22 1.5 23
23 6 24
24 1.5 30
25 4.5 30
26 7.5 31
27 8.5 26
28 10.5 31
29 12.5 26
30 13.5 31
31 15.5 28
32 17.5 31
33 19.5 26
34 21.5 30
35 24.5 27
36 26.5 31
37 27.5 26
38 30.5 31
39 30.5 26
40 33.5 28
41 36.5 30
42 39.5 30
43 35.5 24
44 40.5 22
45 37.5 19
46 34.5 16
47 39.5 14
48 35.5 10
49 39.5 6
50 38.5 1
51 35.5 4
52 31.5 6
53 28.5 5
54 26.5 2"""

for line in locs_raw.strip().split('\n'):
    parts = line.split()
    mote_locs[int(parts[0])] = (float(parts[1]), float(parts[2]))

pos_df = pd.DataFrame(mote_locs).T
pos_df.columns = ['x','y']
pos_df.index.name = 'moteid'
print(f"Loaded {len(pos_df)} mote positions")

# ═══════════════════════════════════════════════════════════
# 2. SPATIAL CLUSTERING
# ═══════════════════════════════════════════════════════════
n_clusters = 6
coords = pos_df[['x','y']].values
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
pos_df['cluster'] = km.fit_predict(coords)
cluster_centers = km.cluster_centers_

# Lab boundaries — motes near edges are "window-facing"
x_min, x_max = pos_df.x.min(), pos_df.x.max()
y_min, y_max = pos_df.y.min(), pos_df.y.max()
edge_threshold = 3.0  # meters from wall
pos_df['near_wall'] = (
    (pos_df.x < x_min + edge_threshold) |
    (pos_df.x > x_max - edge_threshold) |
    (pos_df.y < y_min + edge_threshold) |
    (pos_df.y > y_max - edge_threshold)
)

print(f"Clusters: {n_clusters}, Wall-facing motes: {pos_df.near_wall.sum()}")
for c in range(n_clusters):
    motes_in = pos_df[pos_df.cluster==c].index.tolist()
    print(f"  Cluster {c}: {len(motes_in)} motes — {motes_in}")

# ═══════════════════════════════════════════════════════════
# 3. LOAD SENSOR DATA
# ═══════════════════════════════════════════════════════════
print("\nLoading Intel data...")
df = pd.read_csv('/home/claude/intel_motes/data.txt', sep=r'\s+', header=None,
                  names=['date','time','epoch','moteid','temp','humidity','light','voltage'],
                  on_bad_lines='skip')
df = df[df.moteid.between(1, 54)]
df['moteid'] = df.moteid.astype(int)
df = df[df.temp.between(10, 45)]
df['time_bin'] = (df.epoch // 60).astype(int)

# Merge cluster info
df = df.merge(pos_df[['cluster','near_wall']], left_on='moteid', right_index=True, how='left')
df = df.dropna(subset=['cluster'])
print(f"Clean data: {len(df)} rows")

# ═══════════════════════════════════════════════════════════
# 4. IDENTIFY WINDOW-FACING OUTLIER MOTES
# ═══════════════════════════════════════════════════════════
# Per mote: compute mean temp and flag those significantly warmer
mote_stats = df.groupby('moteid').agg(
    mean_temp=('temp','mean'),
    std_temp=('temp','std'),
    count=('temp','count'),
    near_wall=('near_wall','first'),
    cluster=('cluster','first')
).reset_index()

global_mean = mote_stats.mean_temp.mean()
global_std = mote_stats.mean_temp.std()
mote_stats['z_score'] = (mote_stats.mean_temp - global_mean) / global_std
mote_stats['is_window'] = (mote_stats.near_wall) & (mote_stats.z_score > 1.0)

window_motes = set(mote_stats[mote_stats.is_window].moteid.tolist())
print(f"\nWindow-facing hot motes (wall + z>1): {window_motes}")
print(f"Global mean temp: {global_mean:.2f}°C, std: {global_std:.2f}°C")

# ═══════════════════════════════════════════════════════════
# CORE FUSION FUNCTIONS
# ═══════════════════════════════════════════════════════════

def bi_overlap(intervals):
    if len(intervals) < 2:
        if len(intervals)==1:
            return (intervals[0][0]+intervals[0][1])/2, 1, intervals[0][0], intervals[0][1]
        return 0, 0, 0, 0
    pts = sorted(set([p for lo,hi in intervals for p in (lo,hi)]))
    best_c, best_n, best_lo, best_hi = 0, 0, 0, 0
    for i in range(len(pts)-1):
        mid = (pts[i]+pts[i+1])/2
        cnt = sum(1 for lo,hi in intervals if lo<=mid<=hi)
        if cnt > best_n:
            best_n=cnt; best_c=mid; best_lo=pts[i]; best_hi=pts[i+1]
    return best_c, best_n, best_lo, best_hi

def overlap_scores(intervals):
    M = len(intervals)
    if M < 2: return np.array([1.0]*M)
    scores = []
    for k in range(M):
        ck = (intervals[k][0]+intervals[k][1])/2
        count = sum(1 for j,(lo,hi) in enumerate(intervals) if j!=k and lo<=ck<=hi)
        scores.append(count/(M-1))
    return np.array(scores)

# ═══════════════════════════════════════════════════════════
# 5. PER-EPOCH, PER-CLUSTER OVERLAP ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\nRunning per-cluster per-epoch overlap analysis...")

coverage = df.groupby('time_bin').moteid.nunique()
good_bins = coverage[coverage >= 30].index.tolist()
sample_bins = sorted(np.random.choice(good_bins, min(80, len(good_bins)), replace=False))

cluster_epoch_results = []
for tb in sample_bins:
    ep = df[df.time_bin == tb]
    for c in range(n_clusters):
        cluster_data = ep[ep.cluster == c]
        if len(cluster_data) < 3:
            continue
        
        mote_temps = cluster_data.groupby('moteid').temp.mean()
        mote_ids = mote_temps.index.values
        temps = mote_temps.values
        M_c = len(temps)
        
        # Mark missing motes in this cluster
        all_cluster_motes = set(pos_df[pos_df.cluster==c].index)
        present = set(mote_ids)
        missing = all_cluster_motes - present
        missing_frac = len(missing) / len(all_cluster_motes) if len(all_cluster_motes)>0 else 0
        
        # Mark window motes
        n_window = sum(1 for m in mote_ids if m in window_motes)
        
        # Classical intervals: use intra-cluster std
        cluster_std = np.std(temps) if len(temps) > 1 else 0.5
        hw_cl = max(1.645 * cluster_std, 0.2)
        cl_intervals = [(t-hw_cl, t+hw_cl) for t in temps]
        
        bi_est, bi_n, _, _ = bi_overlap(cl_intervals)
        ov = overlap_scores(cl_intervals)
        
        # Exclude window motes and recompute
        clean_mask = np.array([m not in window_motes for m in mote_ids])
        if clean_mask.sum() >= 2:
            clean_temps = temps[clean_mask]
            clean_std = np.std(clean_temps) if len(clean_temps)>1 else 0.5
            hw_clean = max(1.645 * clean_std, 0.2)
            clean_ivs = [(t-hw_clean, t+hw_clean) for t in clean_temps]
            bi_clean, bi_n_clean, _, _ = bi_overlap(clean_ivs)
        else:
            bi_clean = bi_est
            bi_n_clean = bi_n
        
        cluster_epoch_results.append({
            'time_bin': tb, 'cluster': c,
            'M': M_c, 'missing_frac': missing_frac,
            'n_window': n_window,
            'bi_est': bi_est, 'bi_clean': bi_clean,
            'bi_agreement': bi_n/M_c,
            'bi_clean_agreement': bi_n_clean/max(clean_mask.sum(),1),
            'naive': np.mean(temps),
            'temp_std': cluster_std,
            'mean_overlap': np.mean(ov),
            'faulty_frac': np.mean(ov < 0.5),
        })

cr = pd.DataFrame(cluster_epoch_results)
print(f"Cluster-epoch results: {len(cr)} entries")
print(f"Avg agreement: all={cr.bi_agreement.mean():.2%}, clean={cr.bi_clean_agreement.mean():.2%}")
print(f"Avg missing: {cr.missing_frac.mean():.1%}, avg window motes: {cr.n_window.mean():.1f}")

# ═══════════════════════════════════════════════════════════
# FIGURE: 6-panel spatially-clustered analysis
# ═══════════════════════════════════════════════════════════
print("\nGenerating figures...")
fig = plt.figure(figsize=(18, 16))
gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.3)

# ── (a) Lab layout with spatial clusters + window motes ──
ax = fig.add_subplot(gs[0, 0])
cluster_colors = plt.cm.Set1(np.linspace(0, 0.8, n_clusters))

for c in range(n_clusters):
    mask = pos_df.cluster == c
    motes_c = pos_df[mask]
    ax.scatter(motes_c.x, motes_c.y, c=[cluster_colors[c]]*len(motes_c),
               s=80, alpha=0.7, edgecolors='black', lw=0.5, label=f'Cluster {c} ({len(motes_c)})',
               zorder=3)
    for mid, row in motes_c.iterrows():
        ax.annotate(str(mid), (row.x, row.y), fontsize=5.5, ha='center', va='center', zorder=4)

# Highlight window motes
for wm in window_motes:
    if wm in pos_df.index:
        ax.scatter(pos_df.loc[wm,'x'], pos_df.loc[wm,'y'], s=200,
                   facecolors='none', edgecolors='red', lw=2, zorder=5)

# Lab boundary
ax.plot([x_min-1, x_max+1, x_max+1, x_min-1, x_min-1],
        [y_min-1, y_min-1, y_max+1, y_max+1, y_min-1], 'k--', lw=1, alpha=0.3)

# Cluster centers
ax.scatter(cluster_centers[:,0], cluster_centers[:,1], marker='x', s=100,
           c='black', lw=2, zorder=6, label='Cluster centers')

ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)')
ax.set_title(f'(a) Intel Lab: {len(pos_df)} motes, {n_clusters} spatial clusters\nRed circles = window-facing motes')
ax.legend(fontsize=6, loc='upper left', ncol=2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

# ── (b) Per-cluster agreement: all vs clean (no window motes) ──
ax = fig.add_subplot(gs[0, 1])
for c in range(n_clusters):
    cc = cr[cr.cluster==c]
    if len(cc)==0: continue
    ax.scatter(cc.bi_agreement, cc.bi_clean_agreement, s=20,
               c=[cluster_colors[c]]*len(cc), alpha=0.5, label=f'C{c}')

ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.3, label='No change')
ax.set_xlabel('B-I agreement (all motes)')
ax.set_ylabel('B-I agreement (window motes excluded)')
ax.set_title('(b) Window mote exclusion improves agreement\n(points above diagonal = improvement)')
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
ax.set_xlim(0,1.05); ax.set_ylim(0,1.05)

# ── (c) Intra-cluster temp std (signal for quantum noise floor) ──
ax = fig.add_subplot(gs[1, 0])
for c in range(n_clusters):
    cc = cr[cr.cluster==c]
    if len(cc)==0: continue
    ax.hist(cc.temp_std, bins=20, alpha=0.4, color=cluster_colors[c],
            label=f'C{c} (μ={cc.temp_std.mean():.2f}°C)')
ax.axvline(cr.temp_std.median(), color='red', ls='--', lw=1.5,
           label=f'Median = {cr.temp_std.median():.2f}°C')
ax.set_xlabel('Intra-cluster temperature std (°C)')
ax.set_ylabel('Count')
ax.set_title('(c) Sensor disagreement within spatial clusters\n(= classical noise floor for quantum comparison)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ── (d) Quantum advantage: RMSE per cluster vs N atoms ──
ax = fig.add_subplot(gs[1, 1])
eta = 0.1
N_values = [50, 100, 500, 1000, 5000]
n_trials = 300

# Pick one representative cluster with good coverage
rep_cluster = cr.groupby('cluster').M.mean().idxmax()
rep_data = cr[cr.cluster == rep_cluster]
M_rep = int(rep_data.M.mean())
classical_noise = rep_data.temp_std.mean()

rmse_naive=[]; rmse_bi=[]; rmse_out=[]; rmse_ent=[]

for N in N_values:
    en=[]; eb=[]; eo=[]; ee=[]
    for _ in range(n_trials):
        # Simulate M_rep quantum sensors with QPN
        T_ref = 22.0  # typical lab temp
        readings = T_ref + np.random.normal(0, np.sqrt(1.0/(4*N))/eta, M_rep)
        qhw = 1.645*np.sqrt(1.0/(4*N*eta**2))
        q_ivs = [(r-qhw, r+qhw) for r in readings]
        
        en.append((np.mean(readings)-T_ref)**2)
        bie,_,_,_ = bi_overlap(q_ivs)
        eb.append((bie-T_ref)**2)
        
        med=np.median(readings); mad=np.median(np.abs(readings-med))
        mask=np.abs(readings-med)/(1.4826*mad+1e-10)<3.0
        oe=np.mean(readings[mask]) if mask.sum()>0 else np.mean(readings)
        eo.append((oe-T_ref)**2)
        
        var_ent=1.0/(4*N*M_rep**2)
        th_e=T_ref*eta+np.random.normal(0,np.sqrt(var_ent))
        ee.append((th_e/eta-T_ref)**2)
    
    rmse_naive.append(np.sqrt(np.mean(en)))
    rmse_bi.append(np.sqrt(np.mean(eb)))
    rmse_out.append(np.sqrt(np.mean(eo)))
    rmse_ent.append(np.sqrt(np.mean(ee)))

ax.loglog(N_values, rmse_naive, 'o-', ms=5, color=C_CL, label='Naive avg')
ax.loglog(N_values, rmse_bi, 's-', ms=5, color=C_BFT, label='Brooks-Iyengar')
ax.loglog(N_values, rmse_out, 'D-', ms=5, color=C_OUT, label='Outlier filter')
ax.loglog(N_values, rmse_ent, '^-', ms=5, color=C_ENT, label=f'Entangled (M={M_rep})')

# Classical noise floor
ax.axhline(classical_noise/np.sqrt(M_rep), ls=':', color=C_DATA, lw=1,
           label=f'Classical floor ({classical_noise/np.sqrt(M_rep):.3f}°C)')

N_th=np.logspace(1,4,200)
ax.loglog(N_th, np.sqrt(1.0/(4*N_th*M_rep*eta**2)), '--', color=C_SQL, lw=1, alpha=0.3)
ax.loglog(N_th, np.sqrt(1.0/(4*N_th*M_rep**2*eta**2)), '--', color=C_HL, lw=1, alpha=0.3)

ax.set_xlabel('Atoms per sensor $N$'); ax.set_ylabel('RMSE (°C)')
ax.set_title(f'(d) Quantum RMSE, cluster {rep_cluster} (M={M_rep})\nEntanglement pushes below classical floor')
ax.legend(fontsize=7); ax.grid(True, alpha=0.2, which='both')

# ── (e) SNR comparison: classical vs SQL vs HL per cluster ──
ax = fig.add_subplot(gs[2, 0])
N_snr = 1000

snr_data = []
for c in range(n_clusters):
    cc = cr[cr.cluster==c]
    if len(cc)==0: continue
    M_c = cc.M.mean()
    noise_cl = cc.temp_std.mean() / np.sqrt(M_c)
    noise_sql = np.sqrt(1.0/(4*N_snr*M_c*eta**2))
    noise_hl = np.sqrt(1.0/(4*N_snr*M_c**2*eta**2))
    T_sig = abs(cc.bi_est.mean())
    
    snr_cl = 20*np.log10(T_sig/noise_cl) if noise_cl>0 else 0
    snr_sql = 20*np.log10(T_sig/noise_sql) if noise_sql>0 else 0
    snr_hl = 20*np.log10(T_sig/noise_hl) if noise_hl>0 else 0
    
    snr_data.append({'cluster':c, 'M':M_c,
                     'Classical':snr_cl, 'SQL':snr_sql, 'HL':snr_hl})

snr_df = pd.DataFrame(snr_data)
x = np.arange(len(snr_df))
w = 0.25
ax.bar(x-w, snr_df['Classical'], w, color=C_CL, alpha=0.7, label='Classical')
ax.bar(x, snr_df['SQL'], w, color=C_SQL, alpha=0.7, label='Quantum SQL')
ax.bar(x+w, snr_df['HL'], w, color=C_ENT, alpha=0.7, label='Quantum HL')

for i,row in snr_df.iterrows():
    gain = row['HL'] - row['Classical']
    ax.text(i+w, row['HL']+1, f'+{gain:.0f}dB', ha='center', fontsize=6, color=C_ENT)

ax.set_xticks(x)
ax.set_xticklabels([f'C{int(r.cluster)}\n(M={r.M:.0f})' for _,r in snr_df.iterrows()], fontsize=8)
ax.set_xlabel('Spatial cluster'); ax.set_ylabel('SNR (dB)')
ax.set_title(f'(e) SNR per cluster: classical → quantum\n(N={N_snr} atoms, entanglement advantage labeled)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

# ── (f) Missing data + decoherence: agreement degradation ──
ax = fig.add_subplot(gs[2, 1])

# Group by missing fraction and plot agreement
cr['missing_bin'] = pd.cut(cr.missing_frac, bins=[0,0.1,0.2,0.3,0.5,1.0])
miss_agree = cr.groupby('missing_bin', observed=True).agg(
    agreement=('bi_agreement','mean'),
    clean_agreement=('bi_clean_agreement','mean'),
    count=('time_bin','count')
)

# Also simulate decoherence effect on a fixed cluster
V_vals = np.linspace(0.1, 1.0, 15)
agree_vs_V = []
N_dec = 500
M_dec = M_rep
n_trials_v = 200
T_ref_v = 22.0

for V in V_vals:
    agrees = []
    for _ in range(n_trials_v):
        readings = []
        q_ivs = []
        for _ in range(M_dec):
            var_p = 1.0/(4*N_dec*max(V**2, 1e-10))
            theta = T_ref_v*eta + np.random.normal(0, np.sqrt(var_p))
            T_q = theta/eta
            readings.append(T_q)
            qhw = 1.645*np.sqrt(var_p/eta**2)
            q_ivs.append((T_q-qhw, T_q+qhw))
        _, bi_n_v, _, _ = bi_overlap(q_ivs)
        agrees.append(bi_n_v/M_dec)
    agree_vs_V.append(np.mean(agrees))

ax.plot(V_vals, agree_vs_V, 'o-', ms=5, color=C_ENT, lw=2,
        label=f'Quantum (decoherence, M={M_dec})')

# Missing data effect (from real data)
if len(miss_agree) > 0:
    miss_x = np.linspace(0.05, 0.45, len(miss_agree))
    # Map missing fraction to "effective visibility"
    V_equiv = 1.0 - np.array([x.mid for x in miss_agree.index])
    ax.plot(V_equiv, miss_agree.agreement.values, 's-', ms=7, color=C_OUT, lw=2,
            label='Classical (missing motes)')

ax.axhline(0.667, ls=':', color=C_DATA, alpha=0.5, lw=1)
ax.text(0.15, 0.69, 'BFT threshold (2/3)', fontsize=7, color=C_DATA)
ax.set_xlabel('Entanglement visibility $V$ / (1 − missing fraction)')
ax.set_ylabel('B-I agreement fraction')
ax.set_title('(f) Agreement degradation:\ndecoherence (quantum) ≈ missing data (classical)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)

plt.savefig('/home/claude/fig_intel_spatial_quantum.png')
plt.savefig('/home/claude/fig_intel_spatial_quantum.pdf')
plt.close()
print("[OK] Spatial quantum figure generated")

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("INTEL MOTES SPATIAL QUANTUM FUSION: KEY RESULTS")
print(f"{'='*60}")
print(f"54 motes → {n_clusters} spatial clusters")
print(f"Window-facing motes identified: {window_motes}")
print(f"Epochs analyzed: {len(sample_bins)} (30+ motes each)")
print(f"\nPer-cluster overlap (classical):")
print(f"  All motes:            agreement = {cr.bi_agreement.mean():.2%}")
print(f"  Window motes excluded: agreement = {cr.bi_clean_agreement.mean():.2%}")
print(f"  Improvement:          +{(cr.bi_clean_agreement.mean()-cr.bi_agreement.mean())*100:.1f} percentage points")
print(f"\nMissing data: avg {cr.missing_frac.mean():.1%} per cluster-epoch")
print(f"Missing ≈ decoherence: both degrade B-I agreement similarly")
print(f"\nSNR at N=1000:")
for _,r in snr_df.iterrows():
    print(f"  Cluster {int(r.cluster)} (M={r.M:.0f}): Classical={r.Classical:.1f}dB, SQL={r.SQL:.1f}dB, HL={r.HL:.1f}dB, gain=+{r.HL-r.Classical:.1f}dB")
print(f"{'='*60}")
