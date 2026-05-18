# Quantum-Enhanced Distributed Sensor Fusion

**Lower Bounds on Aggregation from Projection Noise to Heisenberg-Limited Byzantine-Tolerant Networks**

Vasanth Iyer — Grambling State University

---

## Overview

This repository contains the simulation code, datasets, and LaTeX source for a paper on quantum-enhanced distributed sensor fusion. The work derives unified lower bounds on fusion error that bridge classical sensor network algorithms (Brooks-Iyengar, SPOTLESS, predictive outlier detection) with quantum metrology (Standard Quantum Limit → Heisenberg Limit).

### Key Result

For M quantum sensors with N atoms each, sensitivity η, entanglement visibility V ∈ [0,1], and f Byzantine faults:

```
MSE(T̂) ≥ (1 − V²) / [4Nη² · M_eff] + V² / [4Nη² · M_eff²]
```

where M_eff = M − 2f (Brooks-Iyengar BFT) or M_eff = M − f (predictive outlier detection).

## Repository Structure

```
quantum-sensor-fusion/
├── paper/
│   ├── quantum_sensor_fusion.tex     # Full LaTeX source (12 pages)
│   └── quantum_sensor_fusion.pdf     # Compiled PDF
├── simulations/
│   ├── quantum_lower_bounds.py       # Core lower bound derivation + 7 figures
│   ├── generate_figures.py           # Supplementary figures (QPN, scaling, heatmap)
│   └── intel_spatial_quantum.py      # Intel Lab Motes spatial-cluster analysis
├── data/
│   ├── mote_locs.txt                 # Intel Lab 54-mote (x,y) positions
│   ├── DOWNLOAD_INTEL_DATA.txt       # Instructions to get data.txt (150MB)
│   └── sensor_8_table1.csv           # 8-sensor crisp dataset (Table 1)
├── results/
│   ├── intel_mote_statistics.csv     # Per-mote stats + missing value analysis
│   └── intel_epoch_coverage.csv      # Per-epoch mote coverage + missing IDs
├── figures/                          # All generated figures (PNG)
├── requirements.txt
└── README.md
```

## Simulations

### 1. Quantum Lower Bounds (`quantum_lower_bounds.py`)

Derives and validates the unified aggregation bound with Monte Carlo simulations:
- Brooks-Iyengar overlap function on quantum sensor confidence intervals
- Byzantine fault impact: BFT (M−2f) vs predictive outlier (M−f) effective sensors
- Decoherence crossover: critical visibility V* where classical BFT beats degraded entanglement
- Slope analysis confirming −0.5 (SQL) and −1.0 (Heisenberg) scaling exponents

### 2. Intel Lab Motes (`intel_spatial_quantum.py`)

Applies the framework to real sensor data (54 motes, 2.3M readings):
- **Spatial clustering**: k-means on mote (x,y) positions → 6 clusters
- **Window-facing mote detection**: wall-adjacent + z-score > 1.0 → motes 22, 24, 25, 38
- **Missing value analysis**: 38.1% average missing per mote, 10 motes with >50% missing
- **Per-cluster overlap**: 96.5% B-I agreement (all), 97.1% (window-excluded)
- **SNR comparison**: Heisenberg limit provides 20–27 dB over classical per cluster

### 3. Supplementary Figures (`generate_figures.py`)

Single-sensor QPN, atom count scaling, fusion algorithm comparison, advantage heatmap.

## Key Findings

| Metric | Classical | Quantum SQL | Quantum HL |
|--------|-----------|-------------|------------|
| Scaling | 1/√M | 1/√M | **1/M** |
| Intel C2 SNR (M=9) | 35.2 dB | 52.2 dB | **61.9 dB** |
| Outlier advantage over BFT | — | — | **~2.5 dB** |

### Missing Data ≈ Decoherence

A key finding is that missing motes in classical networks degrade Brooks-Iyengar agreement in the same pattern as decoherence visibility V degrades entangled quantum fusion. This validates the classical fusion algorithms as the quantum decoherence management layer.

## Requirements

```bash
pip install numpy matplotlib pandas scikit-learn scipy
```

## Running

```bash
# Generate lower bound figures
python simulations/quantum_lower_bounds.py

# Intel Motes analysis (requires data.txt in data/ directory)
python simulations/intel_spatial_quantum.py

# Compile paper
cd paper && pdflatex quantum_sensor_fusion.tex && pdflatex quantum_sensor_fusion.tex
```

## Related Publications

1. Brooks & Iyengar, "Robust Distributed Computing and Sensing Algorithm," IEEE Computer, 1996
2. Murthy & Iyer, "Fuzzy Logic Based Sensor Fusion," EUSFLAT, 2007
3. Iyer & Iyengar, "F-measure Attribute Performance with Unreliable Sensors," ICDM, 2011
4. Iyer et al., "SPOTLESS: Similarity Patterns of Trajectories in Label-less Sensor Streams," PerCom, 2013
5. Iyer, "Ensemble Stream Model for Data-cleaning in Sensor Networks," PhD Dissertation, FIU, 2013
6. Iyer & Shetty, "Virtual Sensor Tracking using Byzantine Fault Tolerance and Predictive Outlier Model," SPIE, 2015
7. Iyer et al., "Statistical Methods in AI: Rare Event Learning," ISPRS Annals, 2015
8. Iyer et al., "Fast Multi-Modal Reuse: Co-occurrence Pre-trained Deep Learning Models," SPIE, 2019

## License

MIT License

## Citation

```bibtex
@article{iyer2026quantum,
  title={Quantum-Enhanced Distributed Sensor Fusion: Lower Bounds on Aggregation 
         from Projection Noise to Heisenberg-Limited Byzantine-Tolerant Networks},
  author={Iyer, Vasanth},
  institution={Grambling State University},
  year={2026}
}
```
