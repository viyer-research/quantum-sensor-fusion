#!/bin/bash
# Run this after creating the repo at:
#   https://github.com/viyer-research/quantum-sensor-fusion
#
# Usage:
#   cd quantum-sensor-fusion
#   chmod +x push_to_github.sh
#   ./push_to_github.sh

git init
git add -A
git commit -m "Initial commit: Quantum-Enhanced Distributed Sensor Fusion

- Unified lower bound: MSE >= (1-V²)/[4Nη²·M_eff] + V²/[4Nη²·M_eff²]
- Brooks-Iyengar BFT (M-2f) vs predictive outlier (M-f)
- Intel Lab 54-mote validation with spatial clustering
- 12-page paper with 8 simulation figures
- Missing value analysis: 38.1% avg missing per mote
- SNR advantage: 20-27 dB Heisenberg over classical per cluster"

git branch -M main
git remote add origin https://github.com/viyer-research/quantum-sensor-fusion.git
git push -u origin main

echo ""
echo "✅ Pushed to https://github.com/viyer-research/quantum-sensor-fusion"
