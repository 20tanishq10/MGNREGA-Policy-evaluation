# 📊 MGNREGA Multi-Dimensional Policy Evaluation

A comprehensive empirical analysis of the Mahatma Gandhi National Rural Employment Guarantee Act (MGNREGA) using district-level panel data (2014–2024), evaluating its impact across multiple economic dimensions.

---

## 🚀 Project Overview

This project goes beyond traditional single-outcome policy evaluation and develops a **five-pillar framework** to analyze MGNREGA’s impact:

* 🛡️ **Shock Absorption** (Resilience to economic shocks)
* ⚖️ **Distributional Inclusion** (Participation of marginalized groups)
* 💰 **Income Effects** (Welfare improvement)
* ⚙️ **Labor Market Distortion** (Efficiency trade-offs)
* 🔄 **Structural Persistence** (Dynamic/lagged effects)

We combine econometric modeling, panel data analysis, and spatial visualization to provide a **holistic understanding of policy impact**.

---

## 📁 Project Structure

```
MGNREGA-Project/
│
├── MGNREGA Analysis/
│   ├── 01_pillar1_shock_absorption.ipynb
│   ├── 02_pillar2_distributional_effects.ipynb
│   ├── 03_pillar3_income_effect.ipynb
│   ├── 04_pillar4_labor_distortion.ipynb
│   ├── 05_pillar5_structural_persistence.ipynb
│   ├── analysis_utils.py
│   └── outputs/
│
├── MGNREGA EDA/
│   ├── eda_analysis.ipynb
│   └── figures/
│
├── MGNREGA Final Visualization/
│   ├── maps/
│   ├── radar/
│   ├── time_series/
│   └── composite_index/
│
├── MGNREGA Project Report/
│   └── Overleaf Latex Files
│
├── Dataset/
│   ├── Panel_Data_2014_2024.csv
│   ├── district_mapping.csv
│   └── raw_sources/
│
├── README.md
```

---

## 🧠 Methodology

### 📌 Panel Data Framework

We use a fixed-effects model:

[
Y_{it} = \beta \cdot MGNREGA_{it} + \gamma X_{it} + \mu_i + \lambda_t + \epsilon_{it}
]

* District Fixed Effects → control for unobserved heterogeneity
* Time Fixed Effects → control for macro shocks
* Clustered standard errors

---

### 📊 Pillar-wise Estimation

Each pillar is estimated separately:

| Pillar           | Method                              |
| ---------------- | ----------------------------------- |
| Shock Absorption | Interaction with rainfall shocks    |
| Distribution     | Subgroup regressions (SC/ST, women) |
| Income           | Direct FE regression                |
| Distortion       | Income vs productivity comparison   |
| Structural       | Lagged MGNREGA impact               |

---

### 📈 Composite Index

All pillars are normalized:

[
Score = \frac{X - X_{min}}{X_{max} - X_{min}}
]

Final index:

[
Index = \frac{1}{5} \sum_{k=1}^{5} Score_k
]

---

## 📊 Key Findings

* ✅ Strong **shock absorption** during COVID
* ✅ Positive **income effects** across districts
* ⚠️ **Distributional outcomes are uneven**
* ⚠️ Evidence of **efficiency trade-offs**
* ❌ Limited **long-term structural transformation**

👉 MGNREGA acts more as a **stabilization tool** than a growth engine.

---

## 🗺️ Visual Outputs

The project includes:

* 📍 District-level heatmaps (India)
* 📉 Time-series evolution (2014–2024)
* 🕸️ Radar charts (multi-dimensional comparison)
* 📊 EDA dashboards

---

## 🧾 Data Sources

* CMIE Consumer Pyramids (Income & Consumption)
* Directorate of Economics & Statistics (Agriculture)
* IMD Rainfall Data
* NITI Aayog (MPI)
* MGNREGA Official Portal (Scraped)

---

## ⚠️ Limitations

* Observational study (not causal)
* Proxy-based measurement (e.g., yield for productivity)
* District-level aggregation (no household-level insights)

---

## 👨‍💻 Authors

* **Tanishq Gupta** (22322031)
* **Kavish Jain** (22322017)
* **Tushar Singh** (22322032)
* **Yash Kumar** (22322034)

---

## 📌 Future Work

* Add causal identification (IV / DiD)
* Use household-level micro data
* Extend framework to other policies

---

## ⭐ If you found this useful

Give a ⭐ on GitHub and feel free to fork or build upon this work!

---
