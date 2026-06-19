# Hybrid Classical and Quantum Materials Optimization System

A full-stack platform that trains and compares **classical ML** (Random
Forest, XGBoost, Gradient Boosting), **quantum ML** (QNN, VQR, QKR),
**classical optimization** (Newton's Method) and **quantum optimization**
(QAOA) for materials-property prediction and ranking, and presents all
results in a Bootstrap 5 + Chart.js dashboard served by a Flask backend.

---

## 1. Project Overview

| Layer                  | Technology                                               |
| ---------------------- | -------------------------------------------------------- |
| Classical ML           | scikit-learn (Random Forest, Gradient Boosting), XGBoost |
| Quantum ML             | PennyLane (QNN, VQR), Qiskit Machine Learning (QKR)      |
| Classical Optimization | Newton's Method (custom implementation)                  |
| Quantum Optimization   | QAOA via `qiskit-optimization` / `qiskit-algorithms`     |
| Backend                | Flask REST API                                           |
| Frontend               | HTML5, CSS3, Bootstrap 5, Chart.js                       |
| Export                 | OpenPyXL (`exports/results.xlsx`)                        |

The pipeline: load datasets → clean/merge/select 50 materials → 80/20 split
→ train 3 classical + 3 quantum regressors → build ensembles → optimize
rankings with Newton's Method and QAOA → serve everything through a Flask
API → visualize in the dashboard → export to Excel.

---

## 2. Architecture

```
Hybrid_Materials_Optimization/
├── datasets/                  # Matbench-style datasets (+ generator script)
├── classical_models/          # RF / XGBoost / GB + Newton optimizer
├── quantum_models/             # QNN / VQR / QKR + QAOA optimizer
├── backend/                   # Flask app, REST API, services
├── dashboard/                 # Bootstrap 5 + Chart.js frontend
├── exports/                   # results.xlsx + export script
├── reports/                   # (reserved for generated reports)
├── requirements.txt
├── README.md
└── PRD.md
```

---

## 3. Installation

```bash
cd Hybrid_Materials_Optimization
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 is recommended (Qiskit/PennyLane are validated against the
versions pinned in `requirements.txt`).

---

## 4. Dataset Setup

This build ships with `datasets/generate_datasets.py`, which creates local,
schema-compatible **synthetic** versions of the three Matbench files
(`matbench_mp_gap.json.gz`, `matbench_log_kvrh.json.gz`,
`matbench_log_gvrh.json.gz`). This keeps the whole project runnable without
any external network calls.

Regenerate them at any time with:

```bash
cd datasets
python3 generate_datasets.py
```

### Using the real Matbench datasets

To use the actual Matbench benchmark data instead:

```bash
pip install matminer
python -c "
from matminer.datasets import load_dataset
import pandas as pd
for name in ['matbench_mp_gap', 'matbench_log_kvrh', 'matbench_log_gvrh']:
    df = load_dataset(name)
    df.to_json(f'datasets/{name}.json.gz', orient='records', compression='gzip')
"
```

You will then need to add a composition-featurization step (e.g. via
`matminer.featurizers.composition`) so the resulting columns match the
`FEATURE_COLUMNS` list in `classical_models/preprocessing.py`, and rename the
target columns to `mp_gap`, `log_kvrh`, `log_gvrh` respectively, joined on
`material_id`.

---

## 5. Training Instructions

Run the classical pipeline (trains RF/XGB/GB, applies Newton's Method,
writes CSVs + `.pkl` models):

```bash
cd classical_models
python3 evaluate.py
```

Run the quantum pipeline (trains QNN/VQR/QKR, applies QAOA, writes CSVs):

```bash
cd quantum_models
python3 evaluate.py
```

> The quantum pipeline runs full circuit simulations and typically takes
> 30–90 seconds depending on hardware.

Generate the consolidated Excel export:

```bash
cd exports
python3 export_results.py
```

---

## 6. Running the Backend

```bash
cd backend
python3 app.py
```

The Flask server starts on `http://localhost:5000` and also serves the
dashboard (the API and frontend run from the same process).

---

## 7. Running the Dashboard

With the backend running, simply open:

```
http://localhost:5000
```

Pages:

- `/` — Home Dashboard (summary KPIs + overall comparison charts)
- `/classical.html` — Classical Models (RF / XGBoost / GB metrics & charts)
- `/quantum.html` — Quantum Models (QNN / VQR / QKR metrics & charts)
- `/optimizer.html` — Optimizer Comparison (Newton's Method vs. QAOA)
- `/recommendation.html` — Final Recommendation (Top 10 materials)

---

## 8. API Documentation

All endpoints are prefixed with `/api` and return:

```json
{ "status": "success", "data": ... }
```

or, on failure:

```json
{ "status": "error", "message": "..." }
```

| Method | Endpoint                     | Description                                                  |
| ------ | ---------------------------- | ------------------------------------------------------------ |
| GET    | `/api/classical-metrics`     | MAE/RMSE/R²/timings for RF, XGBoost, GB                      |
| GET    | `/api/classical-predictions` | Per-material predictions + ensemble score                    |
| GET    | `/api/quantum-metrics`       | MAE/RMSE/R²/timings for QNN, VQR, QKR                        |
| GET    | `/api/quantum-predictions`   | Per-material predictions + quantum ensemble                  |
| GET    | `/api/classical-optimizer`   | Newton's Method summary (best material, runtime, objective)  |
| GET    | `/api/quantum-optimizer`     | QAOA summary (best material, runtime, iterations, objective) |
| GET    | `/api/recommendations`       | Top 10 materials + best classical/quantum/final picks        |
| GET    | `/api/dashboard-summary`     | Aggregated KPIs for the Home Dashboard                       |

---

## 9. Expected Outputs

After running both `evaluate.py` scripts and `export_results.py`:

```
classical_models/models/{rf,xgb,gb}.pkl
classical_models/results/classical_{predictions,metrics,optimized}.csv
quantum_models/results/quantum_{predictions,metrics,optimized}.csv
exports/results.xlsx   (7 sheets: Classical Metrics, Quantum Metrics,
                         Optimizer Metrics, Classical Ranking, Quantum
                         Ranking, Top Materials, Recommendations)
```

The dashboard will display:

- Total materials (50), total models (6)
- Best classical model / best quantum model (by R²)
- Best optimizer / best material (by objective score)
- Per-model accuracy and runtime charts
- Top 10 ranked materials with predicted vs. optimized scores

---

## 10. Notes on the Quantum Implementation

- **QNN** — PennyLane `AngleEmbedding` + `BasicEntanglerLayers`, trained with
  Adam, 4 qubits (PCA-reduced features).
- **VQR** — PennyLane `AngleEmbedding` + `StronglyEntanglingLayers`, trained
  with gradient descent, distinct observable from QNN.
- **QKR** — Qiskit `ZZFeatureMap` + `FidelityQuantumKernel`, fed into a
  classical `KernelRidge` regressor (precomputed-kernel mode).
- **QAOA** — `qiskit-optimization` QUBO (maximize selected scores subject to
  a cardinality constraint) solved via `qiskit-algorithms` QAOA with COBYLA.

All quantum circuits run on Qiskit/PennyLane's local statevector simulators
— no QPU access or external quantum cloud account is required.

## 11. Images

### Classical Model Training Images
![classical Model Training](/images/classical_model_training.png)

### Quantum Model Training Images
![Quantum Model Training](/images/quantum_model_training.png)

### Home Page
![Home Page](/images/home.png)

### Classical Model Page
![Classical Model Page](/images/classical_model.png)

### Quantum Model Page
![Quantum Model Page](/images/quantum_model.png)

### Optimizer Page
![Optimizer Page](/images/optimizer.png)

### Final Recommendation Page
![Final Recommendation Page](/images/final_recommendation.png)

## Author

Developed by **Goutam Parashuram Gotur**, a Computer Science student with a strong interest in Artificial Intelligence, Quantum Computing, Machine Learning, and Materials Science. This project explores the integration of Classical Machine Learning, Quantum Machine Learning, and Quantum Optimization techniques to predict material properties, optimize material selection, and recommend high-performance materials through an interactive web dashboard.

### Contact

- Name: Goutam Parashuram Gotur
- GitHub: https://github.com/goutam-tech
- Email: goutamgotur2006@gmail.com