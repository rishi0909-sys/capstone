# System Hardware & Environment Compatibility Report
**Generated:** 2026-10-14 23:31:00 IST  
**Project:** Intelligent Pattern Discovery Engine — Retail Intelligence Explorer

---

## 1. Hardware & Operating System Specifications

| Component | Detected Specification |
| :--- | :--- |
| **Operating System** | macOS 26.6.2 (Darwin Kernel 25.6.0 - arm64) |
| **CPU Model** | Apple M2 (arm64) |
| **Cores** | 8 Physical / 8 Logical Cores (Apple Silicon M2) |
| **Total System RAM** | **8.0 GB** Unified Memory |
| **Available RAM (Est.)**| **~1.3 GB** (Active: 1.29 GB, Wired: 1.79 GB, Inactive: 1.19 GB) |
| **Memory Pressure** | Compressor active: 17.39 GB compressed |
| **GPU Model** | Apple M2 (8 Cores, Metal 4) |
| **GPU Architecture** | Unified Memory Architecture (Shared host RAM, No dedicated VRAM) |
| **CUDA Capability** | **None** (Apple Metal only, no NVIDIA CUDA/ROCm) |
| **Project Disk Space** | **14.09 GB free** out of 228.27 GB (93.8% used on /Users/rishabhmudgal/Documents/components/capstone/backend) |

---

## 2. Python & Machine Learning Runtime Audit

| Attribute | Value |
| :--- | :--- |
| **Python Interpreter** | `/Users/rishabhmudgal/Documents/components/capstone/backend/venv/bin/python3` |
| **Python Version** | Python 3.13.1 (64bit) |
| **Virtual Environment** | Active (`backend/venv`) |
| **XGBoost Installation** | Version `3.4.1` |
| **XGBoost Smoke Test** | **PASSED** |
| **XGBoost Execution Backend** | **CPU (OpenMP Multi-threaded)** |
| **NumPy Version** | `2.5.3` |
| **Pandas Version** | `3.0.5` |
| **Scikit-Learn Version** | `1.9.1` |
| **mlxtend Version** | `0.25.0` |
| **FastAPI / Uvicorn** | FastAPI `0.141.1` / Uvicorn `0.53.0` |

---

## 3. Workload Compatibility Verdicts

| Workload | Compatibility Verdict | Estimated Resources | Recommended Configuration |
| :--- | :---: | :--- | :--- |
| **1. FP-Growth Mining** | **COMPATIBLE WITH LIMITATIONS** | RAM: ~600MB - 1.5GB<br>Time: 1-5 sec | Keep `min_support >= 0.010` (1%). Avoid values < 0.003 to prevent memory explosion. |
| **2. Daily Demand XGBoost** | **COMPATIBLE** | RAM: < 300MB<br>Time: < 1 sec | `n_estimators=100, max_depth=4, lr=0.05`. Excellent fit for 365 daily points. |
| **3. Weekly Demand XGBoost** | **COMPATIBLE** | RAM: < 200MB<br>Time: < 0.5 sec | `n_estimators=60, max_depth=3, lr=0.08`. Small sample (51 wks); prevent deep trees. |
| **4. Product-Level Forecasting** | **COMPATIBLE WITH LIMITATIONS** | RAM: ~500MB<br>Time: 5-15 sec | Limit to Top 5–10 volume products. Do not train 4,000 separate models in one loop. |
| **5. Hyperparameter Tuning** | **COMPATIBLE WITH LIMITATIONS** | RAM: ~1.0GB<br>Time: 15-45 sec | Time-series expanding window CV with ~36 trials on 4 worker threads (`n_jobs=4`). |
| **6. Neural-Net / GPU Experiments** | **NOT RECOMMENDED** | RAM: > 3.0GB<br>Time: Minutes | No CUDA GPU available. Small tabular time-series would severely overfit on LSTMs. |

---

## 4. Key Engineering Constraints & Safeguards

1. **8 GB Unified Memory Constraint**:  
   The host machine operates with 8 GB unified memory with active memory compression (~18 GB virtual compressed).  
   *Action*: Heavy deep copying of 500k-row DataFrames must be avoided. Intermediate variables must be garbage collected, and operations must aggregate before feature building.

2. **Apple Silicon CPU-Only Training**:  
   XGBoost on macOS does not support Metal GPU acceleration for regression trees; it runs via multi-threaded CPU OpenMP.  
   *Action*: Set `n_jobs=4` to leave CPU headroom for the Next.js frontend and system tasks.

3. **Disk Space Watch**:  
   The root disk has **14 GB free (93% capacity)**.  
   *Action*: Do not generate redundant intermediate gigabyte-sized CSV dumps or uncompressed checkpoints.
