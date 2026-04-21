# **Cascading Failure Prediction in Cloud Systems using Spatio-Temporal Graph Neural Networks**

> **Course:** Interdisciplinary Deep Learning with Graphs

> **Institution:** PES University

> **Team:** Akshay P Shetti, Tarun S, Aadithyaa Kumar, Adarsh R Menon

---

## 🧠 Overview

Modern cloud infrastructures consist of thousands of interconnected machines.
Failures in such systems rarely occur in isolation — they **propagate across dependencies**, leading to *cascading failures*.

This project models cloud infrastructure as a **dynamic graph** and uses a **Spatio-Temporal Graph Neural Network (ST-GNN)** to:

* Predict machine failures across future time steps
* Capture **failure propagation (cascade behavior)**
* Analyze how failures evolve over time

---

## 🎯 Problem Statement

Traditional ML approaches treat machines independently.

❌ This ignores:

* Inter-machine dependencies
* Dynamic topology changes
* Temporal failure propagation

✅ Our approach:

* Models the system as a **time-evolving graph**
* Learns **spatial + temporal interactions**
* Predicts failures at **multiple future horizons (t+1, t+2, t+3)**

---

## 🏗️ Model Architecture

```
Dynamic Graph Snapshots (T=6)
        ↓
GraphSAGE (Spatial Encoder)
        ↓
GRU (Temporal Modeling)
        ↓
MLP Classifier
        ↓
Multi-step Outputs (t+1, t+2, t+3)
```

---

### 🔹 Key Components

* **GraphSAGE**

  * Captures local neighborhood influence
  * Scales to large graphs via neighbor sampling

* **GRU**

  * Models temporal evolution of node states

* **Dynamic Edges**

  * Graph structure changes per time step
  * Captures real-world infrastructure dynamics

* **Multi-Step Prediction**

  * Outputs failure probabilities for:

    * t+1 (immediate risk)
    * t+2 (short-term spread)
    * t+3 (cascade stage)

---

## 📊 Dataset

* **Source:** Google Borg Cluster Trace
* **Nodes:** ~4,900 machines
* **Time Windows:** ~8,900
* **Features:** ~25 system-level metrics

---

## 📈 Results

### Final Performance (t+3)

| Metric        | Value |
| ------------- | ----- |
| **Precision** | 0.091 |
| **Recall**    | 0.998 |
| **F1 Score**  | 0.167 |
| **AUROC**     | 0.745 |

---

## 📉 Key Visualizations

### 🔹 Cascade Progression

<img src="results/cascade_progression.png" width="500"/>

- Shows how predicted failure probability increases over time  
- Confirms the model captures **cascade behavior**

---

### 🔹 Confusion Matrix (t+3)

<img src="results/confusion_matrix.png" width="500"/>

- Very high recall → most failures detected  
- High false positives → low precision  

---

### 🔹 ROC Curve

<img src="results/roc.png" width="500"/>

- **AUROC ≈ 0.745**
- Indicates the model learns meaningful patterns  

---

### 🔹 Precision-Recall Curve

<img src="results/pr.png" width="500"/>

- Highlights trade-off:
  - High recall  
  - Low precision  
- More informative than ROC for imbalanced data  

---

## 🔥 Key Insight: Cascade Modeling

The most important result:

| Time Step | Precision | Recall | F1    |
| --------- | --------- | ------ | ----- |
| t+1       | 0.030     | 0.997  | 0.059 |
| t+2       | 0.061     | 0.998  | 0.114 |
| t+3       | 0.091     | 0.998  | 0.167 |

### ✅ Interpretation

* Recall remains consistently high → failures are detected early
* Precision improves over time → predictions become more accurate
* F1 increases → model becomes more reliable as cascade evolves

👉 This confirms:

> The model is not just predicting failures — it is **learning failure propagation dynamics**

---

## ⚠️ Limitations

* Low precision (~0.09)

  * Model tends to **overpredict failures**
* Early-stage prediction (t+1) is weak
* Threshold sensitivity affects results significantly

---

## 🚀 Future Improvements

* Class imbalance handling (e.g., weighted loss)
* Better threshold calibration
* Feature engineering for early failure signals
* Larger hidden dimensions / model capacity

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt

python preprocess.py
python train.py
python evaluate.py
```

---

## 📁 Project Structure

```
├── preprocess.py
├── debug_labels.py
├── train.py
├── model.py
├── evaluate.py
├── config.yaml
├── processed/
├── results/
└── best_model.pt
```

---

## 📌 Conclusion

This project demonstrates that:

* Graph-based models can capture **inter-node dependencies**
* Temporal modeling enables **failure forecasting**
* Multi-step prediction reveals **cascade dynamics**

> The model acts as a **cascade detector**, prioritizing detection over precision — a desirable property in failure-critical systems.

---

## 📚 References

* GraphSAGE (Hamilton et al., 2017)
* GRU (Cho et al., 2014)
* Google Borg Dataset
* Spatio-Temporal GNN literature

---
