# **Cascading Failure Prediction in Cloud Systems using Spatio-Temporal Graph Neural Networks**

> **Course:** Interdisciplinary Deep Learning with Graphs
> **Institution:** PES University

> **Team:**
> Akshay P Shetti, Tarun S, Aadithyaa Kumar, Adarsh R Menon

---

## Overview

Modern cloud infrastructures consist of thousands of interconnected machines.
Failures in such systems rarely occur in isolation — they **propagate across dependencies**, leading to *cascading failures*.

This project models cloud infrastructure as a **dynamic graph** and uses a **Spatio-Temporal Graph Neural Network (ST-GNN)** to:

* Predict machine failures across future time steps
* Capture **failure propagation (cascade behavior)**
* Analyze how failures evolve over time

---

## Problem Statement

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

## Model Architecture

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

### Key Components

**GraphSAGE**

* Captures local neighborhood dependencies
* Scales to large graphs via neighbor sampling

**GRU**

* Models temporal evolution of node states

**Dynamic Edges**

* Graph structure changes per time step
* Captures real-world infrastructure dynamics

**Multi-Step Prediction**

* Outputs failure probabilities for:

  * t+1 (immediate risk)
  * t+2 (short-term spread)
  * t+3 (cascade stage)

---

## Dataset

* **Source:** Google Borg Cluster Trace
* **Nodes:** ~4,900 machines
* **Time Windows:** ~8,900
* **Features:** ~25 system-level metrics

---

## Results

### Final Performance (t+3)

| Metric        | Value |
| ------------- | ----- |
| **Precision** | 0.091 |
| **Recall**    | 0.998 |
| **F1 Score**  | 0.167 |
| **AUROC**     | 0.745 |

---

## Baseline Comparison

We compare our ST-GNN with a Logistic Regression baseline trained on **raw machine-level features**.

### Results

| Model               | Task                   | Precision | Recall | F1   |
| ------------------- | ---------------------- | --------- | ------ | ---- |
| Logistic Regression | Static (ever fail)     | 1.00      | 0.64   | 0.78 |
| ST-GNN (Ours)       | Temporal (t+3 cascade) | 0.09      | 0.998  | 0.17 |

---

### Interpretation

* **Logistic Regression**

  * Uses only static machine features
  * Predicts whether a machine will **ever fail**
  * Achieves **high precision** (very few false positives)
  * Misses many failures (**lower recall**)
  * Cannot model:

    * when failures occur
    * how failures propagate
    * inter-machine dependencies

* **ST-GNN (Ours)**

  * Models **graph structure + temporal dynamics**
  * Predicts failures across multiple future steps
  * Achieves **near-perfect recall**
  * Captures **cascading failure behavior**
  * Trades precision for **early and comprehensive detection**

---

### Key Insight

> Logistic Regression answers *“which machines will fail at some point”*,
> while ST-GNN answers *“when failures occur and how they propagate across the system”*.

---

### Practical Importance (False Negatives)

In real cloud systems:

* **False Negatives (missed failures)** are extremely costly
* Missing a failure can lead to:

  * service outages
  * cascading system breakdown
  * large-scale infrastructure impact

Therefore:

> **High recall is more important than high precision**

* ST-GNN prioritizes **recall (~1.0)** → detects almost all failures
* Logistic Regression prioritizes **precision** but misses failures

--> This makes ST-GNN more suitable for **failure-critical systems** 

---

## Key Visualizations

### Cascade Progression

<img src="results/cascade_progression.png" width="500"/>

* Shows increasing failure probability over time
* Confirms the model captures **cascade learning behavior**

---

### Confusion Matrix (t+3)

<img src="results/confusion_matrix.png" width="500"/>

* Very high recall → most failures detected
* High false positives → low precision

---

### ROC Curve

<img src="results/roc.png" width="500"/>

* **AUROC ≈ 0.745**
* Indicates the model learns meaningful patterns

---

### Precision-Recall Curve

<img src="results/pr.png" width="500"/>

* Strong recall, weak precision
* More informative for imbalanced datasets

---

### t-SNE Visualization (Decision Space)

<img src="results/tsne.png" width="600"/>

* Nodes projected into 2D learned representation space
* Failure nodes are not tightly clustered
* Indicates failures are distributed across the system

Key takeaway:

> Failures are not isolated events — they emerge from **distributed system dynamics**

---

## Key Insight: Cascade Modeling

| Time Step | Precision | Recall | F1    |
| --------- | --------- | ------ | ----- |
| t+1       | 0.030     | 0.997  | 0.059 |
| t+2       | 0.061     | 0.998  | 0.114 |
| t+3       | 0.091     | 0.998  | 0.167 |

### Interpretation

* Recall remains consistently high → failures are detected early
* Precision improves over time → predictions become more accurate
* F1 increases → model reliability improves as cascade evolves

This confirms:

> The model is not just predicting failures — it is **learning failure propagation dynamics**

---

## Limitations

* Low precision (~0.09)

  * Model tends to overpredict failures
* Early-stage prediction (t+1) is weaker
* Strong sensitivity to threshold selection

---

## Future Improvements

* Class imbalance handling (weighted loss / focal loss)
* Better threshold calibration
* Feature engineering for early failure signals
* Larger hidden dimensions / model capacity
* Improved temporal modeling (e.g., attention mechanisms)

---

## How to Run

```bash
pip install -r requirements.txt

python preprocess.py
python train.py
python evaluate.py
```

---

## Project Structure

```
├── preprocess.py
├── debug_labels.py
├── train.py
├── model.py
├── evaluate.py
├── LogRegrr_Baseline.py
├── config.yaml
├── processed/
├── results/
└── best_model.pt
```

---

## Conclusion

This project demonstrates:

* Graph-based models capture **inter-node dependencies**
* Temporal modeling enables **failure forecasting**
* Multi-step prediction reveals **cascade dynamics**

> The ST-GNN acts as a **cascade-aware failure detector**, prioritizing recall over precision which is a desirable trade-off in failure-critical cloud systems.

---

## References

* GraphSAGE (Hamilton et al., 2017)
* GRU (Cho et al., 2014)
* Google Borg Dataset
* Spatio-Temporal GNN literature
