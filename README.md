# Evaluating Body Pose Grouping via Clustering Algorithms

![Conference](https://img.shields.io/badge/Presented_at-SPECS_Conference-blue)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Unsupervised_Clustering-orange)

> **Research Poster** presented at the School of Physics, Engineering and Computer Science [SPECS Conference](https://www.herts.ac.uk/research/specs-conference-2026), University of Hertfordshire.

---

## Overview
Training supervised classification models for action recognition (like fall-detection systems for the elderly) requires massive amounts of manually labeled data. This project investigates whether **unsupervised clustering algorithms** can automatically group continuous streams of human movement into discrete, identifiable poses, effectively automating the data annotation process.

Using Indian Classical Dance (Bharatanatyam) as a "ground-truth" dataset due to its strict, predefined geometric postures, we extracted 3D skeletal data via **MediaPipe**, built a custom heuristic filter to clean AI hallucinations, and evaluated three clustering algorithms: **K-Means, DBSCAN, and Agglomerative (Hierarchical) Clustering**.

---

## Repository Structure

* `Data/` - Contains the extracted coordinate streams (Raw and Filtered subsets).
* `Kmeans/` - Jupyter notebooks implementing K-Means clustering, including the Kneedle algorithm for Elbow Plot evaluation.
* `DBSCANCode/` - Implementation of DBSCAN using OPTICS reachability plots for density-based clustering.
* `AgglomCode/` - Hierarchical clustering scripts, including Dendrogram generation and distance thresholding.
* `3dPlottingCode/` - Visualisation scripts used to generate the 3D stick-figure pose grids and t-SNE/PCA scatter plots.

---


## Research Question
How effective are clustering algorithms for grouping raw body pose data extracted from continuous video streams?


## The Problem: Pose Hallucination
Standard pose estimation models (like MediaPipe) often "hallucinate" landmarks when limbs are occluded or out of frame. This project implements a **Dual-Stage Heuristic Filter** (Visibility and Geometric Constraints) to clean noisy coordinate streams before analysis.


## The Data Pipeline

### 1. Pose Extraction
We transcoded AV1 video to H.264 (59.94 FPS) and passed it through the **MediaPipe Pose Landmarker**, extracting 33 3D landmarks `(x, y, z)` and model confidence `(c)` per frame.

### 2. The "Pose Hallucination" Filter
MediaPipe frequently "hallucinates" leg coordinates when lower limbs are occluded or out-of-frame. To fix this, we built a custom dual-stage heuristic filter:
* **Visibility Filter:** Dropped frames where average ankle confidence was below 65% `(c_27 + c_28 / 2 < 0.65)`.
* **Geometric Constraint Filter:** Removed "collapsed skeletons" by ensuring the vertical hip-to-ankle distance was at least 18% of the frame height.

*Result:* Reduced a noisy dataset of 13,474 frames into a high-fidelity subset of 4,789 frames.


## Algorithms & Results

We scaled the 99-feature vectors (33 landmarks × 3 dimensions) using `StandardScaler` and applied the following algorithms:

1. **K-Means (Baseline):** * Used the Kneedle algorithm to identify an inflection point at `k=6` discrete poses.
2. **Agglomerative Clustering:** * Validated our `k=6` choice by generating a Dendrogram (distance threshold = 175), showing stable cluster branches between 5 and 7.
3. **DBSCAN (with OPTICS):** * **Key Finding:** DBSCAN was exceptionally effective at isolating the awkward, transitional frames between dance moves and identifying them as **"noise"** (Cluster -1). This kept the primary physical posture clusters incredibly pure compared to K-Means.

*(Note: t-SNE and PCA visualisations confirmed clear spatial separation of the resulting clusters).*


## Future Work
* **Spatial Normalization:** Implementing coordinate shifts to ensure identical poses performed in different parts of the video frame are clustered together.
* **Motion Features:** Integrating velocity and acceleration features to transition from static pose grouping to full-scale action recognition.

---

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/ml-mahendra/Evaluating-Body-Pose-Grouping-via-Clustering-Algorithms.git

2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn mediapipe matplotlib plotly opencv-python kneed
