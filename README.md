# Adaptive Neuro-Fuzzy Risk Scoring for Real-Time SIEM

## Overview
This project implements a novel **Adaptive Neuro-Fuzzy Inference System (ANFIS)** for real-time alert scoring in SIEM (Security Information and Event Management). By combining the feature representation learning of neural networks with the transparent reasoning of fuzzy logic, this system provides more accurate and interpretable risk scores than traditional models.

## Key Features
- **Hybrid Architecture**: Neural networks for feature learning + Fuzzy reasoning for risk assessment.
- **Real-Time Processing**: Designed for low-latency alert prioritization.
- **Explainable AI**: Fuzzy rules provide human-readable justification for risk scores.
- **Windows Optimized**: Built-in support for Windows Event Log patterns and behavioral metrics.

## Proposed Stack
- **Data Source**: Windows Event Logs
- **Dataset**: TII-SSRC-23 (for training and benchmarking)
- **Core Model**: Neuro-Fuzzy Network (ANFIS)
- **Output**: Real-time Risk-Based SIEM Dashboard

## Methodology

### Phase 1: Data Engineering
- **Parsing**: Advanced parsing of Windows logs.
- **Feature Extraction**:
    - **Categorical**: EventID, LogLevel, LogonType, Source.
    - **Numerical**: Failed login counts (sliding window), Session duration, Bytes transferred, Frequency metrics.
    - **Text**: Optional message embeddings using lightweight encoders.
- **Labeling**: Attack vs. Normal classification using TII-SSRC-23.

### Phase 2: Model Training
- **Neural Layer**: Train representation models to extract high-level features.
- **Fuzzy Layer**: Define fuzzy sets (e.g., `LoginFailureRate`: Low, Medium, High).
- **Integration**: Feed neural outputs into the neuro-fuzzy reasoning layer.

### Phase 3: Real-Time Simulation & Dashboard
- Sequentially stream logs to evaluate:
    - Detection latency.
    - False positive rates.
    - Alert prioritization quality.
- **Visualization**: An interactive SIEM dashboard for monitoring and response.

## Architecture
Detailed architecture specifications can be found in the [Architecture](Architecture) file. (Work in Progress)

## Getting Started
(Instructions to be added as implementation progresses)

## Dataset
This project uses the **TII-SSRC-23** dataset for benchmarking detection accuracy and pretraining neural representations.
