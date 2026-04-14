# ADHD EEG + Explainable Machine Learning

This project investigates whether electroencephalography (EEG) signals can be used to detect attention-deficit/hyperactivity disorder (ADHD) and, more importantly, explain which signal patterns drive the prediction. The goal is to combine predictive modeling with interpretable analysis to support reproducible neuroscience research.

## Project Overview

The project uses EEG features extracted from ADHD and control participants to build a binary classification model. A simple, interpretable baseline is established first, followed by a stronger nonlinear model for comparison. Explainability methods are then applied to identify the EEG features that most influence model decisions.

The resulting workflow supports two complementary outcomes:

- practical performance measurement for classification,
- and scientific interpretation of EEG markers associated with ADHD.

## Research and Industry Value

This project is designed to be useful in both settings:

- In research, it supports reproducible investigation of ADHD-related EEG patterns and feature importance.
- In industry, it demonstrates how to build a transparent, defensible ML pipeline for biosignal classification.

## Problem Statement

The central question is:

Can EEG-derived features distinguish ADHD from control subjects, and can the model explain why it makes those distinctions?

This is not only a prediction task. It is also an interpretability task that aims to identify meaningful neurophysiological patterns rather than treating the classifier as a black box.

## Data Source

The intended dataset is the IEEE DataPort ADHD EEG dataset. To keep the analysis credible, the preprocessing and feature extraction steps were applied consistently across all subjects so that the final model is trained on comparable EEG representations.

## Methodology

1. Preprocess EEG signals and extract structured features.
2. Train logistic regression as the baseline model.
3. Train HistGradientBoostingClassifier as the main nonlinear model.
4. Compare performance on the ADHD versus control classification task.
5. Use SHAP to explain feature contributions for the best-performing model.

Logistic regression provides a transparent reference point and helps establish whether the classification problem is linearly separable to some degree. HistGradientBoostingClassifier serves as a stronger model that can capture nonlinear relationships and interactions in the EEG feature space.

## Explainability Strategy

SHAP is the primary explainability method for this project. It can be used to show which EEG features most strongly push predictions toward ADHD or control, both globally and for individual samples.

This makes it possible to report findings such as:

- frontal theta and beta activity contributing to the decision boundary,
- altered band-power ratios being strongly associated with ADHD classification,
- or particular feature interactions shaping the model’s predictions.

An example interpretation might be:

The theta/beta ratio in frontal regions is a major driver of ADHD classification.

## Evaluation Approach

Model quality should be assessed using both predictive and interpretive criteria.

Predictive evaluation includes:

- accuracy,
- precision,
- recall,
- F1 score,
- and ROC-AUC.

Interpretive evaluation checks whether the most important features are stable across folds or repeated runs and whether they align with known EEG findings in ADHD literature.

## Expected Contributions

This project produces:

- a reproducible ADHD versus control EEG classification pipeline,
- a logistic regression baseline for transparent comparison,
- a higher-capacity HistGradientBoostingClassifier model,
- SHAP-based feature attributions,
- and a concise interpretation of the most influential EEG biomarkers.

## Suggested Next Steps

- Implement EEG preprocessing and feature extraction.
- Train and evaluate the logistic regression baseline.
- Train HistGradientBoostingClassifier as the primary model.
- Generate SHAP explanations for the final model.
- Summarize the most relevant EEG features and their interpretation.

## Repository Structure

## Project Structure

    adhd-eeg-explainability/
    ├─ README.md
    ├─ LICENSE
    ├─ data/
    │  └─ archive.zip
    ├─ notebooks/
    │  ├─ eda.ipynb
    │  ├─ feature_engineering.ipynb
    │  ├─ model_evaluation.ipynb
    │  └─ model_interpretation.ipynb
    ├─ reports/
    │  ├─eda/
    │  │  └─**.png
    │  ├─feature_engineering/
    │  │  ├─**.png
    │  │  └─**.png
    │  ├─model_evaluation/
    │  │  ├─learning_curves/
    │  │  │  ├─logistic_regression_lc.png
    │  │  │  └─hist_gradient_boosting_lc.png
    │  │  ├─validation_curves/
    │  │  │  ├─logistic_regression_vc.png
    │  │  │  └─hist_gradient_boosting_vc.png
    │  ├─ **pr_curve.png
    │  ├─ **roc_curve.png
    │  └─ **confusion_matrix.png
    ├─ models/
    │  └─**.pkl
    ├─ environment.yml
    ├─ requirements.txt
    └─ .gitignore

Add notebooks, scripts, figures, and results as the project develops.
