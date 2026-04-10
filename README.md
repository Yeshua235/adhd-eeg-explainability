# ADHD EEG + Explainable Machine Learning

This project focuses on predicting ADHD from EEG data while also explaining why the model makes its decisions. Instead of treating the model as a black box, the goal is to identify which EEG features are most responsible for ADHD versus control classifications and translate those findings into interpretable neuroscience insights.

## Project Goal

The aim is to build a clear baseline classifier for ADHD detection and then use explainability methods to determine which EEG patterns drive predictions. This creates a workflow that is useful both for performance and for scientific interpretation.

## Core Idea

The project asks two questions at the same time:

1. Can EEG features distinguish ADHD from control participants?
2. Which features are actually influencing the model’s decisions?

That second question is the key difference from a standard classification task.

## Data Source

The intended dataset is the IEEE DataPort ADHD EEG dataset. The analysis should use a consistent preprocessing pipeline so that model outputs and explanations are based on comparable signals across participants.

## Modeling Approach

The recommended workflow is:

1. Preprocess EEG signals and extract structured features.
2. Train logistic regression as the baseline model.
3. Train a stronger nonlinear model such as HistGradientBoostingClassifier for comparison.
4. Evaluate ADHD versus control performance.
5. Apply SHAP to explain feature contributions for the boosted model.

Logistic regression provides a simple and interpretable reference point, while the boosted tree model can capture more complex relationships in the EEG features. SHAP then helps translate the boosted model’s predictions into feature-level explanations.

## Explainability Focus

SHAP is used to show which EEG features most influenced a prediction. This allows the project to move beyond raw accuracy and toward interpretable findings such as:

- frontal theta/beta abnormalities,
- altered band-power ratios,
- or feature combinations that consistently push predictions toward ADHD.

An example result might be:

“Theta/beta ratio in the frontal cortex is a major driver of ADHD classification.”

## Potential Insights

This project can support several useful interpretations:

- which EEG features are most predictive of ADHD,
- whether those features align with known neurophysiological theory,
- and whether the classifier is relying on clinically meaningful patterns rather than noise.

If the explanations are stable across folds or subjects, the project becomes more credible as a neuroscience result rather than only a machine learning benchmark.

## Expected Outputs

Depending on the final implementation, this repository may produce:

- a trained logistic regression baseline and a boosted tree classifier,
- model performance metrics,
- SHAP summary plots and feature importance rankings,
- and a concise interpretation of the most influential EEG markers.

## Project Status

This repository is a research workspace for exploratory analysis and should be adapted to the feature engineering and evaluation strategy used in the final study.

## Next Steps

- Add the preprocessing and feature extraction pipeline.
- Train a logistic regression baseline model.
- Train HistGradientBoostingClassifier as the main comparison model.
- Generate SHAP explanations for the best-performing model.
- Document the most important EEG features and their interpretation.

## Repository Structure

- `README.md`: project overview and research framing.

Add notebooks, scripts, and results as the project develops.
