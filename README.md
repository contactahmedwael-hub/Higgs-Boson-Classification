# Higgs Boson Classification

🔗 **Live demo:** [higgs-boson-classification.streamlit.app](https://higgs-boson-classification-cykjuusudmh4ugbfht2axm.streamlit.app/)

A machine learning project that tells apart **Higgs boson signal events** from **background events** in particle collision data from the ATLAS experiment at CERN.

## The problem
Each row is one collision event described by 30 measurements. The goal is to label every event as signal (`s`) or background (`b`). Signal is extremely rare compared with background, so the project is scored with a special metric, the **Approximate Median Significance (AMS)**, which uses event weights and punishes background events wrongly picked as signal. Plain accuracy is misleading here: calling every event signal gets 34% accuracy but an AMS of only 1.08.

## The data
- 250,000 labelled training events and 550,000 test events without labels.
- Every event has a **weight** that says how much it counts in the metric. Signal is 34% of the rows but only 0.17% of the total weight.
- The value `-999` means a measurement is undefined, mostly because the event has too few jets. It is information, not noise, so it was not replaced blindly.

## What I did
1. **Explored the data:** class counts versus weights, where `-999` appears, how features differ between signal and background, and which features are correlated.
2. **Prepared it carefully:** no leakage (everything learned is fitted on training data only), and `EventId`, `Weight` and `Label` are never used as features.
3. **Validated fairly:** one stratified train/validation split for all comparisons, then 5-fold cross-validation to check the results hold on other splits.
4. **Compared models:** a dummy baseline, Logistic Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, XGBoost and LightGBM.
5. **Ran focused experiments:** ways of handling `-999`, three ways of using event weights, engineered features, feature subsets, one model per jet group, and an ensemble.
6. **Chose the decision threshold for AMS:** instead of the usual 0.5 probability cut, I picked the share of events to call signal that gives the best AMS, using out-of-fold predictions only.

## Results
- **Final model:** XGBoost on the 30 original features, trained with class-balanced weights, calling the **top 15%** of events signal.
- **AMS: about 3.65** on out-of-fold data (individual folds: 3.52 to 3.80).

| Model | Best AMS (validation) |
|---|---|
| Logistic Regression | 2.08 |
| Decision Tree | 2.79 |
| Extra Trees | 3.01 |
| Random Forest | 3.39 |
| HistGradientBoosting | 3.56 |
| LightGBM | 3.60 |
| XGBoost | 3.62 |

## What I learned
- The most accurate model is not the best AMS model, so the metric has to guide every choice.
- Class-balanced weights worked best. Using the raw weights was the worst option.
- The threshold matters: the default 0.5 cut gave AMS 2.36 for one model, the best cut gave 3.56.
- Boosted trees beat linear models by a wide margin, but XGBoost, LightGBM and HistGradientBoosting are statistically tied.
- Engineered features, per-jet-group models and an ensemble gave no gain larger than the noise.

## Limitations
- The test labels are private, so the Kaggle leaderboard score cannot be computed here. The out-of-fold AMS is the best estimate.
- AMS is noisy because the total signal weight is small, so small differences between models are not reliable.
- The 15% cut was chosen on the same out-of-fold data used for the estimate, so 3.65 is slightly optimistic. Hyper-parameters were not tuned.

## Data sources and references
Kaggle "Higgs Boson Machine Learning Challenge" (https://www.kaggle.com/competitions/higgs-boson), CERN Open Data record 328 (https://opendata.cern.ch/record/328), and Adam-Bourdarios et al., *The Higgs boson machine learning challenge*, JMLR W&CP 42 (2015).
