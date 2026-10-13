# Model Card: Hotel Cancellation Risk v2.0.0

## Model summary

This model estimates whether a hotel booking will be cancelled using information available
at booking creation. It is a shallow XGBoost binary classifier preceded by fold-fitted
imputation and one-hot encoding. Raw probabilities are adjusted with Platt scaling fitted on
a later validation window.

The project is a reproducible portfolio reference, not an autonomous decision system.

## Intended use

Suitable uses:

- prioritising bookings for low-impact reminder or confirmation workflows;
- demonstrating temporal ML evaluation, probability calibration, explainability, and drift;
- serving as a baseline for a property-specific model with current, governed data.

Out-of-scope uses:

- automatically cancelling, rejecting, repricing, or penalising a booking;
- making decisions about an individual without human review;
- deployment to a new hotel or region without local validation and recalibration;
- treating feature influence as evidence of causality.

## Prediction contract

- **Prediction time:** booking creation.
- **Target:** `is_canceled`.
- **Direct leakage removed:** `reservation_status`, `reservation_status_date`.
- **Post-booking fields removed:** `assigned_room_type`, `booking_changes`,
  `days_in_waiting_list`.
- **Output:** calibrated cancellation probability between 0 and 1.

The complete feature contract is stored in `artifacts/feature_contract.json` and enforced by
the shared training/inference feature builder.

## Data

The Hotel Booking Demand dataset contains 119,390 rows from a resort hotel and city hotel in
Portugal. It covers arrivals from July 2015 through August 2017.

The split clock is planned arrival date, the reliable event timestamp present in the source.
Lead time can approximate booking creation date, but the dataset does not provide a verified
creation timestamp. The evaluation therefore represents future arrival cohorts rather than a
fully delayed-label production replay.

There is no stable booking identifier and 31,994 rows are exact duplicates across every
source field. Those rows are removed before splitting so indistinguishable observations
cannot appear on both sides of evaluation. The resulting 87,396 rows are assigned by arrival
date:

| Split | Period | Rows | Purpose |
|---|---|---:|---|
| Train | 2015-07-01 to 2016-08-31 | 41,140 | fitting and temporal model selection |
| Validation | 2016-09-01 to 2016-12-31 | 14,564 | calibration and threshold selection |
| Test | 2017-01-01 to 2017-08-31 | 31,692 | one-time final evaluation |

## Model selection

Four candidates are compared with four expanding-window folds inside the training period.
All preprocessing is fitted separately inside each fold. PR-AUC is primary because it makes
performance on the cancellation class explicit under imbalance.

| Candidate | Mean PR-AUC | Mean ROC-AUC | Mean Brier |
|---|---:|---:|---:|
| XGBoost (shallow) | 0.692 | 0.864 | 0.120 |
| Random forest | 0.658 | 0.839 | 0.130 |
| XGBoost (regularised) | 0.656 | 0.857 | 0.123 |
| Logistic regression | 0.567 | 0.820 | 0.161 |

## Final evaluation

Metrics below come only from the untouched 2017 test window.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8319 (95% bootstrap CI 0.8278-0.8363) |
| PR-AUC | 0.6725 (95% bootstrap CI 0.6628-0.6823) |
| Raw / calibrated Brier | 0.1639 / 0.1591 |
| Log loss | 0.4769 |

The operational threshold of 0.1279 is selected on validation data with false negatives
costing four times false positives. On test it gives 97.24% recall, 45.53% precision, F2 of
0.7924, 11,764 false positives, and 279 false negatives. The low threshold is appropriate
only for inexpensive interventions; a costly action requires a different utility model.

## Explainability

Global influence is measured with native XGBoost SHAP contributions over a deterministic
5,000-row test sample. The strongest features are Portugal country membership, number of
special requests, agent code, lead time, requested parking spaces, market segment, customer
type, previous cancellations, and average daily rate.

These values describe associations used by the model. They do not establish causal drivers,
and one-hot feature importance can distribute or concentrate importance in unintuitive ways.

## Risks and limitations

- **Temporal drift:** monthly ROC-AUC falls to roughly 0.80 in July 2017 and Brier loss rises.
- **Evaluation clock:** arrival-cohort testing is not identical to replaying a live scoring
  system by booking-creation and label-availability timestamps.
- **Domain shift:** two historical Portuguese properties cannot represent other hotels.
- **Proxy risk:** country, agent, and channel may proxy geography or commercial access.
- **Duplicate policy:** exact rows may be legitimate separate bookings, but no identifier is
  available to distinguish them. Removal is the safer evaluation choice.
- **Intervention effects:** once staff act on scores, the observed target process can change.
- **Data quality:** impossible guest counts, ADR outliers, or unseen categories need upstream
  validation in a live system.

## Monitoring and retraining

A production owner should monitor monthly:

- input schema, missingness, unseen-category rate, and numerical range violations;
- score distribution and calibration by hotel, channel, and country group;
- PR-AUC, ROC-AUC, Brier score, recall, precision, and intervention volume after labels mature;
- material subgroup performance differences and complaint/override rates.

Suggested review triggers are a Brier increase above 15% relative to the approved baseline,
ROC-AUC below 0.78 for two consecutive mature periods, or an operational alert volume beyond
staff capacity. Any retrained model must repeat temporal validation and receive a new version.

## Reproducibility and ownership

- Training entry point: `python scripts/train_model.py`
- Frozen metadata: `artifacts/model_metadata.json`
- Regression fixture: `artifacts/score_regression.csv`
- CI: lint, compilation, tests, full rebuild, and notebook execution
- Model version: `2.0.0`
- Owner: Yang Li
