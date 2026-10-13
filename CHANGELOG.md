# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - 2026-10-13

### Added

- Chronological train, validation, and untouched test design.
- Four-fold expanding-window model comparison led by PR-AUC.
- Fold-local preprocessing, calibrated probabilities, and cost-aware threshold selection.
- Native XGBoost SHAP influence, bootstrap confidence intervals, and monthly drift evidence.
- Persisted model bundle, feature contract, split manifest, metadata, and score fixture.
- Reusable batch scoring API, automated contracts, and artifact regression tests.
- Model card and compact executable ML evidence notebook.

### Changed

- Reframed the repository from a static business report to a reproducible ML system.
- Replaced optimistic random-split scores with future-period performance.

### Removed

- Direct target leakage and features unavailable at booking creation.
- Exact duplicate rows that could cross evaluation boundaries without a booking identifier.
- Obsolete generated HTML/PDF reports containing the superseded evaluation.

## [1.0.0] - 2026-10-12

### Added

- Initial exploratory hotel booking analysis and classifier comparison.

[2.0.0]: https://github.com/liyang6620/hotel-booking-cancellation-analytics/releases/tag/v2.0.0
