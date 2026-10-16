# HMIS ventilator-liberation study

Reproducible code and synthetic tests for the manuscript:

> Ventilator Liberation After a Recorded Motor-Response Transition in Acute Brain Injury: Associations With Recent Vasopressor Exposure in a Retrospective Landmark Cohort

## Scope

This repository contains code, extraction templates, prespecified dictionaries, analysis routines, figure-generation scripts, software versions, random seeds, aggregate assertions, and synthetic tests. It contains **no patient-level MIMIC-IV data, identifiers, clinical dates, free text, or model input rows**.

MIMIC-IV data are available only to credentialed users through PhysioNet under its data-use agreement. The code expects a local credentialed MIMIC-IV v3.1 installation and does not redistribute data.

## Reproducibility structure

- `reproducibility/sql/`: read-only extraction templates.
- `reproducibility/src/`: landmark, ventilation timeline, endpoint, imputation, and model logic.
- `reproducibility/revision_analyses/`: confirmation-time, proportional-hazards, selection, exposure, and measurement-bound analyses.
- `reproducibility/figure_generation/`: figure scripts and source alignment checks.
- `reproducibility/tests/`: synthetic tests and frozen aggregate assertions.
- `reproducibility/DATA_DICTIONARY.md`: operational definitions.
- `reproducibility/SOFTWARE_ENVIRONMENT.txt`: software and package versions.
- `reproducibility/SHA256SUMS.txt`: internal file digests.

## Frozen aggregate assertions

- 474 source patients → 221 sampled motor transitions → 176 patients in the conditional landmark risk set.
- 176 = 131 primary-rule-confirmed cessation starts + 40 competing deaths + 5 censored.
- 131 = 101 full 168-hour confirmations + 30 alive-discharge surrogate confirmations.

These assertions are aggregate quality-control targets, not patient-level data.

## Local verification

Create the environment described in `reproducibility/SOFTWARE_ENVIRONMENT.txt`, add `reproducibility/src` and `reproducibility/scripts` to `PYTHONPATH`, and run `pytest -q reproducibility/tests` from the repository root. The public release was verified with 48 tests passing and one credentialed-data integration test skipped by design.

## Citation

Use the metadata in `CITATION.cff`. Publication metadata should be updated after the article receives a DOI.

## License

MIT License. The license applies to code, not to MIMIC-IV data or third-party content.
