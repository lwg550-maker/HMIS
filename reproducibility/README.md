# Reproducibility package

This package contains the read-only extraction templates, deterministic landmark and outcome-state logic, covariate construction, multiple-imputation/model code, revision sensitivity analyses, figure code, and synthetic tests used for the manuscript.

## Privacy boundary

No MIMIC-IV patient-level extract, identifier value, date, free text, or row-level model input is included. The SQL templates use identifier column names because those fields are required to join MIMIC-IV tables, but no identifier values are embedded. All patient-level execution remains inside a credentialed MIMIC-IV environment.

## Analysis sequence

1. `gate_a_landmark/` and `src/hmis_liberation/gate_a.py`: read-only event extraction and sampled GCS-M landmark construction.
2. `gate_b_outcome/`, `src/hmis_liberation/gate_b.py`, `src/hmis_liberation/trach_evidence.py`, and `src/evidence_timeline.py`: evidence-aware IMV timeline and candidate-stop state machine.
3. `gate_c_covariates/` and `src/gate_c.py`: frozen 13-df covariate matrix and missingness audit.
4. `src/gate_d.py`: aggregate descriptive outputs and Aalen-Johansen estimates.
5. `gate_e_primary_models/` and `src/gate_e.py`: 50-imputation cause-specific Cox and Fine-Gray analyses.
6. `gate_f_sensitivity/` and `src/gate_f.py`: prespecified sensitivity analyses.
7. `revision_analyses/`: confirmation-time, PEEP-by-time, fixed-time cumulative-incidence, outcome-evidence, selection, and exposure audits added during revision.
8. `figure_generation/`: publication-figure generation and alignment checks.

## Frozen aggregate checks

The public tests assert the following aggregate identities without containing patient-level records: 474 parent patients; 221 sampled GCS-M transitions; 176 members of the conditional risk set and 45 nonentrants; 131 primary-rule-confirmed cessation starts, 40 competing deaths, and 5 administratively censored patients; and 101 full 168-hour plus 30 alive-discharge surrogate confirmations.

## Environment and execution

See `SOFTWARE_ENVIRONMENT.txt`, `DATA_DICTIONARY.md`, and `tests/test_frozen_assertions.py`. SQL must be run read-only against a locally authorized MIMIC-IV installation. Runtime SQL generated from patient-level seed files is intentionally excluded. The default multiple-imputation seed is 20261012; the revision analysis seed is 20261015 where applicable.

Run public synthetic tests from this directory with the project `src` and `scripts` directories on `PYTHONPATH`. Some historical test modules preserve their original import conventions; the top-level frozen-aggregate and revision-analysis tests are the release-gating tests.

