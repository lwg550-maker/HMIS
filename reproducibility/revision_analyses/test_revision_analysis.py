from pathlib import Path
import importlib.util
import sys

import pandas as pd


MODULE = Path(__file__).with_name("revision_analysis.py")
spec = importlib.util.spec_from_file_location("revision_analysis_tested", MODULE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_confirmation_time_state_machine():
    anchor = pd.Timestamp("2026-01-01")
    risk = pd.DataFrame({
        "subject_id": range(1, 8),
        "t_HMIS": [anchor] * 7,
        "death_time": [pd.NaT, pd.NaT, pd.NaT, anchor + pd.Timedelta(days=4), pd.NaT, anchor + pd.Timedelta(days=28), pd.NaT],
        "hospital_discharge": [pd.NaT, anchor + pd.Timedelta(days=3), pd.NaT, anchor + pd.Timedelta(days=4), pd.NaT, anchor + pd.Timedelta(days=28), pd.NaT],
    })
    # Pad to the frozen risk-set size; the function intentionally enforces n=176.
    padding = pd.DataFrame({
        "subject_id": range(8, 177), "t_HMIS": [anchor] * 169,
        "death_time": [pd.NaT] * 169, "hospital_discharge": [pd.NaT] * 169,
    })
    risk = pd.concat([risk, padding], ignore_index=True)
    candidates = pd.DataFrame([
        # full confirmation at day 8
        {"subject_id": 1, "candidate_sequence": 1, "candidate_stop": anchor + pd.Timedelta(days=1), "confirmation_end": anchor + pd.Timedelta(days=8), "candidate_result": "QUALIFIED_SUCCESS", "confirmation_source": "FULL_168H_OBSERVATION", "hospital_discharge": pd.NaT, "death_time": pd.NaT, "t_HMIS": anchor},
        # surrogate confirmation at discharge day 3
        {"subject_id": 2, "candidate_sequence": 1, "candidate_stop": anchor + pd.Timedelta(days=1), "confirmation_end": anchor + pd.Timedelta(days=8), "candidate_result": "QUALIFIED_SUCCESS", "confirmation_source": "ALIVE_DISCHARGE_SURROGATE", "hospital_discharge": anchor + pd.Timedelta(days=3), "death_time": pd.NaT, "t_HMIS": anchor},
        # failed first candidate, successful second
        {"subject_id": 3, "candidate_sequence": 1, "candidate_stop": anchor + pd.Timedelta(days=1), "confirmation_end": anchor + pd.Timedelta(days=8), "candidate_result": "FAILED_REINVASION", "confirmation_source": "NONE", "hospital_discharge": pd.NaT, "death_time": pd.NaT, "t_HMIS": anchor},
        {"subject_id": 3, "candidate_sequence": 2, "candidate_stop": anchor + pd.Timedelta(days=5), "confirmation_end": anchor + pd.Timedelta(days=12), "candidate_result": "QUALIFIED_SUCCESS", "confirmation_source": "FULL_168H_OBSERVATION", "hospital_discharge": pd.NaT, "death_time": pd.NaT, "t_HMIS": anchor},
        # confirmation after a same-time death: death wins
        {"subject_id": 4, "candidate_sequence": 1, "candidate_stop": anchor + pd.Timedelta(days=-3), "confirmation_end": anchor + pd.Timedelta(days=4), "candidate_result": "QUALIFIED_SUCCESS", "confirmation_source": "FULL_168H_OBSERVATION", "hospital_discharge": pd.NaT, "death_time": anchor + pd.Timedelta(days=4), "t_HMIS": anchor},
        # candidate after day 21 is excluded
        {"subject_id": 5, "candidate_sequence": 1, "candidate_stop": anchor + pd.Timedelta(days=22), "confirmation_end": anchor + pd.Timedelta(days=29), "candidate_result": "QUALIFIED_SUCCESS", "confirmation_source": "FULL_168H_OBSERVATION", "hospital_discharge": pd.NaT, "death_time": pd.NaT, "t_HMIS": anchor},
        # day 28 death competes
        {"subject_id": 6, "candidate_sequence": 1, "candidate_stop": anchor + pd.Timedelta(days=20), "confirmation_end": anchor + pd.Timedelta(days=27), "candidate_result": "FAILED_REINVASION", "confirmation_source": "NONE", "hospital_discharge": pd.NaT, "death_time": anchor + pd.Timedelta(days=28), "t_HMIS": anchor},
        # stop exactly day21, full confirmation exactly day28
        {"subject_id": 7, "candidate_sequence": 1, "candidate_stop": anchor + pd.Timedelta(days=21), "confirmation_end": anchor + pd.Timedelta(days=28), "candidate_result": "QUALIFIED_SUCCESS", "confirmation_source": "FULL_168H_OBSERVATION", "hospital_discharge": pd.NaT, "death_time": pd.NaT, "t_HMIS": anchor},
    ])
    out = module.derive_confirmation_time_outcome(risk, candidates).set_index("subject_id")
    assert out.loc[1, "followup_h"] == 8 * 24
    assert out.loc[2, "followup_h"] == 3 * 24
    assert out.loc[3, "followup_h"] == 12 * 24
    assert out.loc[4, "outcome"] == "DEATH_BEFORE_LIBERATION"
    assert out.loc[5, "outcome"] == "NO_LIBERATION_BY_28D"
    assert out.loc[6, "outcome"] == "DEATH_BEFORE_LIBERATION"
    assert out.loc[7, "outcome"] == "SUSTAINED_LIBERATION"
    assert out.loc[7, "followup_h"] == 672

