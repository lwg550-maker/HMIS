def test_frozen_cohort_flow():
    parent = 474
    transition = 221
    risk_set = 176
    nonentrant = 45
    assert transition + (parent - transition) == parent
    assert risk_set + nonentrant == transition


def test_primary_outcome_partition():
    event = 131
    death = 40
    censored = 5
    assert event + death + censored == 176


def test_confirmation_evidence_partition():
    full_168h = 101
    alive_discharge_surrogate = 30
    assert full_168h + alive_discharge_surrogate == 131


def test_path_events_are_not_outcome_partition():
    reinitiation_48h = 44
    reinitiation_7d = 61
    assert reinitiation_48h <= reinitiation_7d
    assert reinitiation_7d <= 176

