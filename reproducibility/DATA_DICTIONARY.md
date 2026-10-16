# Public analysis dictionary

## Landmark items and timing

- GCS motor response: `mimiciv_icu.chartevents.itemid = 223901`.
- Continuous neuromuscular-blockade inputs: 221555, 229233, and 222062. Active continuous infusion makes the sampled motor response unknown; isolated boluses are not assigned an arbitrary washout interval.
- B0 and W1-W5 use the most recent value in the left-open, right-closed four-hour search interval ending at each nominal checkpoint.
- Equal chart times are resolved by the latest nonmissing store time; unresolved different values are unknown.
- A source observation can populate at most one checkpoint. Unknown checkpoints cannot be skipped to connect a low-to-high pair.
- `t_HMIS` is the actual chart time of the first eligible adjacent GCS-M 1-4 to 5-6 transition. The contemporaneous gate is `vent_start <= t_HMIS < vent_end`.

## Respiratory support

- The evidence-aware timeline combines bedside invasive-ventilation intervals with positive-pressure evidence during tracheostomy-labeled intervals.
- Tracheostomy is an airway state, not an outcome. Positive-pressure ventilation after tracheostomy remains IMV; tracheostomy collar or absence of positive-pressure evidence can permit a candidate stop.
- Noninvasive ventilation and high-flow nasal oxygen do not constitute IMV re-initiation.
- A candidate stop must occur strictly after `t_HMIS`. Re-initiation or death at or before 168 hours invalidates that candidate; a failed candidate does not remove the patient from risk.
- The main event time is the start of the first finally qualifying stop, confirmed by full 168-hour observation or the prespecified alive-discharge surrogate.

## Landmark covariate items

- Pupil items: 227288 and 227121; maximum four-hour lookback.
- PEEP source: `mimiciv_derived.ventilator_setting`, most recent in six hours.
- FiO2: item 223835. PaO2/FiO2 pairing uses a maximum two-hour separation and a 24-hour lookback.
- RASS auxiliary item: 228096.

## Continuous sedatives active at landmark

- Propofol 222168.
- Dexmedetomidine 225150 or 229420.
- Midazolam 221668.
- Lorazepam 221385.
- Ketamine 221712.
- Active means that a rate greater than zero covers `t_HMIS` for more than five minutes.

## Vasopressors in the six hours before landmark

- Norepinephrine 221906.
- Epinephrine 221289.
- Phenylephrine 221749.
- Vasopressin 222315.
- Dopamine 221662.
- The manuscript exposure is binary: any qualifying infusion overlap in `(t_HMIS - 6 h, t_HMIS]`. Drug-specific counts and overlaps are descriptive; no data-driven dose threshold was used.

## Frozen 13-df design

Age per decade (1); recorded sex (1); mutually exclusive diagnostic phenotype with TBI reference and SAH, ICH, SDH/EDH indicators (3); log2(1 + current IMV hours) (1); GCS-M 6 versus 5 (1); bilateral reactive pupils versus any abnormality (1); continuous major sedative active (1); recent vasopressor exposure (1); P/F ratio per 50 units (1); PEEP per 5 cmH2O (1); and ordinal anchor year group 0-4 (1).

