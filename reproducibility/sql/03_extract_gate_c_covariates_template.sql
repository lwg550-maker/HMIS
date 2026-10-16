SET default_transaction_read_only = on;
SET statement_timeout = '30min';

-- Runtime seed is generated from the accepted 176-patient Gate A risk set.
-- The generated SQL is restricted because the VALUES clause contains MIMIC IDs.
WITH seed(subject_id,hadm_id,stay_id,t_hmis,landmark_gcsm,target_window,
          active_vent_start,active_vent_end) AS MATERIALIZED (
    VALUES
__SEED_VALUES__
),
dx AS MATERIALIZED (
    SELECT d.hadm_id,
           bool_or(
               (d.icd_version=9 AND left(upper(regexp_replace(d.icd_code,'[^A-Z0-9]','','g')),3)
                    IN ('850','851','852','853','854'))
               OR
               (d.icd_version=10
                AND upper(regexp_replace(d.icd_code,'[^A-Z0-9]','','g')) LIKE 'S06%'
                AND (length(regexp_replace(d.icd_code,'[^A-Z0-9]','','g')) < 7
                     OR right(upper(regexp_replace(d.icd_code,'[^A-Z0-9]','','g')),1)='A'))
           ) AS dx_tbi,
           bool_or((d.icd_version=9 AND left(upper(d.icd_code),3)='431')
                   OR (d.icd_version=10 AND left(upper(d.icd_code),3)='I61')) AS dx_ich,
           bool_or((d.icd_version=9 AND left(upper(d.icd_code),3)='430')
                   OR (d.icd_version=10 AND left(upper(d.icd_code),3)='I60')) AS dx_sah,
           bool_or((d.icd_version=9 AND upper(regexp_replace(d.icd_code,'[^A-Z0-9]','','g'))
                                      IN ('4320','4321'))
                   OR (d.icd_version=10 AND left(upper(regexp_replace(d.icd_code,'[^A-Z0-9]','','g')),4)
                                      IN ('I620','I621'))) AS dx_sdh_edh
    FROM mimiciv_hosp.diagnoses_icd d
    JOIN (SELECT DISTINCT hadm_id FROM seed) s USING (hadm_id)
    GROUP BY d.hadm_id
),
base AS MATERIALIZED (
    SELECT s.*, p.gender AS recorded_sex, p.anchor_year_group,
           p.anchor_age + extract(year FROM a.admittime)::integer - p.anchor_year AS age,
           a.admittime,a.dischtime,a.deathtime,a.discharge_location,
           coalesce(dx.dx_sah,false) AS dx_sah,
           coalesce(dx.dx_ich,false) AS dx_ich,
           coalesce(dx.dx_sdh_edh,false) AS dx_sdh_edh,
           coalesce(dx.dx_tbi,false) AS dx_tbi,
           CASE WHEN dx.dx_sah THEN 'SAH'
                WHEN dx.dx_ich THEN 'ICH'
                WHEN dx.dx_sdh_edh THEN 'SDH_EDH'
                WHEN dx.dx_tbi THEN 'TBI' END AS diagnosis_phenotype
    FROM seed s
    JOIN mimiciv_hosp.admissions a
      ON a.hadm_id=s.hadm_id AND a.subject_id=s.subject_id
    JOIN mimiciv_hosp.patients p ON p.subject_id=s.subject_id
    LEFT JOIN dx ON dx.hadm_id=s.hadm_id
)
SELECT b.subject_id,b.hadm_id,b.stay_id,b.t_hmis,b.landmark_gcsm,b.target_window,
       b.active_vent_start,b.active_vent_end,
       extract(epoch FROM (b.t_hmis-b.active_vent_start))/3600.0 AS current_invasive_segment_h,
       b.age,b.recorded_sex,b.anchor_year_group,
       b.dx_sah::integer AS dx_sah,b.dx_ich::integer AS dx_ich,
       b.dx_sdh_edh::integer AS dx_sdh_edh,b.dx_tbi::integer AS dx_tbi,
       b.diagnosis_phenotype,
       pu.pupil_time,pu.pupil_left_raw,pu.pupil_right_raw,
       pu.pupil_bilateral_reactive,pu.pupil_any_abnormal,
       extract(epoch FROM (b.t_hmis-pu.pupil_time))/3600.0 AS pupil_staleness_h,
       se.propofol_active,se.dexmedetomidine_active,se.midazolam_active,
       se.lorazepam_active,se.ketamine_active,se.major_sedative_active,
       se.active_sedative_drug_n,
       va.norepinephrine_exposed_6h,va.epinephrine_exposed_6h,
       va.phenylephrine_exposed_6h,va.vasopressin_exposed_6h,
       va.dopamine_exposed_6h,va.any_vasopressor_exposed_6h,
       pf.pao2,pf.pao2_time,pf.fio2_fraction AS pf_fio2_fraction,
       pf.fio2_time AS pf_fio2_time,pf.pf_pair_gap_min,pf.pf_ratio,
       extract(epoch FROM (b.t_hmis-greatest(pf.pao2_time,pf.fio2_time)))/3600.0 AS pf_pair_staleness_h,
       pe.peep,pe.peep_time,
       extract(epoch FROM (b.t_hmis-pe.peep_time))/3600.0 AS peep_staleness_h,
       af.fio2_fraction AS auxiliary_fio2,af.fio2_time AS auxiliary_fio2_time,
       rr.resp_rate AS auxiliary_resp_rate,rr.resp_rate_time AS auxiliary_resp_rate_time,
       ra.rass AS auxiliary_rass,ra.rass_time AS auxiliary_rass_time,
       cr.creatinine AS auxiliary_creatinine,cr.creatinine_time AS auxiliary_creatinine_time,
       wb.wbc AS auxiliary_wbc,wb.wbc_time AS auxiliary_wbc_time
FROM base b
LEFT JOIN LATERAL (
    WITH recognized AS (
        SELECT ce.charttime,ce.itemid,ce.value,ce.storetime,
               CASE
                 WHEN lower(ce.value) ~ '(non.?react|fixed)' THEN 0
                 WHEN lower(ce.value) ~ '(brisk|sluggish|reactive)'
                      AND lower(ce.value) !~ '(non.?react|fixed)' THEN 1
               END AS response,
               max(ce.storetime) FILTER (WHERE ce.storetime IS NOT NULL)
                 OVER (PARTITION BY ce.charttime,ce.itemid) AS latest_storetime
        FROM mimiciv_icu.chartevents ce
        WHERE ce.stay_id=b.stay_id AND ce.itemid IN (227288,227121)
          AND ce.charttime>b.t_hmis-interval '4 hours' AND ce.charttime<=b.t_hmis
          AND ce.value IS NOT NULL
    ), chosen AS (
        SELECT charttime,itemid,max(value) AS raw_value,
               CASE WHEN count(DISTINCT response) FILTER (WHERE response IS NOT NULL)=1
                    THEN min(response) FILTER (WHERE response IS NOT NULL) END AS response
        FROM recognized
        WHERE (latest_storetime IS NULL AND storetime IS NULL)
           OR storetime=latest_storetime
        GROUP BY charttime,itemid
    ), paired AS (
        SELECT charttime,
               max(raw_value) FILTER (WHERE itemid=227288) AS left_raw,
               max(raw_value) FILTER (WHERE itemid=227121) AS right_raw,
               max(response) FILTER (WHERE itemid=227288) AS left_response,
               max(response) FILTER (WHERE itemid=227121) AS right_response
        FROM chosen GROUP BY charttime
    )
    SELECT charttime AS pupil_time,left_raw AS pupil_left_raw,right_raw AS pupil_right_raw,
           ((left_response=1) AND (right_response=1))::integer AS pupil_bilateral_reactive,
           ((left_response=0) OR (right_response=0))::integer AS pupil_any_abnormal
    FROM paired
    WHERE left_response IS NOT NULL AND right_response IS NOT NULL
    ORDER BY charttime DESC LIMIT 1
) pu ON true
LEFT JOIN LATERAL (
    SELECT coalesce(bool_or(ie.itemid=222168),false)::integer AS propofol_active,
           coalesce(bool_or(ie.itemid IN (225150,229420)),false)::integer AS dexmedetomidine_active,
           coalesce(bool_or(ie.itemid=221668),false)::integer AS midazolam_active,
           coalesce(bool_or(ie.itemid=221385),false)::integer AS lorazepam_active,
           coalesce(bool_or(ie.itemid=221712),false)::integer AS ketamine_active,
           (count(*)>0)::integer AS major_sedative_active,
           count(DISTINCT CASE WHEN ie.itemid=222168 THEN 'propofol'
                               WHEN ie.itemid IN (225150,229420) THEN 'dexmedetomidine'
                               WHEN ie.itemid=221668 THEN 'midazolam'
                               WHEN ie.itemid=221385 THEN 'lorazepam'
                               WHEN ie.itemid=221712 THEN 'ketamine' END)::integer AS active_sedative_drug_n
    FROM mimiciv_icu.inputevents ie
    WHERE ie.stay_id=b.stay_id
      AND ie.itemid IN (222168,225150,229420,221668,221385,221712)
      AND coalesce(ie.rate,0)>0
      AND ie.endtime>ie.starttime+interval '5 minutes'
      AND coalesce(ie.statusdescription,'')<>'Rewritten'
      AND ie.starttime<=b.t_hmis AND ie.endtime>b.t_hmis
) se ON true
LEFT JOIN LATERAL (
    SELECT coalesce(bool_or(ie.itemid=221906),false)::integer AS norepinephrine_exposed_6h,
           coalesce(bool_or(ie.itemid=221289),false)::integer AS epinephrine_exposed_6h,
           coalesce(bool_or(ie.itemid=221749),false)::integer AS phenylephrine_exposed_6h,
           coalesce(bool_or(ie.itemid=222315),false)::integer AS vasopressin_exposed_6h,
           coalesce(bool_or(ie.itemid=221662),false)::integer AS dopamine_exposed_6h,
           (count(*)>0)::integer AS any_vasopressor_exposed_6h
    FROM mimiciv_icu.inputevents ie
    WHERE ie.stay_id=b.stay_id
      AND ie.itemid IN (221906,221289,221749,222315,221662)
      AND coalesce(ie.rate,0)>0
      AND ie.endtime>ie.starttime+interval '5 minutes'
      AND coalesce(ie.statusdescription,'')<>'Rewritten'
      AND ie.starttime<=b.t_hmis AND ie.endtime>b.t_hmis-interval '6 hours'
) va ON true
LEFT JOIN LATERAL (
    SELECT q.pao2,q.pao2_time,q.fio2_fraction,q.fio2_time,q.pf_pair_gap_min,
           q.pao2/q.fio2_fraction AS pf_ratio
    FROM (
        SELECT bg.po2 AS pao2,bg.charttime AS pao2_time,
               CASE WHEN ce.valuenum BETWEEN 21 AND 100 THEN ce.valuenum/100.0
                    WHEN ce.valuenum BETWEEN 0.21 AND 1.0 THEN ce.valuenum END AS fio2_fraction,
               ce.charttime AS fio2_time,
               abs(extract(epoch FROM (ce.charttime-bg.charttime)))/60.0 AS pf_pair_gap_min
        FROM mimiciv_derived.bg bg
        JOIN mimiciv_icu.chartevents ce
          ON ce.stay_id=b.stay_id AND ce.itemid=223835
         AND ce.charttime>b.t_hmis-interval '24 hours' AND ce.charttime<=b.t_hmis
         AND abs(extract(epoch FROM (ce.charttime-bg.charttime)))<=7200
         AND (ce.valuenum BETWEEN 21 AND 100 OR ce.valuenum BETWEEN 0.21 AND 1.0)
        WHERE bg.subject_id=b.subject_id AND bg.hadm_id=b.hadm_id
          AND bg.specimen='ART.' AND bg.po2 BETWEEN 20 AND 800
          AND bg.charttime>b.t_hmis-interval '24 hours' AND bg.charttime<=b.t_hmis
    ) q
    WHERE q.fio2_fraction BETWEEN 0.21 AND 1.0
    ORDER BY greatest(q.pao2_time,q.fio2_time) DESC,
             q.pf_pair_gap_min ASC,q.pao2_time DESC,q.fio2_time DESC
    LIMIT 1
) pf ON true
LEFT JOIN LATERAL (
    SELECT vs.peep,vs.charttime AS peep_time
    FROM mimiciv_derived.ventilator_setting vs
    WHERE vs.stay_id=b.stay_id AND vs.peep IS NOT NULL
      AND vs.charttime>b.t_hmis-interval '6 hours' AND vs.charttime<=b.t_hmis
    ORDER BY vs.charttime DESC LIMIT 1
) pe ON true
LEFT JOIN LATERAL (
    SELECT CASE WHEN ce.valuenum BETWEEN 21 AND 100 THEN ce.valuenum/100.0
                WHEN ce.valuenum BETWEEN 0.21 AND 1.0 THEN ce.valuenum END AS fio2_fraction,
           ce.charttime AS fio2_time
    FROM mimiciv_icu.chartevents ce
    WHERE ce.stay_id=b.stay_id AND ce.itemid=223835
      AND ce.charttime>b.t_hmis-interval '24 hours' AND ce.charttime<=b.t_hmis
      AND (ce.valuenum BETWEEN 21 AND 100 OR ce.valuenum BETWEEN 0.21 AND 1.0)
    ORDER BY ce.charttime DESC,ce.storetime DESC NULLS LAST LIMIT 1
) af ON true
LEFT JOIN LATERAL (
    SELECT v.resp_rate,v.charttime AS resp_rate_time
    FROM mimiciv_derived.vitalsign v
    WHERE v.stay_id=b.stay_id AND v.resp_rate>0
      AND v.charttime>b.t_hmis-interval '24 hours' AND v.charttime<=b.t_hmis
    ORDER BY v.charttime DESC LIMIT 1
) rr ON true
LEFT JOIN LATERAL (
    SELECT ce.valuenum AS rass,ce.charttime AS rass_time
    FROM mimiciv_icu.chartevents ce
    WHERE ce.stay_id=b.stay_id AND ce.itemid=228096 AND ce.valuenum BETWEEN -5 AND 4
      AND ce.charttime>b.t_hmis-interval '24 hours' AND ce.charttime<=b.t_hmis
    ORDER BY ce.charttime DESC,ce.storetime DESC NULLS LAST LIMIT 1
) ra ON true
LEFT JOIN LATERAL (
    SELECT c.creatinine,c.charttime AS creatinine_time
    FROM mimiciv_derived.chemistry c
    WHERE c.subject_id=b.subject_id AND c.hadm_id=b.hadm_id AND c.creatinine IS NOT NULL
      AND c.charttime>b.t_hmis-interval '24 hours' AND c.charttime<=b.t_hmis
    ORDER BY c.charttime DESC LIMIT 1
) cr ON true
LEFT JOIN LATERAL (
    SELECT c.wbc,c.charttime AS wbc_time
    FROM mimiciv_derived.complete_blood_count c
    WHERE c.subject_id=b.subject_id AND c.hadm_id=b.hadm_id AND c.wbc IS NOT NULL
      AND c.charttime>b.t_hmis-interval '24 hours' AND c.charttime<=b.t_hmis
    ORDER BY c.charttime DESC LIMIT 1
) wb ON true
ORDER BY b.subject_id;
