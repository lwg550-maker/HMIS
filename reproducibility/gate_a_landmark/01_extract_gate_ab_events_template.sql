\set ON_ERROR_STOP on
SET default_transaction_read_only = on;
SET statement_timeout = '60min';

-- MIMIC-IV v3.1 Gate A-B extraction.
-- The seed VALUES block is generated locally from the frozen 474-row cohort.
-- No persistent or temporary database object is created.
WITH seed(subject_id,hadm_id,index_stay_id,t0,index_outtime,index_deathtime) AS (
    VALUES
__SEED_VALUES__
), admission_context AS MATERIALIZED (
    SELECT s.*,
           a.admittime,
           a.dischtime,
           a.deathtime AS hospital_deathtime,
           a.discharge_location
    FROM seed s
    JOIN mimiciv_hosp.admissions a
      ON a.subject_id=s.subject_id
     AND a.hadm_id=s.hadm_id
), same_admission_icu AS MATERIALIZED (
    SELECT a.subject_id,a.hadm_id,a.index_stay_id,a.t0,
           i.stay_id AS source_stay_id
    FROM admission_context a
    JOIN mimiciv_icu.icustays i
      ON i.subject_id=a.subject_id
     AND i.hadm_id=a.hadm_id
), cohort_rows AS (
    SELECT (SELECT count(*) FROM seed)::bigint AS seed_n,
           a.subject_id,a.hadm_id,a.index_stay_id,a.t0,
           a.index_outtime,a.index_deathtime,a.admittime,a.dischtime,
           a.hospital_deathtime,a.discharge_location::text,
           'COHORT'::text AS event_type,
           a.index_stay_id::bigint AS source_stay_id,
           NULL::timestamp AS event_start,NULL::timestamp AS event_end,
           NULL::timestamp AS charttime,NULL::timestamp AS storetime,
           NULL::integer AS itemid,NULL::numeric AS value_num,
           NULL::text AS value_text,NULL::numeric AS rate,NULL::text AS rateuom
    FROM admission_context a
), gcs_rows AS (
    SELECT (SELECT count(*) FROM seed)::bigint AS seed_n,
           a.subject_id,a.hadm_id,a.index_stay_id,a.t0,
           a.index_outtime,a.index_deathtime,a.admittime,a.dischtime,
           a.hospital_deathtime,a.discharge_location::text,
           'GCS_M'::text AS event_type,
           ce.stay_id::bigint AS source_stay_id,
           ce.charttime AS event_start,ce.charttime AS event_end,
           ce.charttime,ce.storetime,ce.itemid,ce.valuenum::numeric AS value_num,
           ce.value::text AS value_text,NULL::numeric AS rate,NULL::text AS rateuom
    FROM admission_context a
    JOIN mimiciv_icu.chartevents ce
      ON ce.stay_id=a.index_stay_id
     AND ce.itemid=223901
     AND ce.valuenum BETWEEN 1 AND 6
     AND ce.charttime>a.t0-interval '4 hours'
     AND ce.charttime<=a.t0+interval '48 hours'
), nmb_rows AS (
    SELECT (SELECT count(*) FROM seed)::bigint AS seed_n,
           a.subject_id,a.hadm_id,a.index_stay_id,a.t0,
           a.index_outtime,a.index_deathtime,a.admittime,a.dischtime,
           a.hospital_deathtime,a.discharge_location::text,
           'NMB_INFUSION'::text AS event_type,
           ie.stay_id::bigint AS source_stay_id,
           ie.starttime AS event_start,ie.endtime AS event_end,
           NULL::timestamp AS charttime,NULL::timestamp AS storetime,
           ie.itemid,ie.amount::numeric AS value_num,
           ie.statusdescription::text AS value_text,
           ie.rate::numeric,ie.rateuom::text
    FROM admission_context a
    JOIN mimiciv_icu.inputevents ie
      ON ie.stay_id=a.index_stay_id
     AND ie.itemid IN (221555,229233,222062)
     AND ie.endtime>ie.starttime+interval '5 minutes'
     AND coalesce(ie.rate,0)>0
     AND ie.starttime<=a.t0+interval '48 hours'
     AND ie.endtime>a.t0
), ventilation_rows AS (
    SELECT (SELECT count(*) FROM seed)::bigint AS seed_n,
           a.subject_id,a.hadm_id,a.index_stay_id,a.t0,
           a.index_outtime,a.index_deathtime,a.admittime,a.dischtime,
           a.hospital_deathtime,a.discharge_location::text,
           CASE WHEN v.ventilation_status='InvasiveVent' THEN 'INVASIVE_VENT'
                WHEN v.ventilation_status='Tracheostomy' THEN 'TRACHEOSTOMY_STATUS'
                ELSE 'RESPIRATORY_SUPPORT' END::text AS event_type,
           i.source_stay_id::bigint,
           v.starttime AS event_start,v.endtime AS event_end,
           NULL::timestamp AS charttime,NULL::timestamp AS storetime,
           NULL::integer AS itemid,NULL::numeric AS value_num,
           v.ventilation_status::text AS value_text,
           NULL::numeric AS rate,NULL::text AS rateuom
    FROM admission_context a
    JOIN same_admission_icu i
      ON i.subject_id=a.subject_id AND i.hadm_id=a.hadm_id
    JOIN mimiciv_derived.ventilation v
      ON v.stay_id=i.source_stay_id
     AND v.endtime>a.t0-interval '4 hours'
     AND v.starttime<=least(a.dischtime,a.t0+interval '37 days')
), trach_oxygen_device_rows AS (
    SELECT (SELECT count(*) FROM seed)::bigint AS seed_n,
           a.subject_id,a.hadm_id,a.index_stay_id,a.t0,
           a.index_outtime,a.index_deathtime,a.admittime,a.dischtime,
           a.hospital_deathtime,a.discharge_location::text,
           'TRACH_OXYGEN_DEVICE'::text AS event_type,
           i.source_stay_id::bigint,
           od.charttime AS event_start,od.charttime AS event_end,
           od.charttime,NULL::timestamp AS storetime,
           NULL::integer AS itemid,NULL::numeric AS value_num,
           concat_ws(' | ',od.o2_delivery_device_1,od.o2_delivery_device_2,
                     od.o2_delivery_device_3,od.o2_delivery_device_4)::text AS value_text,
           NULL::numeric AS rate,NULL::text AS rateuom
    FROM admission_context a
    JOIN same_admission_icu i
      ON i.subject_id=a.subject_id AND i.hadm_id=a.hadm_id
    JOIN mimiciv_derived.oxygen_delivery od
      ON od.stay_id=i.source_stay_id
     AND (
          btrim(od.o2_delivery_device_1) IN ('Tracheostomy tube','Trach mask')
       OR btrim(od.o2_delivery_device_2) IN ('Tracheostomy tube','Trach mask')
       OR btrim(od.o2_delivery_device_3) IN ('Tracheostomy tube','Trach mask')
       OR btrim(od.o2_delivery_device_4) IN ('Tracheostomy tube','Trach mask')
     )
     AND od.charttime>a.t0-interval '4 hours'
     AND od.charttime<=least(a.dischtime,a.t0+interval '37 days')
), ventilator_setting_rows AS (
    SELECT (SELECT count(*) FROM seed)::bigint AS seed_n,
           a.subject_id,a.hadm_id,a.index_stay_id,a.t0,
           a.index_outtime,a.index_deathtime,a.admittime,a.dischtime,
           a.hospital_deathtime,a.discharge_location::text,
           'VENTILATOR_SETTING'::text AS event_type,
           i.source_stay_id::bigint,
           vs.charttime AS event_start,vs.charttime AS event_end,
           vs.charttime,NULL::timestamp AS storetime,
           NULL::integer AS itemid,vs.peep::numeric AS value_num,
           concat_ws(' | ',vs.ventilator_mode,vs.ventilator_mode_hamilton,
                     vs.ventilator_type)::text AS value_text,
           NULL::numeric AS rate,NULL::text AS rateuom
    FROM admission_context a
    JOIN same_admission_icu i
      ON i.subject_id=a.subject_id AND i.hadm_id=a.hadm_id
    JOIN mimiciv_derived.ventilator_setting vs
      ON vs.stay_id=i.source_stay_id
     AND coalesce(vs.ventilator_mode,vs.ventilator_mode_hamilton) IS NOT NULL
     AND vs.charttime>a.t0-interval '4 hours'
     AND vs.charttime<=least(a.dischtime,a.t0+interval '37 days')
), tracheostomy_rows AS (
    SELECT (SELECT count(*) FROM seed)::bigint AS seed_n,
           a.subject_id,a.hadm_id,a.index_stay_id,a.t0,
           a.index_outtime,a.index_deathtime,a.admittime,a.dischtime,
           a.hospital_deathtime,a.discharge_location::text,
           'TRACHEOSTOMY'::text AS event_type,
           i.source_stay_id::bigint,
           p.starttime AS event_start,p.endtime AS event_end,
           NULL::timestamp AS charttime,NULL::timestamp AS storetime,
           p.itemid,NULL::numeric AS value_num,d.label::text AS value_text,
           NULL::numeric AS rate,NULL::text AS rateuom
    FROM admission_context a
    JOIN same_admission_icu i
      ON i.subject_id=a.subject_id AND i.hadm_id=a.hadm_id
    JOIN mimiciv_icu.procedureevents p
      ON p.stay_id=i.source_stay_id
     AND p.itemid IN (225448,226237)
     AND p.starttime>a.t0-interval '4 hours'
     AND p.starttime<=least(a.dischtime,a.t0+interval '37 days')
    LEFT JOIN mimiciv_icu.d_items d ON d.itemid=p.itemid
), unioned AS (
    SELECT * FROM cohort_rows
    UNION ALL SELECT * FROM gcs_rows
    UNION ALL SELECT * FROM nmb_rows
    UNION ALL SELECT * FROM ventilation_rows
    UNION ALL SELECT * FROM trach_oxygen_device_rows
    UNION ALL SELECT * FROM ventilator_setting_rows
    UNION ALL SELECT * FROM tracheostomy_rows
)
SELECT *
FROM unioned
ORDER BY subject_id,
         CASE event_type WHEN 'COHORT' THEN 0 WHEN 'GCS_M' THEN 1
                         WHEN 'NMB_INFUSION' THEN 2 WHEN 'INVASIVE_VENT' THEN 3
                         WHEN 'TRACHEOSTOMY_STATUS' THEN 4
                         WHEN 'TRACH_OXYGEN_DEVICE' THEN 5
                         WHEN 'VENTILATOR_SETTING' THEN 6
                         WHEN 'RESPIRATORY_SUPPORT' THEN 7 ELSE 8 END,
         coalesce(charttime,event_start),storetime,event_end;
