\set ON_ERROR_STOP on
SET statement_timeout = '0';
SET default_transaction_read_only = on;

WITH landmark(subject_id, stay_id, t_hmis) AS (
  VALUES
  __LANDMARK_VALUES__
), drug_dictionary(itemid, drug_name) AS (
  VALUES
    (221906, 'norepinephrine'),
    (221289, 'epinephrine'),
    (221749, 'phenylephrine'),
    (222315, 'vasopressin'),
    (221662, 'dopamine')
)
SELECT l.subject_id,
       d.drug_name,
       ie.itemid,
       ie.starttime,
       ie.endtime,
       EXTRACT(EPOCH FROM (l.t_hmis-ie.starttime))/3600.0 AS hours_from_infusion_start_to_landmark,
       EXTRACT(EPOCH FROM (ie.endtime-l.t_hmis))/3600.0 AS hours_from_landmark_to_infusion_end,
       ie.rate,
       ie.rateuom,
       ie.amount,
       ie.amountuom
FROM landmark l
JOIN mimiciv_icu.inputevents ie
  ON ie.stay_id=l.stay_id
JOIN drug_dictionary d ON d.itemid=ie.itemid
WHERE coalesce(ie.rate,0)>0
  AND ie.endtime>ie.starttime+interval '5 minutes'
  AND coalesce(ie.statusdescription,'')<>'Rewritten'
  AND ie.starttime<=l.t_hmis
  AND ie.endtime>l.t_hmis-interval '6 hours'
ORDER BY l.subject_id, ie.starttime, ie.itemid;
