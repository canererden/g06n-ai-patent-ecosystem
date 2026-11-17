-- Deterministic equal-per-year sample for topic modelling.
-- A balanced sample supports temporal comparisons but is not population-proportional;
-- annual prevalence estimates must therefore be calculated within year.

DECLARE abstracts_per_year INT64 DEFAULT 5000;
DECLARE seed STRING DEFAULT '266665684';

CREATE OR REPLACE TABLE `YOUR_PROJECT.YOUR_DATASET.abstract_sample`
CLUSTER BY publication_year AS
WITH eligible AS (
  SELECT
    family_id,
    publication_number,
    publication_year,
    publication_date,
    priority_date,
    country_code,
    title_en,
    abstract_en,
    FARM_FINGERPRINT(CONCAT(seed, '|', family_id)) AS deterministic_order
  FROM `YOUR_PROJECT.YOUR_DATASET.family_master`
  WHERE abstract_en IS NOT NULL
    AND LENGTH(TRIM(abstract_en)) >= 100
),
ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY publication_year
      ORDER BY deterministic_order, family_id
    ) AS sample_rank,
    COUNT(*) OVER (PARTITION BY publication_year) AS eligible_abstracts_in_year
  FROM eligible
)
SELECT *
FROM ranked
WHERE sample_rank <= abstracts_per_year;
