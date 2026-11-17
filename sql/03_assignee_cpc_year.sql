-- Annual fractional assignee-by-CPC portfolios.
-- CPC features include all co-classifications attached to G06N families. This is
-- necessary to detect vertical application and hardware regimes. A G06N-only
-- indicator is retained for sensitivity analysis.

CREATE OR REPLACE TABLE `YOUR_PROJECT.YOUR_DATASET.assignee_cpc_year`
CLUSTER BY assignee, cpc_code AS
WITH expanded AS (
  SELECT
    family_id,
    publication_year,
    assignee,
    cpc_code,
    STARTS_WITH(cpc_code, 'G06N') AS is_g06n,
    1.0 / (ARRAY_LENGTH(assignees_harmonized) * ARRAY_LENGTH(all_cpc_codes)) AS fractional_family_count
  FROM `YOUR_PROJECT.YOUR_DATASET.family_master`
  CROSS JOIN UNNEST(assignees_harmonized) assignee
  CROSS JOIN UNNEST(all_cpc_codes) cpc_code
  WHERE ARRAY_LENGTH(assignees_harmonized) > 0
    AND ARRAY_LENGTH(all_cpc_codes) > 0
)
SELECT
  family_id,
  publication_year,
  assignee,
  cpc_code,
  is_g06n,
  fractional_family_count
FROM expanded;
