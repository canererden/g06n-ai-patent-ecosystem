-- H1-H3 derivative tables for the US-grant sensitivity population.

CREATE OR REPLACE TABLE `YOUR_PROJECT.YOUR_DATASET.citation_edges_us_grants`
CLUSTER BY citing_assignee, cited_assignee AS
WITH core AS (
  SELECT * FROM `YOUR_PROJECT.YOUR_DATASET.family_master_us_grants`
  WHERE ARRAY_LENGTH(assignees_harmonized) > 0
),
raw_citations AS (
  SELECT DISTINCT
    core.family_id AS citing_family_id,
    core.publication_year AS citing_year,
    COALESCE(NULLIF(cited.family_id, ''), cited.publication_number) AS cited_family_id
  FROM core
  CROSS JOIN UNNEST(core.publication_numbers) source_publication_number
  JOIN `patents-public-data.patents.publications` source
    ON source.publication_number = source_publication_number
  CROSS JOIN UNNEST(source.citation) cit
  JOIN `patents-public-data.patents.publications` cited
    ON cited.publication_number = cit.publication_number
  WHERE cit.publication_number IS NOT NULL
    AND core.family_id != COALESCE(NULLIF(cited.family_id, ''), cited.publication_number)
),
cited_family_assignees AS (
  SELECT
    COALESCE(NULLIF(p.family_id, ''), p.publication_number) AS family_id,
    ARRAY_AGG(DISTINCT UPPER(TRIM(a.name)) IGNORE NULLS) AS assignees
  FROM `patents-public-data.patents.publications` p
  CROSS JOIN UNNEST(p.assignee_harmonized) a
  WHERE NULLIF(TRIM(a.name), '') IS NOT NULL
  GROUP BY family_id
),
expanded AS (
  SELECT
    r.citing_family_id,
    r.cited_family_id,
    r.citing_year,
    citing_assignee,
    cited_assignee,
    1.0 / (ARRAY_LENGTH(c.assignees_harmonized) * ARRAY_LENGTH(d.assignees)) AS fractional_weight
  FROM raw_citations r
  JOIN core c ON c.family_id = r.citing_family_id
  JOIN cited_family_assignees d ON d.family_id = r.cited_family_id
  CROSS JOIN UNNEST(c.assignees_harmonized) citing_assignee
  CROSS JOIN UNNEST(d.assignees) cited_assignee
)
SELECT * FROM expanded WHERE citing_assignee != cited_assignee;

CREATE OR REPLACE TABLE `YOUR_PROJECT.YOUR_DATASET.assignee_cpc_year_us_grants`
CLUSTER BY assignee, cpc_code AS
SELECT
  family_id,
  publication_year,
  assignee,
  cpc_code,
  STARTS_WITH(cpc_code, 'G06N') AS is_g06n,
  1.0 / (ARRAY_LENGTH(assignees_harmonized) * ARRAY_LENGTH(all_cpc_codes)) AS fractional_family_count
FROM `YOUR_PROJECT.YOUR_DATASET.family_master_us_grants`
CROSS JOIN UNNEST(assignees_harmonized) assignee
CROSS JOIN UNNEST(all_cpc_codes) cpc_code
WHERE ARRAY_LENGTH(assignees_harmonized) > 0
  AND ARRAY_LENGTH(all_cpc_codes) > 0;
