-- Google BigQuery Standard SQL
-- Primary population: one English-language representative per simple family
-- having at least one G06N CPC code and published during 2018-2025.
-- Replace YOUR_PROJECT.YOUR_DATASET with a writable destination.

DECLARE start_date INT64 DEFAULT 20180101;
DECLARE end_date INT64 DEFAULT 20251231;

CREATE OR REPLACE TABLE `YOUR_PROJECT.YOUR_DATASET.family_master`
PARTITION BY RANGE_BUCKET(publication_year, GENERATE_ARRAY(2018, 2026, 1))
CLUSTER BY country_code, family_id AS
WITH eligible AS (
  SELECT
    p.publication_number,
    p.application_number,
    COALESCE(NULLIF(p.family_id, ''), p.publication_number) AS family_id,
    p.country_code,
    p.kind_code,
    p.publication_date,
    DIV(p.publication_date, 10000) AS publication_year,
    p.filing_date,
    p.grant_date,
    p.priority_date,
    (SELECT t.text FROM UNNEST(p.title_localized) t
      WHERE t.language = 'en' AND t.text IS NOT NULL LIMIT 1) AS title_en,
    (SELECT a.text FROM UNNEST(p.abstract_localized) a
      WHERE a.language = 'en' AND a.text IS NOT NULL LIMIT 1) AS abstract_en,
    ARRAY(SELECT DISTINCT UPPER(TRIM(a.name))
          FROM UNNEST(p.assignee_harmonized) a
          WHERE NULLIF(TRIM(a.name), '') IS NOT NULL
          ORDER BY 1) AS assignees_harmonized,
    ARRAY(SELECT DISTINCT REGEXP_REPLACE(UPPER(c.code), r'\s+', '')
          FROM UNNEST(p.cpc) c
          WHERE STARTS_WITH(REGEXP_REPLACE(UPPER(c.code), r'\s+', ''), 'G06N')
          ORDER BY 1) AS g06n_cpc_codes,
    ARRAY(SELECT DISTINCT REGEXP_REPLACE(UPPER(c.code), r'\s+', '')
          FROM UNNEST(p.cpc) c
          WHERE NULLIF(TRIM(c.code), '') IS NOT NULL
          ORDER BY 1) AS all_cpc_codes,
    ARRAY_LENGTH(p.citation) AS patent_citation_count
  FROM `patents-public-data.patents.publications` p
  WHERE p.publication_date BETWEEN start_date AND end_date
    AND EXISTS (
      SELECT 1 FROM UNNEST(p.cpc) c
      WHERE STARTS_WITH(REGEXP_REPLACE(UPPER(c.code), r'\s+', ''), 'G06N')
    )
),
aggregated AS (
  SELECT
    family_id,
    MIN(publication_date) AS first_publication_date,
    ARRAY_AGG(STRUCT(
        publication_number, application_number, country_code, kind_code,
        publication_date, publication_year, filing_date, grant_date, priority_date,
        title_en, abstract_en, patent_citation_count
      )
      ORDER BY
        IF(abstract_en IS NOT NULL, 0, 1),
        IF(title_en IS NOT NULL, 0, 1),
        IF(grant_date > 0, 0, 1),
        publication_date,
        publication_number
      LIMIT 1)[OFFSET(0)] AS representative,
    ARRAY_AGG(publication_number ORDER BY publication_number) AS publication_numbers,
    ARRAY_CONCAT_AGG(assignees_harmonized) AS assignees_concat,
    ARRAY_CONCAT_AGG(g06n_cpc_codes) AS g06n_cpc_concat,
    ARRAY_CONCAT_AGG(all_cpc_codes) AS all_cpc_concat,
    COUNT(*) AS publications_in_family
  FROM eligible
  GROUP BY family_id
)
SELECT
  representative.publication_number,
  representative.application_number,
  representative.country_code,
  representative.kind_code,
  first_publication_date AS publication_date,
  DIV(first_publication_date, 10000) AS publication_year,
  representative.filing_date,
  representative.grant_date,
  representative.priority_date,
  representative.title_en,
  representative.abstract_en,
  representative.patent_citation_count,
  family_id,
  publication_numbers,
  ARRAY(SELECT DISTINCT value FROM UNNEST(assignees_concat) value ORDER BY value) AS assignees_harmonized,
  ARRAY(SELECT DISTINCT value FROM UNNEST(g06n_cpc_concat) value ORDER BY value) AS g06n_cpc_codes,
  ARRAY(SELECT DISTINCT value FROM UNNEST(all_cpc_concat) value ORDER BY value) AS all_cpc_codes,
  publications_in_family
FROM aggregated;

-- Audit counts; export this result as data/scope_counts.csv.
SELECT
  publication_year,
  COUNT(*) AS simple_families,
  COUNTIF(ARRAY_LENGTH(assignees_harmonized) = 0) AS missing_assignee_families,
  COUNTIF(abstract_en IS NULL) AS missing_english_abstract_families,
  COUNT(DISTINCT country_code) AS patent_offices,
  SUM(publications_in_family) AS eligible_publications_before_family_deduplication
FROM `YOUR_PROJECT.YOUR_DATASET.family_master`
GROUP BY publication_year
ORDER BY publication_year;
