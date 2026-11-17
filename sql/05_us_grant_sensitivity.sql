-- US-grant sensitivity population. This narrower design improves legal-status
-- comparability but must not be described as global.

CREATE OR REPLACE TABLE `YOUR_PROJECT.YOUR_DATASET.family_master_us_grants` AS
SELECT *
FROM `YOUR_PROJECT.YOUR_DATASET.family_master` family
WHERE EXISTS (
  SELECT 1
  FROM UNNEST(family.publication_numbers) publication_number
  JOIN `patents-public-data.patents.publications` publication
    ON publication.publication_number = publication_number
  WHERE publication.country_code = 'US'
    AND publication.grant_date BETWEEN 20180101 AND 20251231
);
