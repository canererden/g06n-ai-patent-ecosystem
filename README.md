# G06N AI patent ecosystem: data and code

Data and code for the article

> Tutar, H., & Erden, C. *Industrial AI Ecosystems and Innovation Diffusion: Mapping Structural Dominance and Its Societal Implications (2018–2025).* Technology Analysis & Strategic Management.

The repository contains the SQL queries that extract the corpus from Google Patents Public Data, the derived datasets analysed in the article, the analysis code, and the machine-readable results behind every table, figure and number reported in the article.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_all.py --available
```

`run_all.py` first joins the split data files, then re-runs the network (H1), proximity and clustering (H2, H3) and lexical (H4) analyses with all sensitivity specifications, redraws the figures, and compares every number printed in the article with the rebuilt value (`results/manuscript_numbers.md`). Without `--available` it also re-fits BERTopic, which downloads the embedding model and is the only computationally heavy step. The full run takes about 30–40 minutes on a laptop with 8 GB of memory.

## Data

| File | Rows | Content |
|---|---|---|
| `data/citation_edges/part-*.parquet` | 11,080,716 | Fractional assignee-to-assignee citation edges (citing → cited), one row per citing family × cited family × assignee pair; self-citations removed after name harmonization. |
| `data/assignee_cpc_year/part-*.parquet` | 6,420,968 | Family × assignee × CPC code panel with fractional weights. |
| `data/citation_edges_us_grants/part-*.parquet` | 6,791,449 | Citation edges for the US-grant sensitivity population. |
| `data/assignee_cpc_year_us_grants/part-*.parquet` | 1,308,145 | Family × assignee × CPC panel for the US-grant sensitivity population. |
| `data/abstract_sample.parquet` | 40,000 | Year-stratified sample (5,000 per year) of families with an English title and abstract, used for BERTopic and the lexical H4 test. |
| `data/scope_counts.csv` | 8 | Publications, families, missing assignees and missing English abstracts by year. |
| `data/SHA256SUMS.txt` | | Checksums; the large tables are checked on their content after joining. |

**Population.** Simple patent families with at least one current CPC code in subclass G06N and at least one publication dated 1 January 2018 – 31 December 2025, taken from `patents-public-data.patents.publications`: 682,082 families from 1,290,665 publications.

**Extraction dates.** The citation and CPC tables were extracted on **17 November 2026**. The abstract sample was drawn with the same SQL from an earlier extraction of the same population on 21 August 2026 and is kept fixed so that the BERTopic solution reported in the article remains reproducible. Google updates the public patent table continuously (new publications, CPC reclassification), so a new extraction will differ slightly from these files.

The large tables are split into parts below 25 MB; `python src/assemble_data.py` joins them into `data/<name>.parquet` in the original row order and verifies the content checksums.

## Code

| Step | Script | Article output |
|---|---|---|
| Extraction from BigQuery | `sql/01`–`06`, `run_bigquery.py` | Data above |
| Join split files | `src/assemble_data.py` | |
| Data audit | `src/run_data_audit.py` | Section 3.1 |
| Citation network: concentration, centralities, degree-preserving null test (100 randomizations), family bootstrap | `src/run_network.py` | Section 4.1, Table 1 |
| Annual proximity trend (pair fixed effects, 10,000 within-pair permutations); Ward clustering (k = 2–6 by silhouette, bootstrap stability) | `src/run_proximity.py` | Section 4.2, Figure 1 |
| Lexical generative-AI and vision indicators; segmented binomial model with prespecified 2023Q1 break | `src/run_h4_lexical.py` | Section 4.3, Figure 2 |
| BERTopic (all-MiniLM-L6-v2, UMAP, HDBSCAN; five seeds) | `src/run_topics.py` | Section 4.3, Table 2 |
| Figures and number check | `src/make_manuscript_figures.py`, `src/reproduce_manuscript_numbers.py` | Figures 1–2, all reported values |

All random procedures use seed 266665684 (BERTopic: 266665684–266665688). The tested environment is Python 3.12 (`environment.lock.txt`). `results/manifests/` records the parameters and input/output checksums of every run.

### Regenerating the data from BigQuery

```bash
gcloud auth application-default login
python run_bigquery.py --project YOUR_GCP_PROJECT --dataset YOUR_DATASET          # dry run
python run_bigquery.py --project YOUR_GCP_PROJECT --dataset YOUR_DATASET --execute --maximum-bytes-billed <limit>
```

The first query scans about 0.3 TB; the complete sequence ran within the monthly free tier of a BigQuery sandbox project in November 2026.

## Definitions used in the article

- **Unit of observation:** simple patent family. Multiple assignees or CPC codes receive fractional weights.
- **Corporate screen:** a transparent name rule (`NON_CORPORATE_PATTERN` in `src/run_network.py`) that excludes universities, institutes, laboratories, hospitals and public bodies; the network uses assignees with at least 50 G06N families. All-assignee and US-grant results are reported as sensitivities.
- **Table 1 positions:** hub = top five in PageRank; broker = betweenness ≥ 0.25; otherwise cited source.
- **Labels** of topics and clusters are author interpretations of the c-TF-IDF terms and of the over-represented CPC codes listed in `results/manuscript_numbers.md`.

## Licence

Code: MIT License (`LICENSE`). Data: derived from Google Patents Public Data by IFI CLAIMS Patent Services and Google, CC BY 4.0; the derived datasets and results in `data/` and `results/` are shared under CC BY 4.0 with this attribution.

## Citation

Please cite the article and the archived version of this repository (see `CITATION.cff`).
