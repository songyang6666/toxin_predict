# WRR / AGU Open Research Notes

Water Resources Research follows AGU expectations that data and software needed
to understand, evaluate, and reproduce the results are available to reviewers
and readers. This repository contains only the total microcystin prediction
workflow and the data file used by that workflow.

## Prepared repository contents

- Python workflow code under `src/ml_toxin_predict/`.
- Runnable entry points at the repository root.
- Raw workbook under `data/raw/`.
- Generated outputs written to `data/outputs/` and excluded from version
  control by default.
- Optional processed data written to `data/processed/` with
  `--write-processed`.

## Publication checklist

- Archive the released GitHub version in Zenodo or another trusted repository
  to obtain a software DOI.
- Archive the data in a community-accepted, institutional, or generalist
  repository with a DOI where possible.
- Add formal data and software citations to the manuscript reference list.
- Replace placeholder author, repository, DOI, and license details in
  `CITATION.cff` and this document.

## Draft availability statement

The water-quality and microcystin data used for the total microcystin prediction
are available in [repository name] at [DOI or persistent URL] under [license and
access conditions]. The Python code used for preprocessing, KNN model training,
evaluation, and SHAP-based interpretation is preserved at [software DOI] and
developed openly at [GitHub URL] under the MIT License.
