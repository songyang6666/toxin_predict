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
- Add the final repository DOI and manuscript citation after acceptance.

## Draft Open Research statement for peer review

The water-quality and microcystin data supporting this study are publicly
available from the NOAA National Centers for Environmental Information at
https://doi.org/10.25921/11da-3x54. The analysis was conducted using Python,
scikit-learn, and SHAP. Custom scripts for preprocessing, model training and
evaluation, prediction-horizon comparison, figure generation, and SHAP-based
interpretation, together with non-interpolated blocked-validation and baseline
comparisons, are available to editors and reviewers during peer review
through a private GitHub repository:
https://github.com/songyang6666/toxin_predict. Upon acceptance, the repository
will be made public and archived in Zenodo or another trusted repository with a
persistent DOI, and the final Open Research statement and reference list will
be updated accordingly.
