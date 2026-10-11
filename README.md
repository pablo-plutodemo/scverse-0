# scverse-0 — human WAT snRNA-seq: artefact cluster and depot DGE

Reproducible analysis of the CELLxGENE human white adipose tissue (WAT) single-nucleus RNA-seq dataset
(collection [6b701826-37bb-4356-9792-ff41fc4c3161](https://cellxgene.cziscience.com/collections/6b701826-37bb-4356-9792-ff41fc4c3161),
Emont et al.; 65,668 nuclei, 19 donors, visceral + subcutaneous abdominal WAT).

## What the analysis does

1. Downloads the H5AD once (verify-if-exists) and confirms normalisation (`X` = log1p CP10k, `raw.X` = UMI counts).
2. Plots the authors' UMAP by cell type and by *ADIPOQ*.
3. Maps the batch structure: 22 libraries (= samples) nested in 19 donors; depot nested in donor except for 3 paired donors.
4. Identifies the "unknown" cluster as a **late-spermatid gene contamination artefact** (PRM1/TNP1/ODF1/AKAP4…, ~90% from female donors,
   lowest UMI) via a donor-centred scanpy t-test against every other cell type, and excludes it.
5. Builds an adipocyte-only object with raw counts and depot/donor/library fields.
6. Runs visceral-vs-subcutaneous DGE in adipocytes — **pseudobulk per library** (Welch t, BH; paired-donor sign concordance) as the primary
   analysis, with the cell-level donor-adjusted t-test alongside for comparison — and draws a two-panel volcano.

Key result: subcutaneous adipocytes express a positional-identity programme (HOXC6/8/9, TBX15, TWIST1, SIM1, EMX2OS); visceral adipocytes are
enriched for metabolic/lipid-handling genes (ACOT11, RXRG, EBF2). The cell-level test inflates hits with mitochondrial and ambient spermatid RNA.

## Layout

```
src/wat_pipeline.py                      all pipeline functions (download, plots, DE, pseudobulk, volcano)
notebooks/wat_snrna_depot_analysis.py    notebook source (jupytext percent format)
notebooks/wat_snrna_depot_analysis.ipynb executed notebook with outputs
results/                                 figures and DE tables produced by the notebook
environment.yml                          conda environment
```

## Run it

```bash
conda env create -f environment.yml && conda activate wat-snrna
export WAT_DATA_DIR=~/adipose_cxg/data        # optional; default shown. ~1 GB download on first run
cd notebooks
jupytext --to ipynb wat_snrna_depot_analysis.py
jupyter nbconvert --to notebook --execute --inplace wat_snrna_depot_analysis.ipynb
```

Peak memory is ~12 GB (the donor-centred cell-level t-test densifies genes detected in ≥1% of nuclei). Runtime is a few minutes after the download.

## Outputs (`results/`)

| file | content |
|---|---|
| `umap_cell_type.png`, `umap_cell_type_ADIPOQ.png` | authors' UMAP by cell type / ADIPOQ |
| `DGE_unknown_vs_each_cell_type_donorcentred_ttest.csv` | scanpy t-test, unknown vs each of 16 cell types, donor-centred |
| `DGE_unknown_consensus_up.csv` | genes up in "unknown" vs every other type |
| `DGE_unknown_markers_dotplot.png` | top consensus markers next to lineage markers |
| `DGE_adipocytes_visceral_vs_subcutaneous.csv` | per-gene pseudobulk + cell-level statistics, fold changes, detection, paired-donor concordance |
| `DGE_adipocytes_depot_volcano.png` | pseudobulk (a) vs cell-level (b) volcano |

Persistent data files (not in the repo): `$WAT_DATA_DIR/wat_snrna_original.h5ad` and `$WAT_DATA_DIR/wat_snrna_adipocytes.h5ad`.
