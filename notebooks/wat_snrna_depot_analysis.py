# %% [markdown]
# # Human white adipose tissue snRNA-seq (CELLxGENE): cell types, an artefact cluster, and depot differences in adipocytes
#
# Dataset: CELLxGENE collection [6b701826-37bb-4356-9792-ff41fc4c3161](https://cellxgene.cziscience.com/collections/6b701826-37bb-4356-9792-ff41fc4c3161)
# (*Single-Nucleus Analysis of Human White Adipose Tissue Reveals Adipocyte Subsets with Distinct Metabolic Profiles*;
# 65,668 nuclei, 19 donors, visceral and subcutaneous abdominal WAT, 10x 3' v3).
#
# All logic lives in `src/wat_pipeline.py`; this notebook calls it step by step and records the biological reading of each result.
# Persistent files are written to `WAT_DATA_DIR` (default `~/adipose_cxg/data`) and re-used if they already exist.

# %%
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))
sys.path.insert(0, str(Path("..").resolve() / "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import wat_pipeline as wp

RESULTS = Path("..") / "results"; RESULTS.mkdir(exist_ok=True)
print("data dir:", wp.DATA_DIR)

# %% [markdown]
# ## 1. Download (verify-if-exists) and load

# %%
path = wp.fetch_dataset()
adata = wp.load_dataset(path)
adata

# %% [markdown]
# ## 2. How many nuclei, and is the data normalised?
# `X` is log1p of counts scaled to 10,000 per nucleus (the expm1 row sums are ~10,000); raw integer UMI counts are kept in `adata.raw.X`.

# %%
print(f"{adata.n_obs:,} nuclei x {adata.n_vars:,} genes")
wp.normalisation_report(adata)

# %% [markdown]
# ## 3. UMAP by cell type and by ADIPOQ

# %%
fig = wp.plot_umap_celltype(adata)[0]
fig.savefig(RESULTS / "umap_cell_type.png", dpi=150, bbox_inches="tight"); plt.show()

# %%
fig = wp.plot_umap_celltype_and_gene(adata, "ADIPOQ")
fig.savefig(RESULTS / "umap_cell_type_ADIPOQ.png", dpi=150, bbox_inches="tight"); plt.show()
wp.gene_by_celltype(adata, "ADIPOQ").head(6)

# %% [markdown]
# **Reading.** ADIPOQ is restricted to the adipocyte lobe (~70% of adipocyte nuclei) and absent from preadipocytes — a clean terminal-differentiation split.
# Expression is graded within the adipocyte cluster, consistent with the adipocyte sub-states described by the authors.

# %% [markdown]
# ## 4. Sources of batch effects
# The finest technical unit is the library (1:1 with sample), nested in donor. Depot is almost completely nested in donor:
# only three donors contribute both depots. Donor carries sex, BMI, T2D status and ancestry; three Cell Ranger versions were used.

# %%
bo = wp.batch_overview(adata)
print(f"{bo['n_libraries']} libraries = {bo['n_samples']} samples from {bo['n_donors']} donors; paired donors: {bo['paired_donors']}")
print(bo["alignment_software_libraries"].to_string())
bo["donor_x_depot"]

# %% [markdown]
# ## 5. What is the "unknown" cluster? scanpy t-test vs every other cell type, controlling for donor

# %%
de_unknown = wp.de_group_vs_each(adata, "unknown")
cons = wp.consensus_markers(de_unknown, "unknown")
de_unknown.to_csv(RESULTS / "DGE_unknown_vs_each_cell_type_donorcentred_ttest.csv", index=False)
cons.to_csv(RESULTS / "DGE_unknown_consensus_up.csv")
n_other = de_unknown["reference"].nunique()
top = cons[cons.n_refs == n_other].head(20)
top

# %%
lineage = [g for g in ["ACTA2", "MYH11", "RGS5", "PDGFRB", "PLP1", "NRXN1", "DCN", "PDGFRA", "ADIPOQ", "PTPRC", "PECAM1"]]
fig = wp.dotplot(adata, top.index.tolist() + lineage, split_at=len(top),
                 title='"unknown" cluster: top consensus markers (up vs every other type) | lineage markers')
fig.savefig(RESULTS / "DGE_unknown_markers_dotplot.png", dpi=150, bbox_inches="tight"); plt.show()

# %%
qc = wp.qc_summary(adata, "unknown")
print("median UMI (lowest four):\n", qc["median_umi_by_type"].head(4).to_string())
print("\nsex of donors contributing 'unknown' nuclei:\n", qc["sex_fraction"].to_string())
print("\ndepot:\n", qc["depot_counts"].to_string())
print("\nshare of library made up of 'unknown' nuclei (top 5):\n", qc["fraction_of_library"].head(5).to_string())
print("\nauthor label:\n", adata.obs.loc[adata.obs.cell_type == "unknown", "author_cell_type"].value_counts().head(3).to_string())

# %% [markdown]
# **Reading.** The "unknown" cluster is a late-spermatid gene signature (PRM1, PRM3, TNP1, ODF1, AKAP4, GAPDHS, SPATA24, MEIG1, TUBA3C, ACTL7B)
# with no lineage marker of its own; the authors labelled it "PRM1+". It is biologically impossible as sampled — ~90% of the nuclei come from
# **female** donors, mostly visceral, concentrated in a few libraries, with the lowest UMI counts after erythroblasts. It is a
# contamination/low-quality artefact and is excluded from here on.

# %% [markdown]
# ## 6. Adipocyte-only object (raw counts + depot/donor/library fields)

# %%
adi = wp.extract_adipocytes(adata)
print(wp.ADIPOCYTES, adi.shape, adi.X.dtype)
adi.obs.groupby("depot", observed=True).agg(nuclei=("donor_id", "size"), donors=("donor_id", "nunique"), libraries=("library_id", "nunique"))

# %% [markdown]
# ## 7. Visceral vs subcutaneous adipocytes
# Depot is nested in donor, so the primary analysis is **pseudobulk per library** (sum raw counts, log2 CPM, Welch t-test, BH),
# with the three paired donors used as a sign-concordance check. The cell-level donor-adjusted t-test is shown alongside for comparison only:
# nuclei are not independent replicates, so it is anti-conservative.

# %%
T = wp.pseudobulk_depot_de(adi)
T.to_csv(RESULTS / "DGE_adipocytes_visceral_vs_subcutaneous.csv", index=False)
sig = T[(T.pb_padj < 0.05) & (T.pb_log2FC_vis_vs_sq.abs() > 1)]
print(f"{T.attrs['n_libraries']} libraries; paired donors {T.attrs['paired_donors']}")
print(f"pseudobulk: {(sig.pb_log2FC_vis_vs_sq > 0).sum()} up visceral, {(sig.pb_log2FC_vis_vs_sq < 0).sum()} up subcutaneous "
      f"(padj<0.05, |log2FC|>1); {(sig.paired_donors_concordant == len(T.attrs['paired_donors'])).mean():.0%} concordant in all paired donors")
print(f"cell-level: {((T.cell_padj < 0.05) & (T.cell_log2FC_vis_vs_sq.abs() > 1)).sum()} hits at the same thresholds")
print("\nup subcutaneous:", sig.sort_values("pb_log2FC_vis_vs_sq").gene.head(12).tolist())
print("up visceral:    ", sig.sort_values("pb_log2FC_vis_vs_sq").gene.tail(12).tolist()[::-1])

# %%
fig = wp.volcano(T)
fig.savefig(RESULTS / "DGE_adipocytes_depot_volcano.png", dpi=150, bbox_inches="tight"); plt.show()

# %%
T.head(20)[["gene", "pb_log2FC_vis_vs_sq", "pb_padj", "paired_donors_concordant", "cell_log2FC_vis_vs_sq", "cell_padj", "pct_vis", "pct_sq"]].round(3)

# %% [markdown]
# **Reading.**
# - Subcutaneous adipocytes carry a developmental positional-identity programme — HOXC6/HOXC8/HOXC9, TBX15, TWIST1, SIM1, EMX2OS — while visceral
#   adipocytes are enriched for metabolic / lipid-handling genes (ACOT11, RXRG, EBF2, ENOX1). The direction holds in the paired donors for >90% of hits,
#   so this is depot biology, not donor.
# - The cell-level test returns ~3x more hits, but the extra signal is largely technical: mitochondrial transcripts and spermatid genes (PRM1, TNP1 —
#   ambient RNA from the artefact cluster in visceral libraries) rise to the top. When depot is nested in donor, pseudobulk per library is the defensible route.
