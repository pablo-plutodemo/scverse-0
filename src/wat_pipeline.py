"""Reproducible pipeline for the CELLxGENE human white adipose tissue (WAT) snRNA-seq
dataset (collection 6b701826-37bb-4356-9792-ff41fc4c3161, Emont et al. 2022).

Steps (each a plain function; the notebook calls them in order):

1. fetch_dataset         - download the H5AD once (verify-if-exists), store under DATA_DIR
2. load_dataset          - read the file; X = log1p CP10k, raw.X = UMI counts
3. plot_umap_celltype    - authors' UMAP coloured by cell_type
4. plot_umap_gene        - authors' UMAP coloured by one gene (viridis, 99th-pct clip)
5. batch_overview        - library / donor / depot nesting
6. de_group_vs_each      - scanpy t-test of one cell type vs every other, donor-centred
7. extract_adipocytes    - adipocyte-only object with raw counts (verify-if-exists)
8. pseudobulk_depot_de   - visceral vs subcutaneous: pseudobulk per library (primary)
                           + cell-level donor-adjusted t-test (comparison)
9. volcano               - two-panel volcano with labelled extremes

WAT_DATA_DIR overrides the default data directory (~/adipose_cxg/data).
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import anndata as ad
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

COLLECTION_ID = "6b701826-37bb-4356-9792-ff41fc4c3161"
DATASET_ID = "9d8e5dca-03a3-457d-b7fb-844c75735c83"
H5AD_URL = "https://datasets.cellxgene.cziscience.com/a2f61ea6-e115-4304-a674-0ce025e98788.h5ad"

DATA_DIR = Path(os.environ.get("WAT_DATA_DIR", "~/adipose_cxg/data")).expanduser()
ORIGINAL = DATA_DIR / "wat_snrna_original.h5ad"
ADIPOCYTES = DATA_DIR / "wat_snrna_adipocytes.h5ad"

DEPOT_MAP = {
    "visceral abdominal adipose tissue": "visceral",
    "subcutaneous abdominal adipose tissue": "subcutaneous",
}

# label nudges tuned for the authors' embedding so endothelial labels do not overlap
LABEL_OFFSETS = {
    "endothelial stalk cell": (-6.0, -1.5),
    "endothelial cell of arteriole": (4.5, -1.5),
    "endothelial cell of venule": (1.5, 2.5),
    "mesothelial cell": (-1.5, 1.5),
    "pericyte": (-3.0, -1.0),
}


# --------------------------------------------------------------------------- 1
def fetch_dataset(path: Path = ORIGINAL, url: str = H5AD_URL, min_bytes: int = 900_000_000) -> Path:
    """Download the H5AD unless a valid copy already exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > min_bytes:
        with h5py.File(path, "r") as f:
            assert "X" in f and "obs" in f and "raw" in f, "file exists but is not a valid CELLxGENE H5AD"
        return path
    tmp = path.with_suffix(".h5ad.part")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- 2
def load_dataset(path: Path = ORIGINAL) -> ad.AnnData:
    adata = ad.read_h5ad(path)
    adata.var_names = adata.var_names.astype(str)
    return adata


def normalisation_report(adata: ad.AnnData, n: int = 5) -> pd.DataFrame:
    """Row sums of X and expm1(X) plus raw UMI totals for the first n nuclei: shows X = log1p(CP10k)."""
    X = adata.X[:n]
    return pd.DataFrame({
        "X_rowsum": np.asarray(X.sum(1)).ravel(),
        "expm1_X_rowsum": np.asarray(X.expm1().sum(1)).ravel(),
        "raw_UMI": np.asarray(adata.raw.X[:n].sum(1)).ravel(),
    }, index=adata.obs_names[:n]).round(1)


# --------------------------------------------------------------------------- 3/4
def _style_umap(ax, um, labels, min_n=1000, offsets=None, fs=8):
    offsets = offsets or {}
    for c in pd.unique(labels):
        m = labels == c
        if m.sum() < min_n:
            continue
        x, y = np.median(um[m, 0]), np.median(um[m, 1])
        dx, dy = offsets.get(c, (0, 0))
        ax.annotate(c, (x, y), xytext=(x + dx, y + dy), fontsize=fs, ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="k") if (dx or dy) else None)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_aspect("equal")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def plot_umap_celltype(adata: ad.AnnData, key: str = "cell_type", ax=None, seed: int = 0):
    um = adata.obsm["X_umap"]
    ct = adata.obs[key].astype(str).values
    counts = pd.Series(ct).value_counts()
    cmap = plt.get_cmap("tab20")
    colors = {c: cmap(i % 20) for i, c in enumerate(counts.index)}
    order = np.random.default_rng(seed).permutation(len(ct))
    fig, ax = (plt.subplots(figsize=(10, 8)) if ax is None else (ax.figure, ax))
    ax.scatter(um[order, 0], um[order, 1], s=1.5, c=[colors[c] for c in ct[order]], linewidths=0, rasterized=True)
    _style_umap(ax, um, ct, offsets=LABEL_OFFSETS)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, color=colors[c], label=f"{c} ({counts[c]:,})") for c in counts.index]
    ax.set_title(f"a  {key} ({adata.n_obs:,} nuclei)", loc="left")
    return fig, ax, handles


def gene_vector(adata: ad.AnnData, gene: str) -> np.ndarray:
    idx = np.where(adata.var["feature_name"].astype(str).values == gene)[0]
    assert len(idx) == 1, f"{gene}: {len(idx)} matches"
    return np.asarray(adata.X[:, idx[0]].todense()).ravel()


def plot_umap_gene(adata: ad.AnnData, gene: str, ax=None, cax=None):
    um = adata.obsm["X_umap"]
    e = gene_vector(adata, gene)
    vmax = np.percentile(e[e > 0], 99)
    o = np.argsort(e)
    fig, ax = (plt.subplots(figsize=(9, 8)) if ax is None else (ax.figure, ax))
    scp = ax.scatter(um[o, 0], um[o, 1], c=e[o], s=1.5, cmap="viridis", vmin=0, vmax=vmax, linewidths=0, rasterized=True)
    cb = fig.colorbar(scp, ax=None if cax is not None else ax, cax=cax, extend="max", shrink=0.6)
    cb.set_label(f"{gene}  log1p(CP10k)")
    _style_umap(ax, um, adata.obs["cell_type"].astype(str).values, offsets=LABEL_OFFSETS)
    ax.set_title(f"b  {gene}", loc="left")
    return fig, ax


def plot_umap_celltype_and_gene(adata: ad.AnnData, gene: str):
    import matplotlib.gridspec as gs
    fig = plt.figure(figsize=(16, 9.5))
    g = gs.GridSpec(2, 3, width_ratios=[1, 1, 0.04], height_ratios=[1, 0.14], wspace=0.08, hspace=0.02)
    ax1 = fig.add_subplot(g[0, 0]); ax2 = fig.add_subplot(g[0, 1], sharex=ax1, sharey=ax1)
    cax = fig.add_subplot(g[0, 2]); lax = fig.add_subplot(g[1, :]); lax.axis("off")
    _, _, handles = plot_umap_celltype(adata, ax=ax1)
    plot_umap_gene(adata, gene, ax=ax2, cax=cax); ax2.set_ylabel("")
    pos = cax.get_position(); cax.set_position([pos.x0, pos.y0 + pos.height * 0.25, pos.width, pos.height * 0.5])
    lax.legend(handles=handles, loc="center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle(f"Human WAT snRNA-seq (CELLxGENE) — {adata.n_obs:,} nuclei", y=0.95)
    return fig


def gene_by_celltype(adata: ad.AnnData, gene: str) -> pd.DataFrame:
    e = gene_vector(adata, gene)
    df = pd.DataFrame({"cell_type": adata.obs["cell_type"].astype(str).values, "e": e})
    return (df.groupby("cell_type")["e"]
              .agg(frac_detected=lambda v: float((v > 0).mean()), mean_log1p="mean")
              .sort_values("mean_log1p", ascending=False))


# --------------------------------------------------------------------------- 5
def batch_overview(adata: ad.AnnData) -> dict:
    obs = adata.obs
    per_donor = obs.groupby("donor_id", observed=True).agg(
        libraries=("library_id", "nunique"), depots=("tissue", "nunique"), nuclei=("tissue", "size"),
        sex=("sex", "first"), BMI=("BMI", "first"), T2D=("T2D", "first"),
    ).sort_values("libraries", ascending=False)
    donor_x_depot = pd.crosstab(obs["donor_id"], obs["tissue"].map(DEPOT_MAP))
    paired = donor_x_depot.index[(donor_x_depot > 0).all(1)].tolist()
    return {"per_donor": per_donor, "donor_x_depot": donor_x_depot, "paired_donors": paired,
            "alignment_software_libraries": obs.groupby("alignment_software", observed=True)["library_id"].nunique(),
            "n_libraries": obs["library_id"].nunique(), "n_samples": obs["sample_id"].nunique(),
            "n_donors": obs["donor_id"].nunique()}


# --------------------------------------------------------------------------- 6
def de_group_vs_each(adata: ad.AnnData, group: str, key: str = "cell_type", donor_key: str = "donor_id",
                     min_frac_detected: float = 0.01) -> pd.DataFrame:
    """scanpy t-test of `group` vs every other level of `key` after per-donor gene-mean centering
    (equivalent to sc.pp.regress_out with a single categorical covariate, but deterministic and
    single-process). log2FC and detection fractions use the un-centred log1p CP10k values."""
    import scanpy as sc

    det = np.asarray((adata.X > 0).sum(0)).ravel()
    keep = det >= min_frac_detected * adata.n_obs
    sym = adata.var["feature_name"].astype(str).values
    A = ad.AnnData(X=adata.X[:, keep].toarray().astype(np.float32),
                   obs=adata.obs[[key, donor_key]].copy(), var=pd.DataFrame(index=sym[keep]))
    A.var_names_make_unique()
    Xun = A.X.copy()
    donors = A.obs[donor_key].astype(str).values
    for d in np.unique(donors):
        m = donors == d
        A.X[m] -= A.X[m].mean(0)
    labels = A.obs[key].astype(str).values
    gmask = labels == group
    out = []
    for ref in [c for c in pd.unique(labels) if c != group]:
        sc.tl.rank_genes_groups(A, key, groups=[group], reference=ref, method="t-test", use_raw=False)
        d = sc.get.rank_genes_groups_df(A, group).drop(columns=["logfoldchanges"])
        gi = A.var_names.get_indexer(d["names"])
        rm = labels == ref
        mu_g, mu_r = Xun[gmask][:, gi].mean(0), Xun[rm][:, gi].mean(0)
        d["log2FC"] = np.log2(np.expm1(mu_g) + 1e-3) - np.log2(np.expm1(mu_r) + 1e-3)
        d[f"pct_{group}"] = (Xun[gmask][:, gi] > 0).mean(0)
        d["pct_ref"] = (Xun[rm][:, gi] > 0).mean(0)
        d["reference"] = ref
        d["n_ref"] = int(rm.sum())
        out.append(d)
    return pd.concat(out, ignore_index=True)


def consensus_markers(de: pd.DataFrame, group: str, padj=0.01, log2fc=1.0, min_pct=0.1) -> pd.DataFrame:
    sig = de[(de.pvals_adj < padj) & (de.log2FC > log2fc) & (de[f"pct_{group}"] > min_pct)]
    cons = sig.groupby("names").agg(n_refs=("reference", "nunique"), min_log2FC=("log2FC", "min"),
                                    pct_group=(f"pct_{group}", "first"), max_pct_ref=("pct_ref", "max"))
    return cons.sort_values(["n_refs", "min_log2FC"], ascending=False)


def dotplot(adata: ad.AnnData, genes: list[str], split_at: int | None = None, key="cell_type", title=""):
    sym = adata.var["feature_name"].astype(str).values
    gi = [np.where(sym == g)[0][0] for g in genes]
    labels = adata.obs[key].astype(str).values
    cats = pd.Series(labels).value_counts().index.tolist()
    sub = adata.X[:, gi].toarray()
    M = np.array([sub[labels == c].mean(0) for c in cats])
    P = np.array([(sub[labels == c] > 0).mean(0) for c in cats])
    fig, ax = plt.subplots(figsize=(0.4 * len(genes) + 4, 0.35 * len(cats) + 1))
    xx, yy = np.meshgrid(np.arange(len(genes)), np.arange(len(cats)))
    scp = ax.scatter(xx.ravel(), yy.ravel(), s=P.ravel() * 250 + 2, c=M.ravel(), cmap="Reds", vmin=0,
                     vmax=np.percentile(M, 99), edgecolors="grey", linewidths=0.3)
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=8); ax.invert_yaxis()
    if split_at is not None:
        ax.axvline(split_at - 0.5, color="k", lw=0.8, ls="--")
    cb = fig.colorbar(scp, ax=ax, shrink=0.6, pad=0.01); cb.set_label("mean log1p(CP10k)")
    for f, l in [(0.1, "10%"), (0.5, "50%"), (0.9, "90%")]:
        ax.scatter([], [], s=f * 250 + 2, c="w", edgecolors="grey", label=l)
    ax.legend(title="detected", loc="upper left", bbox_to_anchor=(1.12, 1), frameon=False, fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def qc_summary(adata: ad.AnnData, group: str, key="cell_type") -> dict:
    """UMI depth per cell type and the donor-sex / library / depot make-up of one group."""
    umi = np.asarray(adata.raw.X.sum(1)).ravel()
    labels = adata.obs[key].astype(str).values
    q = pd.DataFrame({"ct": labels, "umi": umi, "sex": adata.obs["sex"].astype(str).values,
                      "lib": adata.obs["library_id"].astype(str).values,
                      "depot": adata.obs["tissue"].map(DEPOT_MAP).astype(str).values})
    g = q[q.ct == group]
    return {"median_umi_by_type": q.groupby("ct")["umi"].median().sort_values(),
            "sex_fraction": g["sex"].value_counts(normalize=True).round(2),
            "depot_counts": g["depot"].value_counts(),
            "fraction_of_library": (g.groupby("lib").size() / q.groupby("lib").size()).sort_values(ascending=False).dropna().round(2)}


# --------------------------------------------------------------------------- 7
def extract_adipocytes(adata: ad.AnnData | None, path: Path = ADIPOCYTES) -> ad.AnnData:
    """Adipocyte-only object with raw UMI counts in X plus depot/donor/library fields. Verify-if-exists."""
    path = Path(path)
    if path.exists():
        adi = ad.read_h5ad(path)
        assert adi.X.dtype.kind in "iu" and "depot" in adi.obs and "library_id" in adi.obs, "existing adipocyte file is not valid"
        return adi
    assert adata is not None, "need the full dataset to build the adipocyte file"
    m = (adata.obs["cell_type"].astype(str) == "adipocyte").values
    keep = ["cell_type", "author_cell_type", "tissue", "origin", "donor_id", "library_id", "sample_id", "sex",
            "development_stage", "BMI", "T2D", "Race_Ethnicity", "alignment_software"]
    adi = ad.AnnData(X=sp.csr_matrix(adata.raw.X[m]).astype(np.int32), obs=adata.obs.loc[m, keep].copy(),
                     var=adata.var[["feature_name"]].copy())
    adi.obs["depot"] = adi.obs["tissue"].map(DEPOT_MAP).astype("category")
    adi.obsm["X_umap"] = adata.obsm["X_umap"][m]
    adi.uns["selection"] = {"source": ORIGINAL.name, "filter": "cell_type == 'adipocyte'", "X": "raw UMI counts from adata.raw.X"}
    path.parent.mkdir(parents=True, exist_ok=True)
    adi.write_h5ad(path, compression="gzip")
    return adi


# --------------------------------------------------------------------------- 8
def pseudobulk_depot_de(adi: ad.AnnData, min_cells: int = 20, min_cpm: float = 1.0, min_frac_libs: float = 0.1) -> pd.DataFrame:
    """Primary: sum raw counts per library -> log2(CPM+1) -> Welch t-test visceral vs subcutaneous -> BH;
    donors with both depots are reported as sign-concordance. Secondary: cell-level Welch t-test on
    log1p CP10k after per-donor centering (anti-conservative; comparison only)."""
    sym = adi.var["feature_name"].astype(str).values
    libs = adi.obs["library_id"].astype(str).values
    ulib = np.unique(libs)
    PB = np.vstack([np.asarray(adi.X[libs == l].sum(0)).ravel() for l in ulib])
    first = adi.obs.drop_duplicates("library_id").copy()
    first.index = first["library_id"].astype(str)
    meta = first.loc[ulib, ["depot", "donor_id"]].copy()
    meta["n_cells"] = pd.Series(libs).value_counts().loc[ulib].values
    ok = meta["n_cells"].values >= min_cells
    PB, meta = PB[ok], meta[ok]
    cpm = PB / PB.sum(1, keepdims=True) * 1e6
    lc = np.log2(cpm + 1)
    vis = (meta["depot"] == "visceral").values
    expressed = ((cpm > min_cpm).mean(0) > min_frac_libs) & (sym != "nan")
    t, pv = stats.ttest_ind(lc[vis][:, expressed], lc[~vis][:, expressed], equal_var=False)
    padj = multipletests(pv, method="fdr_bh")[1]
    lfc = lc[vis][:, expressed].mean(0) - lc[~vis][:, expressed].mean(0)
    paired = [d for d in meta["donor_id"].unique() if meta.loc[meta.donor_id == d, "depot"].nunique() == 2]
    conc = np.zeros(int(expressed.sum()), int)
    if paired:
        pdiff = np.vstack([lc[(meta.donor_id == d).values & vis][:, expressed].mean(0)
                           - lc[(meta.donor_id == d).values & ~vis][:, expressed].mean(0) for d in paired])
        conc = (np.sign(pdiff) == np.sign(lfc)).sum(0)
    # cell-level comparison
    tot = np.asarray(adi.X.sum(1)).ravel()
    Xn = np.log1p((sp.diags(1e4 / tot) @ adi.X[:, expressed].astype(np.float32)).toarray())
    Xun = Xn.copy()
    donors = adi.obs["donor_id"].astype(str).values
    for d in np.unique(donors):
        m = donors == d
        Xn[m] -= Xn[m].mean(0)
    cv = (adi.obs["depot"] == "visceral").values
    tc, pc = stats.ttest_ind(Xn[cv], Xn[~cv], equal_var=False)
    mu_v, mu_s = Xun[cv].mean(0), Xun[~cv].mean(0)
    T = pd.DataFrame({
        "gene": sym[expressed], "ensembl": np.asarray(adi.var_names)[expressed],
        "pb_log2FC_vis_vs_sq": lfc, "pb_t": t, "pb_pval": pv, "pb_padj": padj,
        "paired_donors_concordant": conc, "n_paired_donors": len(paired),
        "pb_mean_log2CPM_vis": lc[vis][:, expressed].mean(0), "pb_mean_log2CPM_sq": lc[~vis][:, expressed].mean(0),
        "cell_log2FC_vis_vs_sq": np.log2(np.expm1(mu_v) + 1e-3) - np.log2(np.expm1(mu_s) + 1e-3),
        "cell_t_donoradj": tc, "cell_pval": pc, "cell_padj": multipletests(pc, method="fdr_bh")[1],
        "pct_vis": (Xun[cv] > 0).mean(0), "pct_sq": (Xun[~cv] > 0).mean(0),
    }).sort_values("pb_pval")
    T.attrs["n_libraries"] = int(ok.sum())
    T.attrs["paired_donors"] = paired
    return T


# --------------------------------------------------------------------------- 9
def _place_labels(ax, fig, pts, placed):
    for x, y, g in pts:
        for k in range(12):
            dy = (k // 2 + 1) * 7 * (1 if k % 2 == 0 else -1)
            an = ax.annotate(g, (x, y), xytext=(0, dy), textcoords="offset points", fontsize=7,
                             ha="left" if x > 0 else "right", arrowprops=dict(arrowstyle="-", lw=0.4, color="grey"))
            fig.canvas.draw_idle()
            bb = an.get_window_extent(fig.canvas.get_renderer())
            if not any(bb.overlaps(b) for b in placed):
                placed.append(bb)
                break
            an.remove()


def volcano(T: pd.DataFrame, n_label: int = 14, padj_thr: float = 0.05, lfc_thr: float = 1.0):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    panels = [("pb_log2FC_vis_vs_sq", "pb_padj", "a  Pseudobulk per library (Welch t, BH) — primary", "lfc"),
              ("cell_log2FC_vis_vs_sq", "cell_padj", "b  Cell-level donor-adjusted t-test (anti-conservative) — comparison", "y")]
    for ax, (lf, pa, ttl, rank) in zip(axes, panels):
        x = T[lf].values
        y = -np.log10(np.clip(T[pa].values, 1e-300, None))
        sig = ((T[pa] < padj_thr) & (np.abs(x) > lfc_thr)).values
        col = np.where(sig & (x > 0), "#c0392b", np.where(sig & (x < 0), "#2471a3", "lightgrey"))
        ax.scatter(x, y, s=6, c=col, linewidths=0, rasterized=True)
        ax.axhline(-np.log10(padj_thr), ls="--", lw=0.7, color="k")
        for v in (-lfc_thr, lfc_thr):
            ax.axvline(v, ls="--", lw=0.7, color="k")
        ax.set_xlabel("log2FC  visceral / subcutaneous"); ax.set_ylabel("-log10 adjusted p")
        ax.set_title(ttl, loc="left", fontsize=10)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        S = T[sig].copy(); S["x"] = x[sig]; S["y"] = y[sig]; S = S[~S.gene.str.startswith("ENSG")]
        S["k"] = (S.x.abs() * (1 + S.y)) if rank == "lfc" else S.y
        up = S[S.x > 0].sort_values("k", ascending=False).head(n_label)
        dn = S[S.x < 0].sort_values("k", ascending=False).head(n_label)
        placed = []
        _place_labels(ax, fig, list(zip(up.x, up.y, up.gene)), placed)
        _place_labels(ax, fig, list(zip(dn.x, dn.y, dn.gene)), placed)
        ax.text(0.98, 0.02, f"{int((sig & (x > 0)).sum())} up visceral", transform=ax.transAxes, ha="right", color="#c0392b", fontsize=9)
        ax.text(0.02, 0.02, f"{int((sig & (x < 0)).sum())} up subcutaneous", transform=ax.transAxes, ha="left", color="#2471a3", fontsize=9)
    fig.suptitle("Adipocytes: visceral vs subcutaneous depot DGE", y=0.99)
    fig.tight_layout()
    return fig
