"""Candidate quality features from FEC filing history, TF-IDF text vectorization,
and (optionally) sentence-transformer embeddings.

Teaches: TF-IDF vectorization, Truncated SVD for dimensionality reduction,
text preprocessing with spaCy, sentence-transformer embeddings with PCA,
percentile ranking, and combining structured + NLP signals into a single score.

The candidate quality score captures information not in the district's partisan
history: whether a strong incumbent faces a weak challenger, whether a party
recruited a credible candidate for a tough district, etc.

Architecture
------------
1. Structured tier  (always available)
   experience_cycles  — distinct election cycles the candidate filed in FEC data
   receipts_rank      — percentile rank within (year, party) cohort

2. Text tier  (always available, sklearn only)
   TF-IDF on constructed candidate descriptions → TruncatedSVD → 2 components

3. Embedding tier  (optional, requires sentence-transformers)
   SentenceTransformer("all-MiniLM-L6-v2") → PCA → 2 components
   Gracefully skipped if package not installed.

Output
------
data/interim/candidate_quality.csv  — year, state_po, district,
    candidate_quality_dem_share, experience_cycles_dem, experience_cycles_rep
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.constants import Col
from src.paths import DataFile, ensure

logger = logging.getLogger(__name__)

__all__ = [
    "compute_experience_cycles",
    "select_top_candidates",
    "compute_receipts_rank",
    "build_candidate_descriptions",
    "tfidf_components",
    "embedding_components",
    "compute_quality_scores",
    "build_candidate_quality",
    "load_candidate_quality",
]

_KEY_COLS = [Col.YEAR, Col.STATE, Col.DISTRICT, Col.PARTY]
_CANDIDATE_COLS = [
    Col.YEAR,
    Col.STATE,
    Col.DISTRICT,
    Col.PARTY,
    "candidate_id",
    "candidate_name",
    "receipts",
    "experience_cycles",
]


def compute_experience_cycles(fec_raw: pd.DataFrame) -> pd.DataFrame:
    """Add experience_cycles column — distinct election cycles per candidate_id."""
    exp = (
        fec_raw.groupby("candidate_id")[Col.YEAR]
        .nunique()
        .rename("experience_cycles")
        .reset_index()
    )
    result = fec_raw.copy()
    result = result.drop(columns=["experience_cycles"], errors="ignore")
    result = result.merge(exp, on="candidate_id", how="left")
    return result


def select_top_candidates(fec_raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest-receipts DEM and REP candidate per (year, state, district)."""
    dem_rep = fec_raw[fec_raw[Col.PARTY].isin(["DEM", "REP"])].copy()
    if "experience_cycles" not in dem_rep.columns:
        dem_rep["experience_cycles"] = 1
    idx = dem_rep.groupby(_KEY_COLS)["receipts"].idxmax()
    top = dem_rep.loc[idx, _CANDIDATE_COLS].reset_index(drop=True)
    top["experience_cycles"] = top["experience_cycles"].fillna(1)
    return top


def compute_receipts_rank(candidates: pd.DataFrame) -> pd.DataFrame:
    """Add receipts_rank in [0, 1] — percentile rank within each (year, party) cohort."""
    result = candidates.copy()
    ranks = result.groupby([Col.YEAR, Col.PARTY])["receipts"].transform(
        lambda s: rankdata(s, method="average") / len(s)
    )
    result["receipts_rank"] = ranks
    return result


def build_candidate_descriptions(candidates: pd.DataFrame) -> pd.Series:
    """Construct short text descriptions for TF-IDF / embedding input."""
    party_word = candidates[Col.PARTY].map({"DEM": "democrat", "REP": "republican"}).fillna("other")
    incumbent = candidates["experience_cycles"].apply(
        lambda c: "incumbent" if c >= 2 else "challenger"
    )
    state = candidates[Col.STATE].str.lower()
    district = candidates[Col.DISTRICT].astype(str)

    def _fundraising_tier(rank: float) -> str:
        if rank >= 0.8:
            return "major fundraiser"
        if rank >= 0.4:
            return "competitive"
        return "underfunded"

    fundraising = candidates["receipts_rank"].apply(_fundraising_tier)
    cycles = candidates["experience_cycles"].apply(lambda c: f"{min(int(c), 10)} cycles")

    descriptions = (
        party_word
        + " "
        + incumbent
        + " "
        + state
        + " district "
        + district
        + " "
        + fundraising
        + " "
        + cycles
    )
    return descriptions


def tfidf_components(descriptions: pd.Series, n_components: int = 2) -> pd.DataFrame:
    """TF-IDF → TruncatedSVD, returns DataFrame with columns tfidf_1, tfidf_2, ..."""
    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.98,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    components = svd.fit_transform(tfidf_matrix)
    col_names = [f"tfidf_{i + 1}" for i in range(n_components)]
    return pd.DataFrame(components, columns=col_names, index=descriptions.index)


def embedding_components(descriptions: pd.Series, n_components: int = 2) -> pd.DataFrame | None:
    """Sentence-transformer embeddings → PCA. Returns None if package not installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.info("sentence-transformers not installed — skipping embedding tier")
        return None

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings: np.ndarray = model.encode(descriptions.tolist(), show_progress_bar=False)
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(embeddings)
    col_names = [f"embed_{i + 1}" for i in range(n_components)]
    return pd.DataFrame(components, columns=col_names, index=descriptions.index)


def _min_max_normalize(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-9)


def compute_quality_scores(candidates: pd.DataFrame, use_embeddings: bool = False) -> pd.DataFrame:
    """Combine structured and NLP signals into a single quality_score in [0, 1]."""
    result = candidates.copy()
    result = result.reset_index(drop=True)

    descriptions = build_candidate_descriptions(result)
    tfidf_df = tfidf_components(descriptions)
    result = pd.concat([result, tfidf_df], axis=1)

    if use_embeddings:
        embed_df = embedding_components(descriptions)
        if embed_df is not None:
            result = pd.concat([result, embed_df], axis=1)

    exp_norm = _min_max_normalize(result["experience_cycles"].clip(lower=1, upper=10))
    tfidf_norm = _min_max_normalize(result["tfidf_1"])

    quality_raw = 0.5 * exp_norm + 0.3 * result["receipts_rank"] + 0.2 * tfidf_norm
    result["quality_score"] = _min_max_normalize(quality_raw)
    return result


def build_candidate_quality(
    fec_raw_path: Path,
    use_embeddings: bool = False,
) -> pd.DataFrame:
    """Main pipeline: FEC raw CSV → candidate quality features per (year, state, district).

    Returns DataFrame with year, state_po, district, candidate_quality_dem_share,
    experience_cycles_dem, experience_cycles_rep.
    """
    fec_raw = pd.read_csv(fec_raw_path)
    fec_raw = compute_experience_cycles(fec_raw)

    top = select_top_candidates(fec_raw)
    dem = top[top[Col.PARTY] == "DEM"].copy()
    rep = top[top[Col.PARTY] == "REP"].copy()

    dem = compute_receipts_rank(dem)
    rep = compute_receipts_rank(rep)

    dem = compute_quality_scores(dem, use_embeddings=use_embeddings)
    rep = compute_quality_scores(rep, use_embeddings=use_embeddings)

    merge_cols = [Col.YEAR, Col.STATE, Col.DISTRICT]
    dem_slim = dem[merge_cols + ["quality_score", "experience_cycles"]].rename(
        columns={"quality_score": "dem_quality", "experience_cycles": Col.EXPERIENCE_CYCLES_DEM}
    )
    rep_slim = rep[merge_cols + ["quality_score", "experience_cycles"]].rename(
        columns={"quality_score": "rep_quality", "experience_cycles": Col.EXPERIENCE_CYCLES_REP}
    )

    merged = dem_slim.merge(rep_slim, on=merge_cols, how="outer")
    merged["dem_quality"] = merged["dem_quality"].fillna(0.0)
    merged["rep_quality"] = merged["rep_quality"].fillna(0.0)
    merged[Col.EXPERIENCE_CYCLES_DEM] = merged[Col.EXPERIENCE_CYCLES_DEM].fillna(1.0)
    merged[Col.EXPERIENCE_CYCLES_REP] = merged[Col.EXPERIENCE_CYCLES_REP].fillna(1.0)

    total = merged["dem_quality"] + merged["rep_quality"]
    merged[Col.CANDIDATE_QUALITY] = (merged["dem_quality"] / total.clip(lower=1e-9)).where(
        total > 0, 0.5
    )

    return merged[
        [
            Col.YEAR,
            Col.STATE,
            Col.DISTRICT,
            Col.CANDIDATE_QUALITY,
            Col.EXPERIENCE_CYCLES_DEM,
            Col.EXPERIENCE_CYCLES_REP,
        ]
    ]


def load_candidate_quality(path: Path) -> pd.DataFrame | None:
    """Load candidate_quality.csv from disk, or return None if not present."""
    if not path.exists():
        logger.info("candidate_quality.csv not found — skipping")
        return None
    return pd.read_csv(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = build_candidate_quality(fec_raw_path=DataFile.FUNDRAISING_RAW)
    df.to_csv(ensure(DataFile.CANDIDATE_QUALITY), index=False)
    logger.info("Candidate quality saved → %s (%d rows)", DataFile.CANDIDATE_QUALITY, len(df))


if __name__ == "__main__":
    main()
