"""
rank.py — Phase 2 (Fast Ranking)
==================================
Loads pre-computed artifacts and ranks 100K candidates in <5 minutes.
CPU only. No network. No model loading.

Usage:
    python scripts/rank.py --candidates data/candidates.jsonl --out outputs/submission.xlsx

Output:
    XLSX with columns: candidate_id, rank, score, reasoning
    Top 100 candidates, ranked best-fit first.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT        = Path(__file__).parent.parent
ARTIFACTS   = ROOT / "artifacts"
CONFIG_PATH = ROOT / "scripts" / "jd_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

WEIGHTS = CONFIG["scoring_weights"]


# ─────────────────────────────────────────────
# Step 1: Load artifacts
# ─────────────────────────────────────────────
def load_artifacts():
    print("[1/5] Loading pre-computed artifacts...")
    t0 = time.time()

    embeddings   = np.load(ARTIFACTS / "embeddings.npy").astype(np.float32)
    jd_embedding = np.load(ARTIFACTS / "jd_embedding.npy").astype(np.float32)
    features     = pd.read_parquet(ARTIFACTS / "features.parquet")

    with open(ARTIFACTS / "candidate_ids.json") as f:
        candidate_ids = json.load(f)

    print(f"      embeddings   : {embeddings.shape}")
    print(f"      features     : {features.shape}")
    print(f"      candidate_ids: {len(candidate_ids):,}")
    print(f"      Loaded in {time.time()-t0:.1f}s")
    return embeddings, jd_embedding, features, candidate_ids


# ─────────────────────────────────────────────
# Step 2: Compute semantic similarity
# ─────────────────────────────────────────────
def compute_similarity(embeddings, jd_embedding):
    print("\n[2/5] Computing cosine similarity (vectorized)...")
    t0 = time.time()
    # Embeddings already L2-normalized during precompute
    # so cosine similarity = dot product
    scores = embeddings @ jd_embedding
    print(f"      Done in {time.time()-t0:.2f}s — min={scores.min():.3f} max={scores.max():.3f}")
    return scores


# ─────────────────────────────────────────────
# Step 3: Compute composite score
# ─────────────────────────────────────────────
def compute_composite(sim_scores, features):
    print("\n[3/5] Computing composite scores...")
    t0 = time.time()

    w = WEIGHTS
    sem_w  = float(str(w["semantic_similarity"]))
    title_w= float(str(w["title_fit"]))
    exp_w  = float(str(w["experience_fit"]))
    loc_w  = float(str(w["location_fit"]))

    sim_arr   = sim_scores
    title_arr = features["title_score"].values.astype(np.float32)
    exp_arr   = features["experience_score"].values.astype(np.float32)
    loc_arr   = features["location_score"].values.astype(np.float32)
    beh_arr   = features["behavioral_mult"].values.astype(np.float32)
    hp_arr    = features["is_honeypot"].values.astype(bool)

    # Base score (weighted sum)
    base = (
        sem_w   * sim_arr   +
        title_w * title_arr +
        exp_w   * exp_arr   +
        loc_w   * loc_arr
    )

    # Apply behavioral multiplier
    composite = base * beh_arr

    # Consulting-only penalty
    at_consulting = features["at_consulting_now"].values.astype(bool)
    has_product   = features["has_product_history"].values.astype(bool)
    consulting_only = at_consulting & ~has_product
    composite[consulting_only] *= CONFIG["company_type"]["penalty_if_only_consulting"]
    composite[at_consulting & has_product] *= CONFIG["company_type"]["penalty_if_current_consulting"]

    # Honeypot hard penalty
    composite[hp_arr] *= 0.1

    print(f"      Done in {time.time()-t0:.2f}s")
    print(f"      Score range: {composite.min():.4f} — {composite.max():.4f}")
    return composite


# ─────────────────────────────────────────────
# Step 4: Select top 100
# ─────────────────────────────────────────────
def select_top100(composite, candidate_ids, features):
    print("\n[4/5] Selecting top 100...")
    top_idx = np.argsort(composite)[::-1][:100]

    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        rows.append({
            "array_idx":   int(idx),
            "rank":        rank,
            "score":       round(float(composite[idx]), 6),
            "candidate_id":candidate_ids[idx],
        })

    top_df = pd.DataFrame(rows)
    top_df = top_df.merge(features, on="candidate_id", how="left")
    print(f"      Top score : {top_df['score'].iloc[0]:.4f}")
    print(f"      100th score: {top_df['score'].iloc[-1]:.4f}")
    return top_df


# ─────────────────────────────────────────────
# Step 5: Generate reasoning
# ─────────────────────────────────────────────
def generate_reasoning(row) -> str:
    """
    Builds a data-driven 1-2 sentence reasoning string
    from actual profile facts. No templates — sentence
    structure varies based on what's strongest for each candidate.
    """
    title   = row["current_title"]
    company = row["current_company"]
    yoe     = row["years_of_experience"]
    loc     = row["location"]
    kw_hits = [k for k in row["career_kw_hits"].split("|") if k] if row["career_kw_hits"] else []
    open_w  = row["open_to_work"]
    notice  = row["notice_period_days"]
    github  = row["github_score"]
    product = row["has_product_history"]
    ts      = row["title_score"]

    parts = []

    # Sentence 1: role + experience
    if ts >= 1.0:
        parts.append(f"{title} at {company} with {yoe:.0f} years of experience — strong role alignment with the JD.")
    elif ts >= 0.75:
        parts.append(f"{title} at {company} with {yoe:.0f} years of experience, with adjacent skills relevant to AI engineering.")
    else:
        parts.append(f"{title} at {company} with {yoe:.0f} years of experience.")

    # Sentence 2: career signals + availability
    signals = []
    if kw_hits:
        top_kw = ", ".join(kw_hits[:3])
        signals.append(f"career history mentions {top_kw}")
    if product:
        signals.append("product-company background")
    if open_w:
        signals.append("actively open to work")
    if notice <= 30:
        signals.append(f"available in {notice} days")
    if github >= 50:
        signals.append("active GitHub presence")

    if signals:
        parts.append("Profile shows " + "; ".join(signals) + ".")

    return " ".join(parts)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--out", default="outputs/submission.xlsx")
    args = parser.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)

    t_start = time.time()
    print("=" * 60)
    print("Redrob Ranker — rank.py")
    print("=" * 60)

    embeddings, jd_embedding, features, candidate_ids = load_artifacts()
    sim_scores = compute_similarity(embeddings, jd_embedding)
    composite  = compute_composite(sim_scores, features)
    top_df     = select_top100(composite, candidate_ids, features)

    # Generate reasoning
    print("\n[5/5] Generating per-candidate reasoning...")
    top_df["reasoning"] = top_df.apply(generate_reasoning, axis=1)

    # Build final submission dataframe
    submission = top_df[["candidate_id", "rank", "score", "reasoning"]].copy()

    # Validate score is non-increasing
    assert (submission["score"].diff().dropna() <= 0).all(), \
        "Scores are not non-increasing! Check sorting."

    # Write XLSX
    submission.to_excel(str(out_path), index=False)
    print(f"      Saved {len(submission)} rows to {out_path}")

    elapsed = time.time() - t_start
    print(f"""
{'='*60}
Ranking complete in {elapsed:.1f}s ({elapsed/60:.1f} min)

Output: {out_path}
Rows  : {len(submission)}
Top candidate : {submission.iloc[0]['candidate_id']} (score={submission.iloc[0]['score']:.4f})

Preview of top 5:
""")
    print(submission[["rank","candidate_id","score"]].head(5).to_string(index=False))
    print(f"""
Next steps:
  1. python validate_submission.py outputs/submission.xlsx
  2. git add outputs/submission.xlsx && git commit && git push
{'='*60}
""")


if __name__ == "__main__":
    main()
