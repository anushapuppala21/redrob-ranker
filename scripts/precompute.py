"""
precompute.py — Phase 1 (Offline Pre-computation)
===================================================
Reads all 100K candidates, builds profile text, computes
sentence-transformer embeddings, and extracts structured features.
Saves artifacts that rank.py loads at ranking time.

Run ONCE before ranking. Network and time are fine here.
Expected runtime: 15-25 minutes on CPU.

Usage:
    python scripts/precompute.py

Outputs:
    artifacts/embeddings.npy       float16, shape (100000, 384)
    artifacts/candidate_ids.json   list of candidate_ids in row order
    artifacts/features.parquet     structured per-candidate features
    artifacts/jd_embedding.npy     float32, shape (384,)
"""

import json
import time
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

ROOT        = Path(__file__).parent.parent
DATA_PATH   = ROOT / "data" / "candidates.jsonl"
CONFIG_PATH = ROOT / "scripts" / "jd_config.json"
ARTIFACTS   = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

CONSULTING_FIRMS = set(CONFIG["company_type"]["consulting_firms"])


def is_consulting(company_name: str) -> bool:
    cn = company_name.lower().strip()
    return any(firm in cn for firm in CONSULTING_FIRMS)


def score_title(title: str) -> float:
    t  = title.lower()
    tp = CONFIG["title_patterns"]
    for level in ["strong_match", "good_match", "weak_match"]:
        if any(p in t for p in tp[level]["patterns"]):
            return tp[level]["score"]
    for p in tp["no_match"]["patterns"]:
        if p in t:
            return 0.0
    return 0.2


def score_experience(yoe: float) -> float:
    if yoe < 3:     return 0.2
    elif yoe <= 4:  return 0.5
    elif yoe <= 9:  return 1.0
    elif yoe <= 12: return 0.8
    elif yoe <= 20: return 0.5
    else:           return 0.2


def score_location(location: str, country: str, willing_relocate: bool) -> float:
    loc = location.lower()
    lc  = CONFIG["location"]
    base = lc["outside_india"]["score"]
    if any(city in loc for city in lc["preferred"]["cities"]):
        base = lc["preferred"]["score"]
    elif any(city in loc for city in lc["acceptable"]["cities"]):
        base = lc["acceptable"]["score"]
    elif country.lower() == "india" or "india" in loc:
        base = lc["other_india"]["score"]
    if willing_relocate:
        base = min(1.0, base + lc["relocate_bonus"])
    return base


def compute_behavioral_multiplier(signals: dict) -> float:
    bs    = CONFIG["behavioral_signals"]
    today = date.today()
    mult  = 1.0

    lad = signals.get("last_active_date")
    if lad:
        try:
            days = (today - datetime.strptime(lad, "%Y-%m-%d").date()).days
            if days <= 30:    mult *= bs["recency"]["active_30d"]
            elif days <= 90:  mult *= bs["recency"]["active_90d"]
            elif days <= 180: mult *= bs["recency"]["active_180d"]
            else:             mult *= bs["recency"]["dead_180d_plus"]
        except Exception:
            pass

    if signals.get("open_to_work_flag"):
        mult *= bs["open_to_work"]["true"]

    rr = signals.get("recruiter_response_rate", 0.5)
    if rr >= 0.7:  mult *= bs["response_rate"]["high_0.7_plus"]
    elif rr < 0.3: mult *= bs["response_rate"]["low_under_0.3"]

    np_days = signals.get("notice_period_days", 60)
    if np_days <= 15:    mult *= bs["notice_period"]["0_to_15d"]
    elif np_days <= 30:  mult *= bs["notice_period"]["16_to_30d"]
    elif np_days <= 60:  mult *= bs["notice_period"]["31_to_60d"]
    elif np_days <= 90:  mult *= bs["notice_period"]["61_to_90d"]
    else:                mult *= bs["notice_period"]["over_90d"]

    gas = signals.get("github_activity_score", -1)
    if gas >= 50:   mult *= bs["github_active"]["score_50_plus"]
    elif gas == -1: mult *= bs["github_active"]["no_github"]

    pcs = signals.get("profile_completeness_score", 70)
    if pcs >= 80:  mult *= bs["profile_completeness"]["above_80"]
    elif pcs < 50: mult *= bs["profile_completeness"]["below_50"]

    return round(
        max(bs["multiplier_min"], min(bs["multiplier_max"], mult)), 4
    )


def detect_honeypot(c: dict) -> bool:
    hp     = CONFIG["honeypot_flags"]
    skills = c.get("skills", [])
    career = c.get("career_history", [])
    yoe    = c["profile"].get("years_of_experience", 0)

    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_zero >= hp["expert_skill_zero_duration_count_threshold"]:
        return True

    total_months = sum(r.get("duration_months", 0) for r in career)
    if yoe > 0 and total_months > yoe * 12 * hp["career_months_exceed_yoe_ratio"]:
        return True

    if yoe > hp["impossible_yoe_threshold"]:
        return True

    return False


def build_profile_text(c: dict) -> str:
    parts = []
    p = c["profile"]
    if p.get("headline"):
        parts.append(p["headline"])
    if p.get("summary"):
        parts.append(p["summary"])
    if p.get("current_title"):
        parts.append(f"Current role: {p['current_title']} at {p.get('current_company', '')}")
    for role in c.get("career_history", []):
        title   = role.get("title", "")
        company = role.get("company", "")
        desc    = role.get("description", "")
        if title or desc:
            parts.append(f"{title} at {company}: {desc}".strip())
    return " | ".join(parts)


def has_product_company_history(career: list) -> bool:
    return any(not is_consulting(r.get("company", "")) for r in career)


def avg_tenure(career: list) -> float:
    durations = [r.get("duration_months", 0) for r in career if r.get("duration_months", 0) > 0]
    return sum(durations) / len(durations) if durations else 0.0


def career_keyword_hits(career: list) -> list:
    strong = CONFIG["career_text_keywords"]["strong_signals"]
    found  = set()
    for role in career:
        desc = role.get("description", "").lower()
        for kw in strong:
            if kw in desc:
                found.add(kw)
    return sorted(found)


def main():
    t0 = time.time()
    print("=" * 60)
    print("Redrob Ranker — precompute.py")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/5] Loading candidates...")
    candidates = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"      Loaded {len(candidates):,} candidates")

    # Step 2: Build profile texts
    print("\n[2/5] Building profile texts...")
    profile_texts = []
    candidate_ids = []
    for c in tqdm(candidates, desc="      Texts"):
        profile_texts.append(build_profile_text(c))
        candidate_ids.append(c["candidate_id"])

    with open(ARTIFACTS / "candidate_ids.json", "w") as f:
        json.dump(candidate_ids, f)
    print(f"      Saved candidate_ids.json")

    # Step 3: Embeddings
    print("\n[3/5] Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("      Encoding 100K candidates (~15-20 min on CPU)...")
    embeddings = model.encode(
        profile_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embeddings_f16 = embeddings.astype(np.float16)
    np.save(ARTIFACTS / "embeddings.npy", embeddings_f16)
    size_mb = (ARTIFACTS / "embeddings.npy").stat().st_size / (1024 ** 2)
    print(f"      Saved embeddings.npy — {embeddings_f16.shape}, {size_mb:.1f} MB")

    # Step 4: JD embedding
    print("\n[4/5] Computing JD embedding...")
    jd_text      = CONFIG["jd_text_for_embedding"]
    jd_embedding = model.encode(jd_text, convert_to_numpy=True, normalize_embeddings=True)
    np.save(ARTIFACTS / "jd_embedding.npy", jd_embedding.astype(np.float32))
    print(f"      Saved jd_embedding.npy — shape {jd_embedding.shape}")

    # Step 5: Structured features
    print("\n[5/5] Computing structured features...")
    rows = []
    for c in tqdm(candidates, desc="      Features"):
        p       = c["profile"]
        signals = c["redrob_signals"]
        career  = c.get("career_history", [])
        rows.append({
            "candidate_id":        c["candidate_id"],
            "current_title":       p.get("current_title", ""),
            "current_company":     p.get("current_company", ""),
            "years_of_experience": p.get("years_of_experience", 0),
            "location":            p.get("location", ""),
            "country":             p.get("country", ""),
            "title_score":         score_title(p.get("current_title", "")),
            "experience_score":    score_experience(p.get("years_of_experience", 0)),
            "location_score":      score_location(
                                       p.get("location", ""),
                                       p.get("country", ""),
                                       signals.get("willing_to_relocate", False)
                                   ),
            "behavioral_mult":     compute_behavioral_multiplier(signals),
            "is_honeypot":         detect_honeypot(c),
            "has_product_history": has_product_company_history(career),
            "at_consulting_now":   is_consulting(p.get("current_company", "")),
            "avg_tenure_months":   round(avg_tenure(career), 1),
            "career_kw_hits":      "|".join(career_keyword_hits(career)),
            "open_to_work":        signals.get("open_to_work_flag", False),
            "notice_period_days":  signals.get("notice_period_days", 90),
            "github_score":        signals.get("github_activity_score", -1),
            "response_rate":       signals.get("recruiter_response_rate", 0.0),
            "profile_completeness":signals.get("profile_completeness_score", 0),
            "willing_to_relocate": signals.get("willing_to_relocate", False),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(ARTIFACTS / "features.parquet", index=False)
    size_mb = (ARTIFACTS / "features.parquet").stat().st_size / (1024 ** 2)
    print(f"      Saved features.parquet — {len(df):,} rows, {size_mb:.1f} MB")

    elapsed = time.time() - t0
    print(f"""
{'='*60}
Pre-computation complete in {elapsed/60:.1f} minutes

Artifacts saved to: artifacts/
  embeddings.npy       {(ARTIFACTS/'embeddings.npy').stat().st_size/1024**2:.1f} MB
  features.parquet     {(ARTIFACTS/'features.parquet').stat().st_size/1024**2:.1f} MB
  jd_embedding.npy     {(ARTIFACTS/'jd_embedding.npy').stat().st_size/1024:.1f} KB
  candidate_ids.json   {(ARTIFACTS/'candidate_ids.json').stat().st_size/1024:.1f} KB

Honeypots detected : {df['is_honeypot'].sum()}
Strong title match : {(df['title_score'] == 1.0).sum():,}
Zero title match   : {(df['title_score'] == 0.0).sum():,}

Next: python scripts/rank.py --candidates data/candidates.jsonl --out outputs/submission.xlsx
{'='*60}
""")


if __name__ == "__main__":
    main()
