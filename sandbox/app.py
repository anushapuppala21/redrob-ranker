"""
HireSignal — AI Candidate Ranking Sandbox
==========================================
Streamlit demo for the Redrob Hackathon submission.
Runs the full ranking pipeline on a small candidate sample.

Deploy to HuggingFace Spaces as a Streamlit app.
"""

import json
import io
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HireSignal — AI Candidate Ranker",
    page_icon="🎯",
    layout="wide"
)

# ─────────────────────────────────────────────
# Config (embedded — no file dependency)
# ─────────────────────────────────────────────
JD_TEXT = """Senior AI Engineer with production experience in embeddings-based
retrieval systems, vector databases, hybrid search, semantic search, and ranking
evaluation frameworks. Experience with sentence-transformers, FAISS, Pinecone,
Weaviate, Elasticsearch, or similar. Strong Python. Hands-on with NDCG, MRR,
MAP evaluation. LLM integration, fine-tuning with LoRA or QLoRA a plus.
Worked at product companies, not consulting. Shipped end-to-end ranking or
recommendation systems to real users. 5-9 years experience.
Located in India, preferably Pune or Noida."""

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "ltimindtree", "persistent systems"
}

STRONG_TITLES = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "applied scientist", "research engineer", "nlp engineer",
    "senior ai", "senior ml", "lead ai", "lead ml"
]
GOOD_TITLES = [
    "data scientist", "deep learning", "mlops", "backend engineer",
    "software engineer", "senior software"
]
WEAK_TITLES = [
    "data engineer", "analytics engineer", "data analyst",
    "full stack", "cloud engineer", "devops"
]
NO_TITLES = [
    "business analyst", "hr manager", "accountant", "project manager",
    "customer support", "content writer", "sales", "civil engineer",
    "mechanical engineer", "graphic designer", "marketing manager"
]

CAREER_KEYWORDS = [
    "retrieval", "embedding", "vector", "semantic search", "ranking",
    "recommendation", "llm", "fine-tun", "rag", "faiss", "pinecone",
    "elasticsearch", "ndcg", "mrr", "a/b test", "mlops", "inference"
]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def is_consulting(company: str) -> bool:
    c = company.lower()
    return any(f in c for f in CONSULTING_FIRMS)

def score_title(title: str) -> float:
    t = title.lower()
    if any(p in t for p in STRONG_TITLES): return 1.0
    if any(p in t for p in GOOD_TITLES):   return 0.75
    if any(p in t for p in WEAK_TITLES):   return 0.35
    if any(p in t for p in NO_TITLES):     return 0.0
    return 0.2

def score_experience(yoe: float) -> float:
    if yoe < 3:     return 0.2
    elif yoe <= 4:  return 0.5
    elif yoe <= 9:  return 1.0
    elif yoe <= 12: return 0.8
    else:           return 0.5

def score_location(loc: str, country: str, relocate: bool) -> float:
    l = loc.lower()
    base = 0.1
    if any(c in l for c in ["pune", "noida"]):
        base = 1.0
    elif any(c in l for c in ["hyderabad", "mumbai", "delhi", "bangalore",
                                "bengaluru", "gurugram", "gurgaon", "chennai"]):
        base = 0.8
    elif country.lower() == "india" or "india" in l:
        base = 0.6
    if relocate:
        base = min(1.0, base + 0.15)
    return base

def behavioral_mult(signals: dict) -> float:
    today = date.today()
    mult  = 1.0
    lad = signals.get("last_active_date")
    if lad:
        try:
            days = (today - datetime.strptime(lad, "%Y-%m-%d").date()).days
            if days <= 30:    mult *= 1.15
            elif days <= 90:  mult *= 1.0
            elif days <= 180: mult *= 0.85
            else:             mult *= 0.6
        except Exception:
            pass
    if signals.get("open_to_work_flag"):   mult *= 1.1
    rr = signals.get("recruiter_response_rate", 0.5)
    if rr >= 0.7: mult *= 1.1
    elif rr < 0.3: mult *= 0.8
    np_d = signals.get("notice_period_days", 60)
    if np_d <= 30:   mult *= 1.05
    elif np_d > 90:  mult *= 0.85
    gas = signals.get("github_activity_score", -1)
    if gas >= 50: mult *= 1.05
    elif gas == -1: mult *= 0.95
    return round(max(0.4, min(1.3, mult)), 4)

def build_text(c: dict) -> str:
    parts = []
    p = c["profile"]
    if p.get("headline"):     parts.append(p["headline"])
    if p.get("summary"):      parts.append(p["summary"])
    if p.get("current_title"):
        parts.append(f"Current role: {p['current_title']} at {p.get('current_company','')}")
    for role in c.get("career_history", []):
        t = role.get("title","")
        co = role.get("company","")
        d = role.get("description","")
        if t or d:
            parts.append(f"{t} at {co}: {d}".strip())
    return " | ".join(parts)

def career_kw(career: list) -> list:
    found = set()
    for role in career:
        desc = role.get("description","").lower()
        for kw in CAREER_KEYWORDS:
            if kw in desc:
                found.add(kw)
    return sorted(found)

def make_reasoning(row) -> str:
    parts = []
    ts = row["title_score"]
    if ts >= 1.0:
        parts.append(f"{row['current_title']} at {row['current_company']} with {row['yoe']:.0f} yrs experience — strong AI/ML role alignment.")
    elif ts >= 0.75:
        parts.append(f"{row['current_title']} at {row['current_company']} with {row['yoe']:.0f} yrs experience — adjacent ML background.")
    else:
        parts.append(f"{row['current_title']} at {row['current_company']} with {row['yoe']:.0f} yrs experience.")
    sigs = []
    kws = [k for k in row["kw_hits"].split("|") if k]
    if kws:  sigs.append(f"career mentions {', '.join(kws[:3])}")
    if row["has_product"]: sigs.append("product-company background")
    if row["open_to_work"]: sigs.append("open to work")
    if row["notice"] <= 30: sigs.append(f"{row['notice']}-day notice")
    if sigs: parts.append("Signals: " + "; ".join(sigs) + ".")
    return " ".join(parts)

@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data(show_spinner="Computing JD embedding...")
def get_jd_embedding():
    model = load_model()
    return model.encode(JD_TEXT, normalize_embeddings=True)

def rank_candidates(candidates: list) -> pd.DataFrame:
    model = load_model()
    jd_emb = get_jd_embedding()

    texts = [build_text(c) for c in candidates]
    embs  = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sims  = embs @ jd_emb

    rows = []
    for i, c in enumerate(candidates):
        p       = c["profile"]
        signals = c["redrob_signals"]
        career  = c.get("career_history", [])
        ts = score_title(p.get("current_title",""))
        es = score_experience(p.get("years_of_experience", 0))
        ls = score_location(p.get("location",""), p.get("country",""),
                            signals.get("willing_to_relocate", False))
        bm = behavioral_mult(signals)
        has_product = any(not is_consulting(r.get("company","")) for r in career)
        at_con = is_consulting(p.get("current_company",""))

        base = 0.50*sims[i] + 0.30*ts + 0.10*es + 0.10*ls
        score = base * bm
        if at_con and not has_product: score *= 0.4
        elif at_con: score *= 0.7

        rows.append({
            "candidate_id":   c["candidate_id"],
            "current_title":  p.get("current_title",""),
            "current_company":p.get("current_company",""),
            "yoe":            p.get("years_of_experience", 0),
            "location":       p.get("location",""),
            "title_score":    ts,
            "score":          round(float(score), 6),
            "has_product":    has_product,
            "open_to_work":   signals.get("open_to_work_flag", False),
            "notice":         signals.get("notice_period_days", 90),
            "kw_hits":        "|".join(career_kw(career)),
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    top_n = min(10, len(df))
    df = df.head(top_n).copy()
    df.insert(0, "rank", range(1, top_n+1))
    df["reasoning"] = df.apply(make_reasoning, axis=1)
    return df

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🎯 HireSignal — AI Candidate Ranker")
st.caption("Redrob Hackathon — INDIA.RUNS Data & AI Challenge | by Anusha Puppala")

st.markdown("""
**What this does:** Ranks candidates for a Senior AI Engineer role the way a 
great recruiter would — using semantic understanding of career history, 
not keyword matching.
""")

with st.expander("📋 Job Description", expanded=False):
    st.markdown(f"```\n{JD_TEXT}\n```")

st.divider()

# Input
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Input candidates")
    mode = st.radio("Source", ["Use built-in 50-candidate sample", "Upload your own JSONL"])

candidates = []

if mode == "Use built-in 50-candidate sample":
    sample_path = Path(__file__).parent / "sample_candidates.json"
    if sample_path.exists():
        with open(sample_path) as f:
            candidates = json.load(f)
        st.success(f"Loaded {len(candidates)} sample candidates")
    else:
        st.error("sample_candidates.json not found in sandbox folder.")
else:
    uploaded = st.file_uploader("Upload candidates (.jsonl)", type=["jsonl", "json"])
    if uploaded:
        content = uploaded.read().decode("utf-8")
        if uploaded.name.endswith(".jsonl"):
            candidates = [json.loads(l) for l in content.splitlines() if l.strip()]
        else:
            candidates = json.loads(content)
        candidates = candidates[:200]  # cap at 200 for sandbox speed
        st.success(f"Loaded {len(candidates)} candidates (capped at 200 for sandbox)")

with col2:
    st.subheader("Scoring weights")
    st.markdown("""
| Component | Weight | Why |
|---|---|---|
| Semantic similarity | 50% | Career text vs JD understanding |
| Title fit | 30% | Primary role alignment signal |
| Experience fit | 10% | Years in JD target band (5-9) |
| Location fit | 10% | India presence / relocation |

*Skills excluded — EDA showed random assignment in dataset.*
""")

st.divider()

if candidates:
    if st.button("🚀 Rank Candidates", type="primary"):
        with st.spinner("Ranking candidates..."):
            results = rank_candidates(candidates)

        st.subheader(f"🏆 Top {len(results)} Candidates")

        # Display table
        display_df = results[["rank", "candidate_id", "current_title",
                               "current_company", "yoe", "score", "reasoning"]].copy()
        display_df.columns = ["Rank", "ID", "Title", "Company", "YoE", "Score", "Reasoning"]
        display_df["Score"] = display_df["Score"].round(4)
        display_df["YoE"] = display_df["YoE"].astype(int)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download
        csv_buf = io.StringIO()
        results[["candidate_id","rank","score","reasoning"]].to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download ranked CSV",
            csv_buf.getvalue(),
            file_name="hiresignal_results.csv",
            mime="text/csv"
        )

        # Stats
        st.divider()
        st.subheader("📊 Quick stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candidates ranked", len(candidates))
        c2.metric("Top score", f"{results['score'].max():.3f}")
        c3.metric("Strong title matches", int((results['title_score']==1.0).sum()))
        c4.metric("Product-co backgrounds", int(results['has_product'].sum()))
else:
    st.info("👆 Select a candidate source above and click Rank Candidates.")

st.divider()
st.caption("GitHub: github.com/anushapuppala21/redrob-ranker")
