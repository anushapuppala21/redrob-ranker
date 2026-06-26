"""
01_eda.py — Candidate Pool Exploration
========================================
Run this script to understand the 100K candidate dataset before
building the scoring system. Each section prints findings that
directly feed jd_config.json.

Usage:
    python notebooks/01_eda.py

Sections:
    1. Load data
    2. Basic stats
    3. Title & role distribution
    4. Company type (product vs consulting)
    5. Location distribution
    6. Experience distribution
    7. Skills landscape
    8. Behavioral signals
    9. Honeypot detection patterns
    10. Key findings summary
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "candidates.jsonl"
SAMPLE_PATH = ROOT / "data" / "sample" / "sample_candidates.json"

# ─────────────────────────────────────────────
# SECTION 1: Load data
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 1 — Loading data")
print("="*60)

candidates = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            candidates.append(json.loads(line))

print(f"Total candidates loaded: {len(candidates):,}")
print(f"Sample fields: {list(candidates[0].keys())}")

# ─────────────────────────────────────────────
# SECTION 2: Basic stats
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 2 — Basic stats")
print("="*60)

countries = Counter(c["profile"]["country"] for c in candidates)
print("\nTop 10 countries:")
for country, count in countries.most_common(10):
    print(f"  {country:<30} {count:>7,}  ({count/len(candidates)*100:.1f}%)")

industries = Counter(c["profile"]["current_industry"] for c in candidates)
print("\nTop 15 industries:")
for ind, count in industries.most_common(15):
    print(f"  {ind:<45} {count:>6,}")

company_sizes = Counter(c["profile"]["current_company_size"] for c in candidates)
print("\nCompany size distribution:")
for size, count in sorted(company_sizes.items()):
    print(f"  {size:<15} {count:>7,}  ({count/len(candidates)*100:.1f}%)")

# ─────────────────────────────────────────────
# SECTION 3: Title & role distribution
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 3 — Title & role distribution")
print("="*60)

titles = Counter(c["profile"]["current_title"] for c in candidates)
print("\nTop 30 current titles:")
for title, count in titles.most_common(30):
    print(f"  {title:<55} {count:>6,}")

# Broad role buckets
AI_ML_PATTERNS = [
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp", "deep learning", "research scientist", "applied scientist",
    "ml researcher", "ai researcher", "computer vision", "mlops",
    "llm", "generative ai", "prompt engineer"
]
DATA_ENG_PATTERNS = [
    "data engineer", "data architect", "etl", "pipeline", "spark",
    "big data", "analytics engineer"
]
SOFTWARE_PATTERNS = [
    "software engineer", "software developer", "backend", "frontend",
    "full stack", "fullstack", "sde ", "swe "
]
CONSULTING_PATTERNS = [
    "consultant", "analyst", "associate", "manager", "business"
]
UNRELATED_PATTERNS = [
    "marketing", "sales", "hr ", "human resource", "recruiter",
    "accountant", "finance", "content", "designer", "product manager"
]

def classify_title(title):
    t = title.lower()
    if any(p in t for p in AI_ML_PATTERNS):
        return "AI/ML"
    if any(p in t for p in DATA_ENG_PATTERNS):
        return "Data Engineering"
    if any(p in t for p in SOFTWARE_PATTERNS):
        return "Software Engineering"
    if any(p in t for p in CONSULTING_PATTERNS):
        return "Consulting/Management"
    if any(p in t for p in UNRELATED_PATTERNS):
        return "Unrelated"
    return "Other"

role_buckets = Counter(classify_title(c["profile"]["current_title"]) for c in candidates)
print("\nRole bucket distribution:")
for bucket, count in role_buckets.most_common():
    print(f"  {bucket:<30} {count:>7,}  ({count/len(candidates)*100:.1f}%)")

# ─────────────────────────────────────────────
# SECTION 4: Company type (product vs consulting)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 4 — Company type: product vs consulting")
print("="*60)

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies", "tech mahindra",
    "mphasis", "hexaware", "niit", "l&t infotech", "mindtree",
    "ltimindtree", "persistent systems", "birlasoft", "coforge",
    "kpit", "cyient", "sonata", "zensar", "mastech", "igate"
}

def is_consulting(company_name):
    cn = company_name.lower().strip()
    return any(firm in cn for firm in CONSULTING_FIRMS)

companies = Counter(c["profile"]["current_company"] for c in candidates)
print(f"\nUnique companies: {len(companies):,}")
print("\nTop 30 current companies:")
for co, count in companies.most_common(30):
    flag = " ← CONSULTING" if is_consulting(co) else ""
    print(f"  {co:<45} {count:>5,}{flag}")

# Classify all candidates
consulting_count = sum(1 for c in candidates if is_consulting(c["profile"]["current_company"]))
product_count = len(candidates) - consulting_count
print(f"\nConsulting-firm candidates: {consulting_count:,} ({consulting_count/len(candidates)*100:.1f}%)")
print(f"Non-consulting candidates:  {product_count:,} ({product_count/len(candidates)*100:.1f}%)")

# ─────────────────────────────────────────────
# SECTION 5: Location distribution
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 5 — Location distribution")
print("="*60)

locations = Counter(c["profile"]["location"] for c in candidates)
print("\nTop 25 locations:")
for loc, count in locations.most_common(25):
    print(f"  {loc:<40} {count:>6,}")

# JD-preferred locations
PREFERRED_LOCS = ["pune", "noida"]
ACCEPTABLE_LOCS = ["hyderabad", "mumbai", "delhi", "bangalore", "bengaluru", "chennai", "gurugram", "gurgaon"]

def classify_location(loc):
    l = loc.lower()
    if any(p in l for p in PREFERRED_LOCS):
        return "preferred"
    if any(p in l for p in ACCEPTABLE_LOCS):
        return "acceptable"
    if "india" in l or any(city in l for city in ["kolkata", "ahmedabad", "jaipur", "lucknow", "kochi"]):
        return "other_india"
    return "outside_india_or_unknown"

loc_buckets = Counter(classify_location(c["profile"]["location"]) for c in candidates)
willing_relocate = sum(1 for c in candidates if c["redrob_signals"].get("willing_to_relocate", False))
print("\nLocation fit distribution:")
for bucket, count in loc_buckets.most_common():
    print(f"  {bucket:<30} {count:>7,}  ({count/len(candidates)*100:.1f}%)")
print(f"\n  Willing to relocate:         {willing_relocate:>7,}  ({willing_relocate/len(candidates)*100:.1f}%)")

# ─────────────────────────────────────────────
# SECTION 6: Experience distribution
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 6 — Experience distribution")
print("="*60)

yoe_values = [c["profile"]["years_of_experience"] for c in candidates]
buckets = {"0-2": 0, "3-4": 0, "5-6": 0, "7-9": 0, "10-14": 0, "15+": 0}
for y in yoe_values:
    if y <= 2:   buckets["0-2"] += 1
    elif y <= 4: buckets["3-4"] += 1
    elif y <= 6: buckets["5-6"] += 1
    elif y <= 9: buckets["7-9"] += 1
    elif y <= 14:buckets["10-14"] += 1
    else:        buckets["15+"] += 1

print("\nExperience distribution:")
for band, count in buckets.items():
    bar = "█" * (count // 500)
    print(f"  {band:<8} {count:>7,}  {bar}")

jd_band = sum(1 for y in yoe_values if 5 <= y <= 9)
print(f"\n  In JD target band (5-9 yrs): {jd_band:,} ({jd_band/len(candidates)*100:.1f}%)")

avg_yoe = sum(yoe_values) / len(yoe_values)
print(f"  Mean YoE across all: {avg_yoe:.1f}")

# ─────────────────────────────────────────────
# SECTION 7: Skills landscape
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 7 — Skills landscape")
print("="*60)

all_skills = []
for c in candidates:
    for s in c.get("skills", []):
        all_skills.append(s["name"].lower().strip())

skill_counts = Counter(all_skills)
print(f"\nUnique skills in dataset: {len(skill_counts):,}")
print("\nTop 40 skills:")
for skill, count in skill_counts.most_common(40):
    pct = count / len(candidates) * 100
    print(f"  {skill:<45} {count:>6,}  ({pct:.1f}%)")

# JD must-have skills check
MUST_HAVE_SKILLS = [
    "embeddings", "vector", "retrieval", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "elasticsearch", "opensearch", "sentence-transformers",
    "semantic search", "hybrid search", "ranking", "ndcg", "python",
    "transformer", "bert", "llm", "fine-tuning", "lora", "rag"
]
print("\nJD must-have skill prevalence in pool:")
for skill in MUST_HAVE_SKILLS:
    matches = sum(1 for s in all_skills if skill in s)
    print(f"  {skill:<35} {matches:>6,}  ({matches/len(candidates)*100:.1f}%)")

# ─────────────────────────────────────────────
# SECTION 8: Behavioral signals
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 8 — Behavioral signals")
print("="*60)

today = date.today()

# Last active recency
days_inactive = []
for c in candidates:
    lad = c["redrob_signals"].get("last_active_date")
    if lad:
        try:
            d = datetime.strptime(lad, "%Y-%m-%d").date()
            days_inactive.append((today - d).days)
        except:
            pass

if days_inactive:
    inactive_buckets = {"Active (≤30d)": 0, "Recent (31-90d)": 0,
                        "Stale (91-180d)": 0, "Dead (>180d)": 0}
    for d in days_inactive:
        if d <= 30:   inactive_buckets["Active (≤30d)"] += 1
        elif d <= 90: inactive_buckets["Recent (31-90d)"] += 1
        elif d <= 180:inactive_buckets["Stale (91-180d)"] += 1
        else:         inactive_buckets["Dead (>180d)"] += 1
    print("\nRecency of last login:")
    for bucket, count in inactive_buckets.items():
        print(f"  {bucket:<25} {count:>7,}  ({count/len(candidates)*100:.1f}%)")

# Open to work
open_to_work = sum(1 for c in candidates if c["redrob_signals"].get("open_to_work_flag"))
print(f"\n  Open to work:                {open_to_work:>7,}  ({open_to_work/len(candidates)*100:.1f}%)")

# Response rate distribution
rr_values = [c["redrob_signals"].get("recruiter_response_rate", 0) for c in candidates]
high_rr = sum(1 for r in rr_values if r >= 0.7)
low_rr  = sum(1 for r in rr_values if r < 0.3)
print(f"  High response rate (≥70%):  {high_rr:>7,}  ({high_rr/len(candidates)*100:.1f}%)")
print(f"  Low response rate (<30%):   {low_rr:>7,}  ({low_rr/len(candidates)*100:.1f}%)")

# Notice period
notice_values = [c["redrob_signals"].get("notice_period_days", 90) for c in candidates]
notice_buckets = {"0-15d": 0, "16-30d": 0, "31-60d": 0, "61-90d": 0, "90+d": 0}
for n in notice_values:
    if n <= 15:   notice_buckets["0-15d"] += 1
    elif n <= 30: notice_buckets["16-30d"] += 1
    elif n <= 60: notice_buckets["31-60d"] += 1
    elif n <= 90: notice_buckets["61-90d"] += 1
    else:         notice_buckets["90+d"] += 1
print("\nNotice period distribution (JD prefers ≤30 days):")
for bucket, count in notice_buckets.items():
    print(f"  {bucket:<12} {count:>7,}  ({count/len(candidates)*100:.1f}%)")

# GitHub activity
github_linked   = sum(1 for c in candidates if c["redrob_signals"].get("github_activity_score", -1) >= 0)
github_active   = sum(1 for c in candidates if c["redrob_signals"].get("github_activity_score", -1) >= 50)
print(f"\n  GitHub linked:               {github_linked:>7,}  ({github_linked/len(candidates)*100:.1f}%)")
print(f"  GitHub active (score ≥50):   {github_active:>7,}  ({github_active/len(candidates)*100:.1f}%)")

# ─────────────────────────────────────────────
# SECTION 9: Honeypot detection patterns
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 9 — Honeypot detection patterns")
print("="*60)

def get_red_flags(c):
    flags = []
    profile = c["profile"]
    signals = c["redrob_signals"]
    career  = c.get("career_history", [])
    skills  = c.get("skills", [])

    yoe = profile.get("years_of_experience", 0)

    # 1. Expert skill with 0 months duration
    expert_zero = [s for s in skills
                   if s.get("proficiency") == "expert"
                   and s.get("duration_months", 1) == 0]
    if expert_zero:
        flags.append(f"expert_skill_zero_duration({len(expert_zero)})")

    # 2. Too many expert skills (keyword stuffer)
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 15:
        flags.append(f"too_many_expert_skills({expert_count})")

    # 3. Career history duration vs stated YoE mismatch
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    stated_months = yoe * 12
    if stated_months > 0 and total_career_months > stated_months * 1.4:
        flags.append(f"career_months_exceed_yoe({total_career_months}vs{stated_months:.0f})")

    # 4. Overlapping job dates
    date_ranges = []
    for role in career:
        try:
            start = datetime.strptime(role["start_date"], "%Y-%m-%d").date()
            end_raw = role.get("end_date")
            end = datetime.strptime(end_raw, "%Y-%m-%d").date() if end_raw else date.today()
            date_ranges.append((start, end))
        except:
            pass
    date_ranges.sort()
    for i in range(len(date_ranges) - 1):
        if date_ranges[i][1] > date_ranges[i+1][0]:
            flags.append("overlapping_job_dates")
            break

    # 5. End date before start date in any role
    for role in career:
        try:
            start = datetime.strptime(role["start_date"], "%Y-%m-%d").date()
            end_raw = role.get("end_date")
            if end_raw:
                end = datetime.strptime(end_raw, "%Y-%m-%d").date()
                if end < start:
                    flags.append("end_before_start")
                    break
        except:
            pass

    # 6. Signal values outside documented range
    if not (0 <= signals.get("profile_completeness_score", 50) <= 100):
        flags.append("completeness_out_of_range")
    if not (0 <= signals.get("recruiter_response_rate", 0.5) <= 1):
        flags.append("response_rate_out_of_range")
    oar = signals.get("offer_acceptance_rate", 0)
    if not (-1 <= oar <= 1):
        flags.append("offer_acceptance_out_of_range")
    gas = signals.get("github_activity_score", 0)
    if not (-1 <= gas <= 100):
        flags.append("github_score_out_of_range")

    # 7. YoE impossibly high (>45)
    if yoe > 45:
        flags.append(f"impossible_yoe({yoe})")

    # 8. Profile completeness very high but no career history
    if signals.get("profile_completeness_score", 0) > 90 and len(career) == 0:
        flags.append("complete_profile_no_career")

    return flags

print("\nScanning all 100K candidates for red flags...")
flag_counter = Counter()
flagged_candidates = []

for c in candidates:
    flags = get_red_flags(c)
    if flags:
        flagged_candidates.append((c["candidate_id"], flags))
        for f in flags:
            # normalize flag name without number
            flag_name = re.sub(r'\(.*?\)', '', f)
            flag_counter[flag_name] += 1

print(f"\nTotal flagged candidates: {len(flagged_candidates):,}")
print("\nFlag type breakdown:")
for flag, count in flag_counter.most_common():
    print(f"  {flag:<45} {count:>6,}")

# Candidates with 2+ flags (likely honeypots)
multi_flagged = [(cid, flags) for cid, flags in flagged_candidates if len(flags) >= 2]
print(f"\nCandidates with 2+ flags (likely honeypots): {len(multi_flagged):,}")
print("\nSample multi-flagged candidates:")
for cid, flags in multi_flagged[:10]:
    print(f"  {cid}: {flags}")

# ─────────────────────────────────────────────
# SECTION 10: Key findings summary
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 10 — KEY FINDINGS FOR jd_config.json")
print("="*60)

in_jd_band = sum(1 for y in yoe_values if 5 <= y <= 9)
ai_ml_count = role_buckets.get("AI/ML", 0)
active_count = inactive_buckets.get("Active (≤30d)", 0) if days_inactive else 0

print(f"""
Dataset snapshot:
  Total candidates         : {len(candidates):,}
  In JD experience band    : {in_jd_band:,} ({in_jd_band/len(candidates)*100:.1f}%)
  AI/ML role titles        : {ai_ml_count:,} ({ai_ml_count/len(candidates)*100:.1f}%)
  At consulting firms      : {consulting_count:,} ({consulting_count/len(candidates)*100:.1f}%)
  Active last 30 days      : {active_count:,} ({active_count/len(candidates)*100:.1f}%)
  Open to work             : {open_to_work:,} ({open_to_work/len(candidates)*100:.1f}%)
  Flagged (potential HP)   : {len(flagged_candidates):,} ({len(flagged_candidates)/len(candidates)*100:.1f}%)
  Multi-flagged (≥2 flags) : {len(multi_flagged):,} (likely ~80 real honeypots)

Next step: Use these findings to fill in scripts/jd_config.json
""")

print("EDA complete.")
