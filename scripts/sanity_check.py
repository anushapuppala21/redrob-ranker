"""
sanity_check.py
===============
Run this FIRST after setting up your environment.
Verifies that all libraries are installed correctly and the
embedding model works end-to-end on a tiny sample.

Usage:
    python scripts/sanity_check.py

Expected output:
    [1/4] Libraries OK
    [2/4] Sample data loaded — 50 candidates
    [3/4] Embedding model loaded — all-MiniLM-L6-v2
    [4/4] Embedding shape: (384,)
    
    All checks passed. Environment is ready.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

def check_libraries():
    try:
        import numpy as np
        import pandas as pd
        import pyarrow
        import tqdm
        import openpyxl
        print("[1/5] Core libraries OK")
    except ImportError as e:
        print(f"[1/5] FAILED — missing library: {e}")
        print("      Run: pip install -r requirements.txt")
        sys.exit(1)

def check_sample_data():
    # Try sample_candidates.json first
    sample_path = ROOT / "data" / "sample" / "sample_candidates.json"
    if not sample_path.exists():
        print(f"[2/5] WARNING — sample file not found at {sample_path}")
        print("      Copy sample_candidates.json into data/sample/")
        return None
    with open(sample_path) as f:
        candidates = json.load(f)
    print(f"[2/5] Sample data loaded — {len(candidates)} candidates")
    return candidates

def check_full_data():
    full_path = ROOT / "data" / "candidates.jsonl"
    if not full_path.exists():
        print("[3/5] WARNING — candidates.jsonl not found in data/")
        print("      Place the full dataset there before running precompute.py")
    else:
        size_mb = full_path.stat().st_size / (1024 ** 2)
        print(f"[3/5] Full dataset found — {size_mb:.0f} MB")

def check_embedding_model(candidates):
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        print("[4/5] Loading embedding model (downloads once if not cached)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        if candidates:
            text = candidates[0]["profile"]["summary"]
        else:
            text = "Senior AI engineer with experience in embeddings and retrieval systems."

        embedding = model.encode(text)
        print(f"[4/5] Embedding model OK — shape: {embedding.shape}")
        return True
    except Exception as e:
        print(f"[4/5] FAILED — {e}")
        return False

def check_excel_write():
    try:
        import pandas as pd
        test_df = pd.DataFrame({
            "candidate_id": ["CAND_0000001"],
            "rank": [1],
            "score": [0.95],
            "reasoning": ["Test reasoning sentence."]
        })
        out_path = ROOT / "outputs" / "test_output.xlsx"
        out_path.parent.mkdir(exist_ok=True)
        test_df.to_excel(out_path, index=False)
        out_path.unlink()  # clean up
        print("[5/5] Excel write OK (openpyxl working)")
    except Exception as e:
        print(f"[5/5] FAILED — Excel write error: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("Redrob Ranker — Environment Sanity Check")
    print("=" * 50)

    check_libraries()
    candidates = check_sample_data()
    check_full_data()
    ok = check_embedding_model(candidates)
    check_excel_write()

    print()
    if ok:
        print("All checks passed. Environment is ready.")
        print("Next step: python scripts/precompute.py")
    else:
        print("Fix the issues above before proceeding.")
