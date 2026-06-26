"""
precompute.py — Phase 1 (offline)
===================================
Pre-computes sentence-transformer embeddings and structured features
for all 100K candidates. Saves artifacts that rank.py loads.

Run once before ranking. Can take ~15-20 minutes. Network OK here.

Usage:
    python scripts/precompute.py

Outputs:
    artifacts/embeddings.npy       (float16, shape: 100000 x 384)
    artifacts/features.parquet     (structured per-candidate features)
    artifacts/jd_embedding.npy     (float32, shape: 384)
"""

# TODO: implement in Phase 2
# See architecture in README.md

print("precompute.py — to be implemented in Phase 2")
