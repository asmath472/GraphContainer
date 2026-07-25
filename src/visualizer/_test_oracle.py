"""Standalone test for oracle.py – run from its own directory."""
import sys, os, logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
sys.path.insert(0, os.path.dirname(__file__))
from oracle import OracleResolver, STYLE_FASTINSIGHT, STYLE_LIGHTRAG, STYLE_HIPPORAG

DATASET_BASE   = "/mnt/disk2/cjhyun/InfoRAG/data/datasets"
RAG_BASE       = "/mnt/disk2/cjhyun/GraphContainer/data/rag_storage"
HIPPORAG_BASE  = "/mnt/disk2/cjhyun/HippoRAG/outputs"

OK   = "\033[92m✓ PASSED\033[0m"
FAIL = "\033[91m✗ FAILED\033[0m"

def sep(title): print("\n" + "="*60 + f"\n  {title}\n" + "="*60)

# ── TEST 1: fastinsight (Component Graph) ────────────────────────────────────
sep("TEST 1: fastinsight / Component Graph (acl)")
r = OracleResolver(
    style=STYLE_FASTINSIGHT,
    dataset_dir=f"{DATASET_BASE}/acl",
)
print("status:", r.status())
q1 = ("When training a transition-based parser using an oracle that maps "
      "dependency trees to transition sequences, how are individual training "
      "instances represented—i.e., what do the pairs (c, t) consist of?")
res1 = r.resolve(q1)
print("gold nodes:", res1)
print(OK if res1 == ["9078664_0", "9431510_3"] else FAIL + f" got {res1}")

# ── TEST 2: hipporag / Topology-Semantic Graph (hotpotqa) ────────────────────
sep("TEST 2: hipporag / Topology-Semantic Graph (hotpotqa)")
r2 = OracleResolver(
    style=STYLE_HIPPORAG,
    dataset_dir=f"{DATASET_BASE}/hotpotqa",
    hipporag_outputs_dir=f"{HIPPORAG_BASE}/hotpotqa",
)
print("status:", r2.status())

q2a = "what is one of the stars of  The Newcomers known for"
res2a = r2.resolve(q2a)
print("q2a gold (first 5):", res2a[:5])
ok2a = "Chris Evans" in res2a and "The Newcomers" in res2a
print(OK if ok2a else FAIL + f" Chris Evans present: {'Chris Evans' in res2a}")

q2b = 'The fictional private detective that appears in "The Adventure of the Seven Clocks" what written by whom?'
res2b = r2.resolve(q2b)
print("q2b gold (first 5):", res2b[:5])
ok2b = "Sherlock Holmes" in res2b and "Sir Arthur Conan Doyle" in res2b
print(OK if ok2b else FAIL + f" Sherlock Holmes present: {'Sherlock Holmes' in res2b}")

# also test resolve_with_edges (hipporag returns empty edges)
nodes, edges = r2.resolve_with_edges(q2a)
print(f"resolve_with_edges: {len(nodes)} nodes, {len(edges)} edges (expected 0 edges for hipporag)")
print(OK if edges == [] else FAIL + " expected no edges")

# ── TEST 3: lightrag / Attribute Bundle Graph (scifact) ──────────────────────
sep("TEST 3: lightrag / Attribute Bundle Graph (scifact)")
r3 = OracleResolver(
    style=STYLE_LIGHTRAG,
    dataset_dir=f"{DATASET_BASE}/scifact",
    rag_storage_dir=f"{RAG_BASE}/fastinsight/scifact-bge-m3",
)
print("status:", r3.status())
# scifact-bge-m3 doesn't have kv_store files → resolve returns []
q3 = "A breast cancer patient's capacity to metabolize tamoxifen influences treatment outcome."
res3, edges3 = r3.resolve_with_edges(q3)
print(f"gold nodes: {res3[:5] if res3 else '[] (no kv_store files in this rag_storage path)'}")
print(OK + " (loaded without crash)")

# ── TEST 4: invalid style raises ValueError ───────────────────────────────────
sep("TEST 4: invalid style raises ValueError")
try:
    OracleResolver(style="bad_style", dataset_dir=f"{DATASET_BASE}/acl")
    print(FAIL + " should have raised ValueError")
except ValueError as e:
    print(OK, "ValueError:", e)

print("\n" + "="*60)
print("ALL TESTS DONE")
