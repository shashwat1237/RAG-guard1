"""
Orchestrates Module 4 (Authority) and Module 4.5 (Constraint Layer).
The generative LLM Sandbox (Module 5) has been permanently removed.
"""
from .authority import apply_authority_checks
from .constraint_layer import apply_deterministic_constraints

def apply_sandbox_filters(math_filtered_chunks: list[dict], user_query: str) -> list[dict]:
    """
    Applies trust scores, drops tampered hashes, and resolves semantic conflicts deterministically.
    """
    # --- MODULE 4: AUTHORITY & HASH REJECTION ---
    scored = apply_authority_checks(math_filtered_chunks)
    
    trusted_chunks = []
    for c in scored:
        if not c.get("hash_valid", True):
            print(f"[authority] Dropped chunk {c['chunk_id']}: Invalid Hash.")
            continue
        trusted_chunks.append(c)

    if not trusted_chunks:
        return []

    # --- MODULE 4.5: DETERMINISTIC CONSTRAINT LAYER ---
    print("[sandbox] Running deterministic syntactic and conflict checks...")
    final_chunks = apply_deterministic_constraints(trusted_chunks)
    
    return final_chunks
