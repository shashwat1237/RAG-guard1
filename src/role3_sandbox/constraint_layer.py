"""
Module 4.5 -- Deterministic Constraint Layer
"""
import spacy
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
from sentence_transformers import CrossEncoder
from scipy.special import softmax

# Load models globally to avoid overhead per call
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Phase 1: Injection Intent Model
tokenizer = AutoTokenizer.from_pretrained("hlyn-labs/prompt-injection-judge-deberta-70m")
# Assumes the quantized model is saved as 'intent_judge_int8.onnx' in the project root.
try:
    ort_session = ort.InferenceSession("intent_judge_int8.onnx")
except Exception as e:
    print(f"[Constraint Layer] ONNX model not found, skipping intent phase. Error: {e}")
    ort_session = None

# Phase 2: NLI Cross-Encoder
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

def _is_imperative(text: str) -> bool:
    """Dependency parsing to detect imperative root verbs."""
    doc = nlp(text)
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            # If the root verb has no nominal subject, it's imperative
            if not any(child.dep_ == "nsubj" for child in token.children):
                return True
    return False

def _get_injection_prob(text: str) -> float:
    """Runs the INT8 ONNX graph for intent classification."""
    if ort_session is None:
         return 0.0
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True)
    ort_inputs = {
        ort_session.get_inputs()[0].name: inputs["input_ids"],
        ort_session.get_inputs()[1].name: inputs["attention_mask"]
    }
    logits = ort_session.run(None, ort_inputs)[0]
    return float(softmax(logits, axis=1)[0][1])

def apply_deterministic_constraints(chunks: list[dict], intent_threshold=0.45, conflict_threshold=0.72) -> list[dict]:
    """Applies syntactic gating and pairwise NLI contradiction graphing."""
    if not chunks:
        return []

    # --- Phase 1: Syntactic Gating & Intent Classification ---
    syntactically_safe = []
    for chunk in chunks:
        text = chunk["text"]
        if _is_imperative(text):
            prob = _get_injection_prob(text)
            if prob > intent_threshold:
                print(f"[Module 4.5] Dropped chunk {chunk['chunk_id']}: Imperative injection detected (prob: {prob:.2f})")
                continue
        syntactically_safe.append(chunk)

    if len(syntactically_safe) < 2:
        return syntactically_safe

    # --- Phase 2: Pairwise Contradiction Graphing ---
    n = len(syntactically_safe)
    pairs = []
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((syntactically_safe[i]["text"], syntactically_safe[j]["text"]))
            pair_indices.append((i, j))

    # Output logits: [Contradiction, Entailment, Neutral]
    scores = nli_model.predict(pairs)
    probs = softmax(scores, axis=1)
    
    # Construct conflict graph edges E
    edges = set()
    for idx, (i, j) in enumerate(pair_indices):
        contradiction_prob = probs[idx][0]
        if contradiction_prob > conflict_threshold:
            edges.add((i, j))

    # --- Weighted Minimum Vertex Cover (Greedy Approximation) ---
    # We want to break all edges by removing vertices, minimizing the sum of authority lost.
    # Let G = (V, E). We greedily remove the node involved in an edge with the lowest authority score.
    dropped_indices = set()
    
    while edges:
        # Find all nodes currently involved in conflicts
        conflicted_nodes = set()
        for u, v in edges:
            conflicted_nodes.add(u)
            conflicted_nodes.add(v)
            
        # Pick the node with the absolute lowest authority score
        weakest_node = min(conflicted_nodes, key=lambda idx: syntactically_safe[idx].get("authority_score", 0))
        dropped_indices.add(weakest_node)
        print(f"[Module 4.5] Dropped chunk {syntactically_safe[weakest_node]['chunk_id']}: Resolving contradiction.")
        
        # Remove all edges connected to the dropped node
        edges = {(u, v) for u, v in edges if u != weakest_node and v != weakest_node}

    return [chunk for i, chunk in enumerate(syntactically_safe) if i not in dropped_indices]
