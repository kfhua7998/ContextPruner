"""
Complete evaluation on SQuAD dataset
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re
import random
from typing import List, Tuple
from datasets import load_dataset
import time


# ============ 1. Set random seeds ============
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============ 2. Helper functions ============

def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def get_sentence_importance(question: str, sentences: List[str], embedder) -> List[float]:
    if not sentences:
        return []
    q_emb = embedder.encode([question])[0]
    s_embs = embedder.encode(sentences)
    return cosine_similarity([q_emb], s_embs)[0].tolist()


def filter_by_importance(sentences: List[str], scores: List[float], keep_ratio: float) -> Tuple[str, List[int]]:
    if keep_ratio >= 1.0:
        return " ".join(sentences), list(range(len(sentences)))
    n_keep = max(1, int(len(sentences) * keep_ratio))
    indices = np.argsort(scores)[::-1][:n_keep]
    kept = sorted(indices)
    return " ".join([sentences[i] for i in kept]), kept


def filter_random(sentences: List[str], keep_ratio: float) -> Tuple[str, List[int]]:
    if keep_ratio >= 1.0:
        return " ".join(sentences), list(range(len(sentences)))
    n_keep = max(1, int(len(sentences) * keep_ratio))
    indices = sorted(random.sample(range(len(sentences)), n_keep))
    return " ".join([sentences[i] for i in indices]), indices


def shuffle_sentences(sentences: List[str]) -> str:
    shuffled = sentences.copy()
    random.shuffle(shuffled)
    return " ".join(shuffled)


def answer_question(question: str, context: str, llm, tokenizer) -> Tuple[str, float]:
    prompt = f"""Please answer in English. Answer based only on the context.

Context: {context}

Question: {question}

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(llm.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start_time
    
    generated = outputs[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    answer = answer.strip().split('\n')[0] if answer else "Not sure"
    
    return answer, elapsed


# ============ 3. Load models ============
print("=" * 60)
print("Loading models...")
print("=" * 60)

embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')
model_name = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Models loaded!")


# ============ 4. Load SQuAD dataset ============
print("\n" + "=" * 60)
print("Loading SQuAD dataset...")
print("=" * 60)

squad = load_dataset("squad", split="validation")

NUM_SAMPLES = 100
test_samples = []

for i in range(NUM_SAMPLES):
    answer_text = squad[i]["answers"]["text"][0]
    if len(answer_text) < 50:
        test_samples.append({
            "question": squad[i]["question"],
            "context": squad[i]["context"],
            "answer": answer_text.lower()
        })

print(f"Loaded {len(test_samples)} valid test samples")


# ============ 5. Run experiment ============
KEEP_RATIOS = [1.0, 0.7, 0.5, 0.3]

results = {
    ratio: {
        "baseline": {"correct": 0, "total": 0, "time": 0},
        "importance": {"correct": 0, "total": 0, "time": 0},
        "random": {"correct": 0, "total": 0, "time": 0},
        "shuffled": {"correct": 0, "total": 0, "time": 0},
        "avg_kept_ratio": 0
    }
    for ratio in KEEP_RATIOS
}

print("\n" + "=" * 60)
print("Starting experiment...")
print("=" * 60)

for ratio in KEEP_RATIOS:
    print(f"\n--- Testing keep ratio: {ratio:.0%} ---")
    
    for idx, sample in enumerate(tqdm(test_samples, desc=f"Ratio {ratio:.0%}")):
        question = sample["question"]
        context = sample["context"]
        answer = sample["answer"]
        
        sentences = split_sentences(context)
        if not sentences:
            continue
        
        scores = get_sentence_importance(question, sentences, embedder)
        
        # Baseline
        pred, elapsed = answer_question(question, context, llm, tokenizer)
        if answer in pred.lower():
            results[ratio]["baseline"]["correct"] += 1
        results[ratio]["baseline"]["total"] += 1
        results[ratio]["baseline"]["time"] += elapsed
        
        # Semantic pruning
        filtered_imp, kept_imp = filter_by_importance(sentences, scores, ratio)
        pred, elapsed = answer_question(question, filtered_imp, llm, tokenizer)
        if answer in pred.lower():
            results[ratio]["importance"]["correct"] += 1
        results[ratio]["importance"]["total"] += 1
        results[ratio]["importance"]["time"] += elapsed
        
        if ratio < 1.0:
            results[ratio]["avg_kept_ratio"] += len(kept_imp) / len(sentences)
        
        # Random pruning (only at 100% and 50%)
        if ratio in [0.5, 1.0]:
            filtered_rand, _ = filter_random(sentences, ratio)
            pred, elapsed = answer_question(question, filtered_rand, llm, tokenizer)
            if answer in pred.lower():
                results[ratio]["random"]["correct"] += 1
            results[ratio]["random"]["total"] += 1
            results[ratio]["random"]["time"] += elapsed
        
        # Shuffled (only at 100%)
        if ratio == 1.0:
            shuffled = shuffle_sentences(sentences)
            pred, elapsed = answer_question(question, shuffled, llm, tokenizer)
            if answer in pred.lower():
                results[ratio]["shuffled"]["correct"] += 1
            results[ratio]["shuffled"]["total"] += 1
            results[ratio]["shuffled"]["time"] += elapsed
    
    if results[ratio]["avg_kept_ratio"] > 0:
        results[ratio]["avg_kept_ratio"] /= len(test_samples)


# ============ 6. Print results ============
print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)

print("\n[Accuracy Comparison]")
print(f"\n{'Keep Ratio':<12} {'Baseline':<10} {'Semantic':<12} {'Random':<12} {'Shuffled':<12}")
print("-" * 60)

for ratio in KEEP_RATIOS:
    baseline_acc = results[ratio]["baseline"]["correct"] / results[ratio]["baseline"]["total"] if results[ratio]["baseline"]["total"] > 0 else 0
    imp_acc = results[ratio]["importance"]["correct"] / results[ratio]["importance"]["total"] if results[ratio]["importance"]["total"] > 0 else 0
    rand_acc = results[ratio]["random"]["correct"] / results[ratio]["random"]["total"] if results[ratio]["random"]["total"] > 0 else "-"
    shuffled_acc = results[ratio]["shuffled"]["correct"] / results[ratio]["shuffled"]["total"] if results[ratio]["shuffled"]["total"] > 0 else "-"
    
    rand_str = f"{rand_acc:.0%}" if rand_acc != "-" else "-"
    shuffled_str = f"{shuffled_acc:.0%}" if shuffled_acc != "-" else "-"
    
    print(f"{ratio:.0%} {' ':<8} {baseline_acc:.0%} {' ':<6} {imp_acc:.0%} {' ':<8} {rand_str:<10} {shuffled_str:<10}")

print("\n[Compression Effect]")
print(f"\n{'Keep Ratio':<12} {'Actual Kept Ratio':<18} {'Speedup':<12}")
print("-" * 42)

for ratio in KEEP_RATIOS:
    if ratio < 1.0:
        kept = results[ratio]["avg_kept_ratio"]
        speedup = 1.0 / kept if kept > 0 else 1.0
        print(f"{ratio:.0%} {' ':<8} {kept:.0%} {' ':<12} {speedup:.2f}x")
    else:
        print(f"{ratio:.0%} {' ':<8} 100% {' ':<12} 1.00x")

print("\n" + "=" * 60)
print("Experiment Complete!")
print("=" * 60)
