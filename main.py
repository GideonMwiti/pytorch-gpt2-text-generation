#!/usr/bin/env python3
"""
Project 48: Text Generation with GPT-2
======================================
Autoregressive language modeling and open-ended text generation utilizing
the pretrained GPT-2 Transformer (124M parameters) from Hugging Face & PyTorch.

Key Highlights:
  - Pretrained GPT-2 causal language model architecture (12 layers, 768 hidden, 12 heads).
  - Comparative decoding strategy study:
      1. Greedy Search (argmax)
      2. Beam Search (num_beams=4 with n-gram blocking)
      3. Temperature-Scaled Sampling (T in [0.2, 0.7, 1.2])
      4. Top-k Filtering (k=50)
      5. Top-p (Nucleus) Filtering (p=0.9)
  - Quantitative generation metrics:
      * Token-level log-likelihood & sequence Perplexity (PPL)
      * Token confidence & entropy trajectories
      * Distinct-1 and Distinct-2 n-gram lexical diversity
      * Repetition degradation percentage
      * Generation latency & throughput (tokens/sec)
  - 4-Panel diagnostic visualization artifact.
  - Interactive CLI extension mode.
"""

import os
import sys
import json
import time
import math
import argparse
from typing import List, Dict, Tuple, Any

# Suppress Hugging Face symlink warnings on Windows & enforce offline mode
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPTS = os.path.join(SCRIPT_DIR, "data", "prompts.json")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "outputs", "gpt2_generation_comparison.png")

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")


# ==============================================================================
# 1. Lexical Diversity & Text Quality Metrics
# ==============================================================================

def tokenize_words(text: str) -> List[str]:
    """Extract alphanumeric words in lowercase."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


def compute_distinct_ngrams(tokens: List[str], n: int = 1) -> float:
    """Compute Distinct-n metric: ratio of unique n-grams to total n-grams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute_repetition_rate(tokens: List[str], n: int = 3) -> float:
    """Compute percentage of duplicate n-grams (repetition degradation)."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return ((total - unique) / total) * 100.0 if total > 0 else 0.0


# ==============================================================================
# 2. GPT-2 Generation Engine
# ==============================================================================

class GPT2TextGenerator:
    """Wrapper around Hugging Face GPT-2 Causal Language Model."""

    def __init__(self, model_name: str = "gpt2", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[*] Initializing GPT-2 Text Generator on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[+] Loaded '{model_name}' ({param_count:,} parameters) successfully.")

    def generate(
        self,
        prompt: str,
        strategy: str = "top_p",
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 2,
    ) -> Dict[str, Any]:
        """
        Generate text continuation using selected decoding strategy.
        Supports: 'greedy', 'beam', 'temperature', 'top_k', 'top_p'.
        """
        inputs = self.tokenizer(prompt.strip(), return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        strat = strategy.lower()
        if strat == "greedy":
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = 1
        elif strat == "beam":
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = num_beams
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        elif strat == "temperature":
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(0.01, temperature)
            gen_kwargs["top_k"] = 0
            gen_kwargs["top_p"] = 1.0
        elif strat == "top_k":
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = 1.0
        elif strat == "top_p":
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = 0
            gen_kwargs["top_p"] = top_p
        else:
            raise ValueError(f"Unknown generation strategy: '{strategy}'")

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - start_time

        gen_ids = outputs.sequences[0]
        new_ids = gen_ids[prompt_len:]
        full_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        continuation = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # ---------------------------------------------------------------------
        # Compute Sequence Perplexity & Token Confidence
        # ---------------------------------------------------------------------
        token_probs = []
        token_entropies = []
        log_likelihoods = []

        if hasattr(outputs, "scores") and outputs.scores:
            for step_idx, step_logits in enumerate(outputs.scores):
                # step_logits: (batch_size * num_beams, vocab_size)
                # First row represents leading beam/sequence
                logits = step_logits[0]
                probs = torch.softmax(logits, dim=-1)

                # Entropy: -sum(p * log p)
                log_probs = torch.log_softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum().item()
                token_entropies.append(entropy)

                # Probability of generated token
                if step_idx < len(new_ids):
                    target_id = new_ids[step_idx].item()
                    p_target = probs[target_id].item()
                    token_probs.append(p_target)
                    log_likelihoods.append(math.log(max(p_target, 1e-12)))

        avg_neg_log_lik = -np.mean(log_likelihoods) if log_likelihoods else 0.0
        perplexity = math.exp(avg_neg_log_lik) if log_likelihoods else 1.0

        all_words = tokenize_words(full_text)
        new_words = tokenize_words(continuation)

        dist1 = compute_distinct_ngrams(new_words, n=1)
        dist2 = compute_distinct_ngrams(new_words, n=2)
        rep3 = compute_repetition_rate(new_words, n=3)

        return {
            "strategy": strategy,
            "prompt": prompt,
            "continuation": continuation,
            "full_text": full_text,
            "num_new_tokens": len(new_ids),
            "latency_sec": elapsed,
            "throughput_tok_sec": len(new_ids) / elapsed if elapsed > 0 else 0.0,
            "perplexity": perplexity,
            "mean_token_prob": float(np.mean(token_probs)) if token_probs else 0.0,
            "token_probs": token_probs,
            "token_entropies": token_entropies,
            "distinct_1": dist1,
            "distinct_2": dist2,
            "repetition_3": rep3,
        }


# ==============================================================================
# 3. Diagnostic Visualization Suite
# ==============================================================================

def generate_visualization(
    benchmark_data: Dict[str, List[Dict[str, Any]]],
    sample_trajectories: Dict[str, List[float]],
    output_path: str = DEFAULT_OUTPUT
):
    """
    Generate professional 4-panel diagnostic figure using pure Matplotlib:
      (a) Perplexity vs. Distinct-2 Diversity Trade-off
      (b) Token-Level Probability Trajectory (Confidence Dynamics)
      (c) Decoding Latency & Throughput Comparison
      (d) Tri-gram Repetition Profile across Decoding Strategies
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    strategies = list(benchmark_data.keys())
    strat_colors = {
        "greedy": "#d62728",
        "beam": "#ff7f0e",
        "temperature": "#2ca02c",
        "top_k": "#1f77b4",
        "top_p": "#9467bd",
    }

    # -------------------------------------------------------------------------
    # Panel 1: Perplexity vs Distinct-2 Lexical Diversity
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    avg_ppls = [np.mean([r["perplexity"] for r in benchmark_data[s]]) for s in strategies]
    avg_dist2 = [np.mean([r["distinct_2"] for r in benchmark_data[s]]) for s in strategies]

    for s, ppl, dist in zip(strategies, avg_ppls, avg_dist2):
        c = strat_colors.get(s, "#333333")
        ax1.scatter(dist, ppl, s=180, color=c, edgecolor="black", linewidth=1.5, label=s.upper(), zorder=4)
        ax1.annotate(
            s.upper(),
            (dist, ppl),
            textcoords="offset points",
            xytext=(10, 8),
            fontweight="bold",
            fontsize=9
        )

    ax1.set_xlabel("Lexical Diversity (Distinct-2 Ratio)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Sequence Perplexity (PPL)", fontsize=10, fontweight="bold")
    ax1.set_title("Language Model Quality: Perplexity vs. Diversity Trade-off", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 2: Token-Level Probability Trajectory
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    for s, probs in sample_trajectories.items():
        if probs:
            steps = np.arange(1, len(probs) + 1)
            c = strat_colors.get(s, "#333333")
            ax2.plot(steps, probs, label=s.upper(), color=c, linewidth=2.0, marker="o", markersize=4, alpha=0.85)

    ax2.set_xlabel("Generated Token Step ($t$)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Assigned Token Probability $P(x_t | x_{<t})$", fontsize=10, fontweight="bold")
    ax2.set_title("Autoregressive Token Confidence Dynamics across Steps", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 3: Decoding Latency & Throughput
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    avg_latency = [np.mean([r["latency_sec"] for r in benchmark_data[s]]) for s in strategies]
    avg_speed = [np.mean([r["throughput_tok_sec"] for r in benchmark_data[s]]) for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35
    b1 = ax3.bar(x - width/2, avg_latency, width, label="Latency (s)", color="#457b9d", edgecolor="black")
    ax3.set_ylabel("Generation Latency (seconds)", color="#457b9d", fontsize=10, fontweight="bold")
    ax3.tick_params(axis="y", labelcolor="#457b9d")

    ax3_twin = ax3.twinx()
    b2 = ax3_twin.bar(x + width/2, avg_speed, width, label="Throughput (tok/s)", color="#2a9d8f", edgecolor="black")
    ax3_twin.set_ylabel("Decoding Throughput (tokens/s)", color="#2a9d8f", fontsize=10, fontweight="bold")
    ax3_twin.tick_params(axis="y", labelcolor="#2a9d8f")

    ax3.set_xticks(x)
    ax3.set_xticklabels([s.upper() for s in strategies], fontsize=9, fontweight="bold")
    ax3.set_title("Computational Efficiency: Latency & Generation Speed", fontsize=11, fontweight="bold")

    # Combined legend
    bars = [b1, b2]
    labels = ["Latency (s)", "Throughput (tok/s)"]
    ax3.legend(bars, labels, loc="upper left", framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 4: Repetition Degradation Profile
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    avg_rep3 = [np.mean([r["repetition_3"] for r in benchmark_data[s]]) for s in strategies]
    colors = [strat_colors.get(s, "#555") for s in strategies]

    bars4 = ax4.bar([s.upper() for s in strategies], avg_rep3, color=colors, edgecolor="black", alpha=0.85)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax4.set_ylabel("Tri-gram Repetition Rate (%)", fontsize=10, fontweight="bold")
    ax4.set_title("Degeneracy Analysis: Tri-Gram Repetition Percentage", fontsize=11, fontweight="bold")
    ax4.set_ylim(0, max(max(avg_rep3) * 1.35, 15))
    ax4.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.suptitle("PyTorch & GPT-2 (124M): Decoding Strategy Comparative Benchmark", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved diagnostic visualization figure: {output_path}")


# ==============================================================================
# 4. Benchmark & Main Pipeline
# ==============================================================================

def run_benchmark(prompts_path: str = DEFAULT_PROMPTS) -> None:
    """Execute complete multi-strategy benchmark across prompt dataset."""
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    generator = GPT2TextGenerator()

    strategies = ["greedy", "beam", "temperature", "top_k", "top_p"]
    benchmark_results: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}
    sample_trajectories: Dict[str, List[float]] = {}

    print("\n" + "=" * 80)
    print("GPT-2 TEXT GENERATION STRATEGY BENCHMARK (124M Parameters)")
    print("=" * 80)

    for p_idx, item in enumerate(prompts_data):
        prompt = item["prompt"]
        category = item["category"]
        print(f"\n[Prompt {p_idx + 1}/{len(prompts_data)}] ({category})")
        print(f"  Prompt: \"{prompt}\"")

        for s in strategies:
            res = generator.generate(
                prompt,
                strategy=s,
                max_new_tokens=40,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                num_beams=4,
            )
            benchmark_results[s].append(res)

            # Store probability trajectory for the first prompt
            if p_idx == 0:
                sample_trajectories[s] = res["token_probs"]

            first_sentence = res["continuation"].strip().replace("\n", " ")
            if len(first_sentence) > 90:
                first_sentence = first_sentence[:87] + "..."
            print(f"    [{s.upper():<11}] PPL: {res['perplexity']:<6.2f} | Dist-2: {res['distinct_2']:.2f} | Speed: {res['throughput_tok_sec']:.1f} tok/s | Text: \"{first_sentence}\"")

    # Generate Visualization Figure
    generate_visualization(benchmark_results, sample_trajectories)

    # -------------------------------------------------------------------------
    # Aggregate Comparison Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DECODING STRATEGY AGGREGATE SUMMARY (N=6 Prompts)")
    print("=" * 80)
    header = f"{'Strategy':<13} | {'Avg PPL':<8} | {'Dist-1':<7} | {'Dist-2':<7} | {'Rep-3 %':<8} | {'Latency (s)':<11} | {'Speed (tok/s)':<13}"
    print(header)
    print("-" * len(header))

    for s in strategies:
        records = benchmark_results[s]
        avg_ppl = np.mean([r["perplexity"] for r in records])
        avg_d1 = np.mean([r["distinct_1"] for r in records])
        avg_d2 = np.mean([r["distinct_2"] for r in records])
        avg_rep = np.mean([r["repetition_3"] for r in records])
        avg_lat = np.mean([r["latency_sec"] for r in records])
        avg_spd = np.mean([r["throughput_tok_sec"] for r in records])

        print(f"{s.upper():<13} | {avg_ppl:<8.2f} | {avg_d1:<7.2f} | {avg_d2:<7.2f} | {avg_rep:<7.1f}% | {avg_lat:<11.3f} | {avg_spd:<13.1f}")
    print("=" * 80 + "\n")


def interactive_generator():
    """Interactive console prompt for user text generation."""
    generator = GPT2TextGenerator()
    print("\n" + "=" * 80)
    print("Interactive GPT-2 Text Generator (Enter 'q' to quit)")
    print("=" * 80)

    strategies = ["greedy", "beam", "top_p"]

    while True:
        try:
            prompt = input("\nEnter prompt prefix: ").strip()
            if prompt.lower() == "q":
                break
            if not prompt:
                continue

            print("\nGenerating extensions across key strategies...")
            for s in strategies:
                res = generator.generate(prompt, strategy=s, max_new_tokens=40)
                print(f"\n[{s.upper()} - PPL: {res['perplexity']:.2f}, Latency: {res['latency_sec']:.2f}s]")
                print(f"{prompt} \033[92m{res['continuation']}\033[0m")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive session.")
            break


def main():
    parser = argparse.ArgumentParser(description="GPT-2 Text Generation & Decoding Comparison")
    parser.add_argument("--prompts", type=str, default=DEFAULT_PROMPTS, help="Path to prompts JSON")
    parser.add_argument("--prompt", type=str, default="", help="Single prompt prefix to extend")
    parser.add_argument("--strategy", type=str, default="top_p", choices=["greedy", "beam", "temperature", "top_k", "top_p"], help="Decoding strategy")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering value")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus threshold")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt loop")
    args = parser.parse_args()

    if args.interactive:
        interactive_generator()
    elif args.prompt:
        generator = GPT2TextGenerator()
        res = generator.generate(
            args.prompt,
            strategy=args.strategy,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print("\n" + "=" * 80)
        print(f"PROMPT ({args.strategy.upper()} DECODING):")
        print(args.prompt)
        print("\nCONTINUATION:")
        print(res["continuation"])
        print("-" * 80)
        print(f"PPL: {res['perplexity']:.2f} | Distinct-2: {res['distinct_2']:.2f} | Latency: {res['latency_sec']:.2f}s | Speed: {res['throughput_tok_sec']:.1f} tok/s")
        print("=" * 80 + "\n")
    else:
        run_benchmark(args.prompts)


if __name__ == "__main__":
    main()

