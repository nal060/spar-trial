# SPAR Trial Project: Comparing Data Attribution Methods for Undesired Behaviors Surfaced via Model Diffing

## Overview for AI Assistant

This document provides complete context for a research project. If you are Claude Code or another AI assistant helping with this project, read this entire document before making any code changes or suggestions.

---

## 1. Research Goal

**Primary Question:** Given an undesired behavior surfaced by Logit Diff Amplification (LDA), which data attribution method most effectively identifies the training examples responsible — and does removing those examples actually reduce the behavior?

**Why This Matters:**
- Language models sometimes exhibit harmful behaviors that only appear rarely
- LDA (from Goodfire Research) can surface these rare behaviors, but that's a known technique
- The open question is: once we find a bad behavior, how do we trace it back to training data?
- Multiple attribution methods exist (gradient-based, representation-based, LLM-based), but there's no established comparison using a ground-truth validation signal
- We validate by actually removing flagged data and retraining — the "gold standard" test the mentor specifically asked for

**What Makes This Novel:**
- Head-to-head comparison of three attribution methods on the same surfaced behaviors
- Validation via LoRA fine-tuning on filtered data (closing the loop from detection to remediation)
- The baseline control (fine-tuning without any removal) isolates the effect of attribution-guided removal from the effect of additional fine-tuning

---

## 2. Technical Background

### 2.1 Logit Diff Amplification (LDA)

LDA is a technique from Goodfire Research (August 2025) for surfacing rare behaviors.

**The Formula:**
```
logits_amplified = logits_after + α * (logits_after - logits_before)
```

Where α controls amplification strength. Higher α surfaces rarer behaviors.

### 2.2 Data Attribution Methods

We compare three methods:

**Method 1: Gradient Similarity**
- Compute gradient of LM loss for the harmful output
- Compute gradient of LM loss for each candidate training document
- Rank by cosine similarity of gradient vectors
- Intuition: if training on doc X would update weights similarly to behavior Y, X likely contributed to Y
- We restrict to middle layers (8-12) for memory efficiency

**Method 2: Activation Clustering**
- Run harmful output and candidate docs through the model
- Extract mean-pooled hidden states from a target layer
- Rank candidates by cosine similarity in activation space
- Also cluster candidates (KMeans) to understand thematic grouping
- Intuition: docs that produce similar internal representations likely teach similar behaviors

**Method 3: LLM Judge (Claude API)**
- Present each candidate training doc alongside the harmful output to Claude
- Ask Claude to rate (0-10) how likely the training doc caused the behavior
- Captures semantic/causal relationships that purely numerical methods might miss
- Different paradigm: uses world knowledge rather than model internals

**Shared Pre-Filter: BM25**
- All methods operate on a BM25-filtered candidate set (~100 docs from 50k)
- This is a computational necessity, not a standalone attribution method

### 2.3 Validation: LoRA Patching

For each attribution method:
1. Take its top-K flagged training documents
2. Fine-tune the post-RLVR model (via LoRA) on training data **excluding** those documents
3. Re-run LDA on the patched model
4. Measure whether the harmful behavior rate decreases

A **baseline** control fine-tunes on all data (no removal), isolating the effect of removal from the effect of additional training.

### 2.4 The Models

**OLMo 2 1B** from Allen Institute for AI (fully open weights, data, and code).

**Checkpoints:**
1. `allenai/OLMo-2-0425-1B-SFT` — After supervised fine-tuning, BEFORE RLVR
2. `allenai/OLMo-2-0425-1B-Instruct` — After RLVR

**Training Data:** Tulu 3 dataset (`allenai/tulu-3-sft-mixture`)

---

## 3. Project Structure

```
spar/
├── PROJECT_CONTEXT.md                 # This file
├── lda_attribution_notebook.ipynb     # Main Colab notebook (5 phases)
├── behavior_rates.png                 # Phase 2: harmful rate vs alpha
├── patching_comparison.png            # Phase 4: per-prompt patching results
├── final_results.png                  # Phase 5: comprehensive 4-panel figure
├── phase2_results.json                # Phase 2: all samples + classifications
├── final_results.json                 # All results in one file
├── patching_comparison.csv            # Phase 4: comparison table
└── writeup/
    ├── slides.md                      # Presentation slides
    └── report.md                      # Written report
```

---

## 4. Notebook Walkthrough

### Phase 1: Setup and LDA Implementation (Cells 0-13)
- Install deps (torch, transformers, peft, scikit-learn, anthropic, etc.)
- Load two OLMo 2 1B checkpoints
- Implement LDA sampling function
- Scan 8 test prompts at α = {0, 0.5, 1.0, 2.0} to find interesting behaviors

### Phase 2: Systematic Behavior Surfacing (Cells 14-26)
- Select 3+ focus prompts based on Phase 1 scan
- Generate 50 samples per (prompt, α) combination
- Classify each output using **Claude API** (with keyword fallback)
- Compute harmful rate at each α
- Plot behavior_rates.png

### Phase 3: Data Attribution — Three Methods (Cells 27-41)
- Select harmful outputs from high-α generations
- Load 50k training docs from Tulu 3
- BM25 pre-filter: 100 candidates per harmful output
- **Method 1 (Gradient Similarity):** backward pass per candidate, cosine sim of loss gradients
- **Method 2 (Activation Clustering):** forward pass per candidate, cosine sim in hidden-state space + KMeans clustering
- **Method 3 (LLM Judge):** Claude rates each candidate 0-10 for causal plausibility
- Compare methods: top-K overlap, Spearman rank correlation, Jaccard similarity

### Phase 4: Patching — Validate Attribution (Cells 42-53)
- Prepare filtered training sets (one per method + unfiltered baseline)
- LoRA fine-tune post-RLVR model on each filtered set (200 steps, r=8)
- Re-run LDA at high α on each patched model
- Compare: which method's removals most reduce harmful rate?
- Visualize with bar charts per prompt

### Phase 5: Analysis and Write-up (Cells 54-63)
- Summary statistics
- 4-panel comprehensive figure (behavior rates, method agreement heatmap, patching comparison, summary)
- Key findings template
- Export all results

---

## 5. Important Implementation Notes

### 5.1 Memory Management (Colab T4, 16GB VRAM)
- Two 1B models in float16 ≈ 4GB
- LoRA adapters add minimal overhead
- Use `torch.no_grad()` for generation
- Only enable gradients during gradient attribution
- `deepcopy(model_post)` for each LoRA adapter — watch memory
- Use `torch.cuda.empty_cache()` and `gc.collect()` between adapters

### 5.2 API Key
- The notebook assumes `ANTHROPIC_API_KEY` is set in the environment
- Used for harm classification (Phase 2) and LLM judge attribution (Phase 3)
- Falls back to keyword heuristics if API calls fail

### 5.3 Key Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| N_SAMPLES | 50 | Per (prompt, α). Detects >2% rate behaviors |
| N_BM25_CANDIDATES | 100 | Shared candidate pool for all methods |
| TOP_K | 20 | Docs flagged per method for removal |
| FINETUNE_SIZE | 2000 | Training set size for LoRA patching |
| FINETUNE_STEPS | 200 | Short but enough for directional signal |
| LoRA r | 8 | Low-rank, memory-efficient |
| ACTIVATION_LAYER | 10 | Middle layer for activation extraction |

---

## 6. Expected Results

### What Success Looks Like:
1. LDA surfaces rare behaviors (harmful rate increases with α)
2. The three methods produce meaningfully different rankings (not identical)
3. At least one method's removals reduce harmful rate more than baseline
4. The comparison produces a clear table/plot showing relative effectiveness

### What Partial Success Looks Like:
- LDA works but all methods perform similarly (no differentiation)
- Patching shows directional effects but not statistically significant
- Some prompts show clear patterns, others don't

### What Failure Looks Like:
- LDA doesn't surface harmful behaviors at any α
- Patching has no effect regardless of method
- All methods flag the same documents (no comparison to make)

---

## 7. Evaluation Criteria (SPAR Trial)

1. **Empirical results:** Plots, tables, concrete findings
2. **Research taste:** Did you ask an interesting question, not just replicate?
3. **Technical competence:** Does the code work? Is methodology sound?
4. **Intellectual honesty:** Are limitations acknowledged?
5. **Communication:** Is the write-up clear?

---

## 8. Key Deliverables

1. **behavior_rates.png** — Shows LDA surfaces rare behaviors
2. **patching_comparison.png** — The main result: which method works best
3. **final_results.png** — Comprehensive 4-panel figure for slides
4. **Comparison table** — Method agreement (Jaccard/Spearman) + patching effectiveness
5. **Case study** — One traced example: harmful output → attributed docs → patching effect
6. **Write-up** — Report with all findings and limitations

---

## 9. References

1. Goodfire LDA: https://www.goodfire.ai/research/model-diff-amplification
2. OLMo 2: https://allenai.org/blog/olmo2
3. Tulu 3 Dataset: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
4. Influence Functions Survey: https://arxiv.org/abs/2410.01285

---

*This document should be read in full by any AI assistant helping with this project.*
