# SPAR Trial: LDA + Data Attribution Project

## Quick Start

### Option 1: Google Colab (Recommended)
1. Upload `lda_attribution_notebook.ipynb` to Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run cells sequentially

### Option 2: Local with GPU
```bash
pip install torch transformers datasets accelerate rank_bm25 matplotlib pandas tqdm
jupyter notebook lda_attribution_notebook.ipynb
```

## Files

| File | Purpose |
|------|---------|
| `PROJECT_CONTEXT.md` | **Read this first.** Complete project context for you or AI assistants |
| `lda_attribution_notebook.ipynb` | Main Colab notebook with all code |
| `README.md` | This file |

## For Claude Code / AI Assistants

If you're an AI assistant helping with this project:

1. **Read `PROJECT_CONTEXT.md` completely** before making any changes
2. The notebook is self-contained and should run top-to-bottom
3. Key decision points are marked with comments like "Edit this based on your observations"
4. If something breaks, check:
   - GPU availability (`torch.cuda.is_available()`)
   - Model loading (may need `trust_remote_code=True`)
   - Memory issues (reduce `N_SAMPLES` or `N_TRAINING_DOCS`)

## Expected Timeline

| Phase | Hours | Output |
|-------|-------|--------|
| Setup & LDA | 3-4 | Working LDA, initial behavior scan |
| Systematic Surfacing | 4-5 | Behavior rate plots |
| Data Attribution | 4-5 | Attribution table |
| Write-up | 3-4 | Final report |

## Key Parameters to Adjust

```python
# In the notebook:
N_SAMPLES = 50          # Samples per (prompt, alpha) - increase for reliability
N_TRAINING_DOCS = 50000 # Training docs to load - decrease if memory issues
N_BM25_CANDIDATES = 30  # Candidates from BM25
N_GRADIENT_CANDIDATES = 10  # Candidates for gradient attribution
```

## Troubleshooting

**Out of Memory:**
- Reduce `N_SAMPLES` to 20
- Reduce `N_TRAINING_DOCS` to 20000
- Use Pythia-410M instead of OLMo 2 1B

**Model won't load:**
- Add `trust_remote_code=True` to `from_pretrained()`
- Try specific revision: `revision="main"`

**Slow generation:**
- Reduce `max_tokens` in generation functions
- Use smaller alpha range: `[0.0, 1.0, 2.0]`

## Deadline

January 30, 2025 - Submit write-up with results via Slack
