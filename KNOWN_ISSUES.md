## Gap list not fully exhaustive (match_llm.py)
On jobs with many missing requirements, the model sometimes lists only the
single most critical gap (e.g. "Arabic") instead of all of them (FastAPI,
JWT, etc). The SCORE is still correct and appropriately capped — this only
affects the human-readable gap explanation, not ranking. Low priority.
Revisit only if the gap explanations become important later.
