# Redrob AI Challenge — Candidate Ranking System
**Redrob × Hack2Skill — India Runs Data & AI Challenge**

**Team**: BeyondCV — Abhinav Thakur (Lead), Anurudh Shrestha, Aashika Kumari, Ayushi Choudhary
---

 ## Problem

Recruiters screening 100K+ candidates for a single role spend days on manual filtering — and still miss top talent, because traditional ATS systems match keywords, not careers. This project builds a system that understands what actually makes a candidate qualified: career trajectory, real-world IR/retrieval experience, behavioral signals, and genuine fit — not just whether they listed the right buzzwords.

## What it does

Given candidates.jsonl (up to 100K profiles) and a fixed job description, rank_final.py streams, filters, scores, and ranks candidates end-to-end, producing the top 100 with a JD-grounded, per-candidate reasoning string — in under 3 minutes, on CPU only, with no internet access required.

**100K+ candidates → Pre-filter (~1,500) → Semantic + Multi-signal scoring → Honeypot filter → Top 100 + reasoning**

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run ranker (produces submission.csv)
python rank_final.py candidates.jsonl --out submission.csv
```

**Runtime:** ~3 minutes on CPU for 100K candidates

**Compute:**- no GPU, no internet — the embedding model is bundled locally under models/
Files with ≤200 lines are treated as sandbox/demo input and skip the pre-filter, so small sample files still produce full output.
---

## 🔗 Sandbox Demo

Try the ranker live on sample candidates (no setup needed):

👉 **[Open in Google Colab](https://colab.research.google.com/drive/1iY1G2Vz7iVfTe84aoJxTHPnU_LJibhGm#scrollTo=p1ZGVRE5GrNZ)**

Upload any JSONL or JSON candidate file (≤100 candidates) and get ranked output instantly.

---

## 📁 Repository Structure

├── rank_final.py                    # End-to-end ranking pipeline (single script, single command)
├── requirements.txt                 # Dependencies
├── candidate_schema.json            # JSON Schema for a single candidate profile
├── submission_metadata.yaml         # Team info, compute environment, methodology declaration
├── models/
│   └── all-MiniLM-L6-v2/            # Bundled sentence-transformers model (offline, ~80MB)
├── Sample_data/
│   ├── sample_candidates.json       # Sample candidates for local/sandbox testing
│   └── sample_submission.csv        # Example output format
├── Notebooks/                       # Exploration & development notebooks
│   ├── dataset-study.ipynb
│   ├── dataset_cleaning.ipynb
│   ├── feature_engineering.ipynb
│   ├── feature_engineering_advanced.ipynb
│   ├── feature_engineering_advanced_2.ipynb
│   ├── hybrid_ranking.ipynb
│   └── full_ranking.ipynb
└── README.md

---

## 🧠 Approach

### Stage 1 — Fast Pre-filter
Streams `candidates.jsonl` line-by-line without loading into RAM. Filters to ~1,500–3,000 relevant candidates using an exact title whitelist and IR keyword matching. Reduces encoding workload by ~85%.

**Hard excluded titles** (zero relevance to JD): HR Manager, Marketing Manager, Accountant, Civil Engineer, Operations Manager, and 8 others.

**Strong titles** (pass immediately): ML Engineer, Recommendation Systems Engineer, NLP Engineer, Senior AI Engineer, Applied ML Engineer, and variants.

**Conditional titles** (pass only with IR keyword evidence): Software Engineer, Data Scientist, Backend Engineer — must show retrieval/ranking/vector DB keywords in skills or career history.

### Stage 2 — Semantic Scoring
Three embedding passes combined into a single batched encode call (batch_size=128):
- `skill_similarity` — cosine similarity of skills text vs JD skill requirements
- `career_similarity` — cosine similarity of career history vs JD career profile
- `profile_similarity` — cosine similarity of profile text vs JD profile description

Model: `all-MiniLM-L6-v2` (80MB, CPU-optimized, loaded from local path — no internet required)

```python
semantic_score = (
    0.45 * career_similarity +
    0.45 * skill_similarity  +
    0.10 * profile_similarity
)
```

### Stage 3 — Multi-Signal Scoring

```python
final_score = (
    0.42 * semantic_score_norm        # Core JD fit
  + 0.20 * career_score              # YOE, tenure, degree, job duration
  + 0.16 * behavior_score            # Recruiter response, GitHub, interview rate
  + 0.10 * availability_score        # Open to work, willing to relocate, recency
  + 0.09 * skill_score               # Skill count, duration, endorsements
  + 0.10 * title_prior_norm          # Title relevance to JD
  + 0.06 * location_score            # Location preference signal
  + 0.05 * industry_bonus_norm       # Product vs services company
  + 0.05 * ir_bonus_guarded          # IR/retrieval keyword presence (title-gated)
  - 0.05 * integrity_penalty         # Job hopping, inactivity
  - 0.05 * transition_penalty        # Career-title mismatch
  - 0.05 * career_company_penalty    # 100% services company career
  - 0.06 * notice_penalty_norm       # Notice period (soft signal)
  - 0.03 * profile_consistency_penalty  # YOE vs career mismatch
)
```

### Stage 4 — Honeypot Detection
Candidates with `|stated_YOE − actual_career_months/12| > 3 years` receive a `-0.30` penalty, pushing impossible profiles out of top-100.

### Stage 5 — Reasoning Generation
Per-candidate reasoning strings are generated with:
- Specific facts (company name, YOE, industry)
- JD connection (IR/retrieval keywords, experience band fit)
- Honest gap acknowledgment (notice period, inactivity, services background)
- Rank-consistent tone (positive for top-35, concern language for rank 60+)

---

## ⚙️ Key Design Decisions

**Why `all-MiniLM-L6-v2`?**
Fastest CPU-compatible model (80MB, ~2x faster than BGE-small). Quality difference is negligible for IR keyword similarity tasks. Bundled locally — no internet required in Docker sandbox.

**Why pre-filter before embedding?**
100K candidates × 3 text types = 300K encode calls. Pre-filtering to ~1,500 relevant candidates reduces this to ~4,500 calls, bringing runtime from ~40 min to ~3 min on CPU.

**Why `ir_bonus_guarded`?**
Raw IR keyword bonus rewards anyone who lists "Pinecone" in their skills — including Marketing Managers. Guarded version only grants the bonus if `title_prior > 0 AND career_similarity > 0.3`, ensuring keyword presence is backed by career evidence.

**Why `-0.30` for YOE gap?**
Honeypot candidates in this dataset all share the same signature: stated YOE is 8–11 years higher than actual career history. A deterministic penalty is more reliable than a learned signal for this pattern.

---
## Results 


-**Top 10:** : all ML/AI or Recommendation Systems Engineers from product companies (Salesforce, LinkedIn, Zomato, Netflix, Microsoft, Krutrim, Haptik), YOE range 5.7–8.8 years — inside the JD's 5–9 year target band
**-Honeypots in top 100**: 0 (eliminated by YOE-gap detection)
**-Score spread across top 100**: 0.724 – 0.894
**-Runtime:** : under 3 minutes on CPU for 100K candidates

## 📦 Requirements

```
sentence-transformers==5.5.1
pandas
numpy
scikit-learn
```

Install:
```bash
pip install -r requirements.txt
```

---

## 🔄 Reproduce

```bash
# Full pipeline — reads candidates.jsonl, writes submission.csv
python rank_final.py candidates.jsonl --out submission.csv
```

## Links

- **GitHub:** https://github.com/abhinavt1325/Redrob-candidate-ranking
- **Sandbox Demo (Colab):** https://colab.research.google.com/drive/1iY1G2Vz7iVfTe84aoJxTHPnU_LJibhGm

No GPU required. No internet required. No pre-computation step needed.

---

*BeyondCV | India Runs Data and AI Challenge 2026*
