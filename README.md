# 🏆 BeyondCV: 2-Stage High-Throughput CPU Candidate Ranker
### **Track:** India Runs Data and AI Challenge 2026 | **Team:** BeyondCV

---

[![Track: Data & AI Challenge](https://img.shields.io/badge/Challenge-India%20Runs%20Data%20%26%20AI%20Challenge%202026-blue.svg?style=for-the-badge)]()
[![Model: Sentence-Transformers](https://img.shields.io/badge/Model-all--MiniLM--L6--v2%20%28Offline%29-orange.svg?style=for-the-badge)]()
[![Throughput: 100K in <3 min](https://img.shields.io/badge/Throughput-100K%20candidates%20%2F%20%3C3%20min-brightgreen.svg?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 🌟 Executive Summary

**BeyondCV** is a production-grade, cost-efficient, two-stage candidate ranking system engineered specifically for high-throughput talent acquisition challenges. Operating entirely on standard consumer CPU architectures, it streams, pre-filters, embeds, and scores **100,000+ candidates in under 3 minutes** with zero external API dependencies or running costs.

By coupling semantic search embeddings (powered by a local, CPU-optimized `all-MiniLM-L6-v2` model) with a robust 13-signal heuristic scoring engine, BeyondCV matches candidates to Job Descriptions (JDs) with high precision, actively guards against profile fraud (via custom anti-honeypot detection), and constructs deterministic, hallucination-free reasoning explaining every single ranking decision.

---

## 🔗 Try the Sandbox Demo (Google Colab)

Test the ranker live without any local environment setup:

👉 **[Launch Interactive Google Colab Sandbox](https://colab.research.google.com/drive/1iY1G2Vz7iVfTe84aoJxTHPnU_LJibhGm#scrollTo=p1ZGVRE5GrNZ)**

> [!NOTE]
> Simply upload any JSONL candidate file (or use the provided 50 sample candidates) to instantly experience the ranking pipeline and view the generated matching justifications.

---

## 📁 Repository Structure

```markdown
├── rank_final.py                   # Primary ranker script (streams, embeds, scores, and ranks)
├── candidate_schema.json           # JSON validation schema for candidate profiles
├── requirements.txt                # Exact python package versions
├── submission_metadata.yaml        # Team metadata and execution configurations
├── Notebooks/                      # Development and research notebooks
│   ├── dataset-study.ipynb         # EDA and data exploration
│   ├── dataset_cleaning.ipynb      # Raw data pre-processing
│   ├── feature_engineering.ipynb   # Baseline feature generation
│   ├── hybrid_ranking.ipynb        # Semantic-heuristic hybrid testing
│   └── full_ranking.ipynb          # End-to-end ranker tuning
├── Sample_data/                    # Mock datasets for quick validation
│   ├── sample_candidates.json      # 50 sample candidates for demo runs
│   └── sample_submission.csv       # Sample output format
└── models/
    └── all-MiniLM-L6-v2/           # Bundled Sentence-Transformers model for offline inference
```

---
## Pipeline

### Stage 1 — Fast Pre-filter (streamed, ~30s)
Reads `candidates.jsonl` line-by-line (never loads all 100K into memory) and keeps only plausibly relevant candidates:

- **Hard-excluded titles** — zero relevance to the JD (HR Manager, Marketing Manager, Content Writer, Accountant, Civil/Mechanical Engineer, etc.) are rejected instantly.
- **Strong titles** — ML Engineer, AI Engineer, NLP Engineer, Recommendation Systems Engineer, Search Engineer, Senior Applied Scientist, etc. — pass automatically.
- **Conditional titles** — Software Engineer, Data Scientist, Backend Engineer, AI Specialist/Research Engineer — pass only if their skills or career history contain an IR/retrieval keyword (FAISS, Pinecone, Qdrant, Weaviate, Elasticsearch, embeddings, NDCG, recsys, etc.).

This typically cuts 100K candidates down to ~1,500, reducing embedding workload by ~98%.

### Stage 2 — Feature Engineering
For every surviving candidate, the script derives:
- **Basic counts** — skills, jobs, certifications, languages, degrees
- **Career features** — current company tenure, total career months, job-switch count, average job duration
- **Skill features** — average skill duration & endorsements
- **Education features** — highest degree score (PhD > M.Tech/M.E./M.S. > B.Tech/B.E. > B.Sc)
- **Text fields** — `skills_text`, `career_text`, `education_text`, `profile_text` for semantic encoding

### Stage 3 — Semantic Encoding & Similarity
All candidate texts (skills, career, profile) plus three hand-written JD reference texts are encoded in a **single batched pass** with `all-MiniLM-L6-v2` (`batch_size=128`, CPU-optimized, loaded from a local path). Cosine similarity against the JD embeddings produces:

```python
semantic_score = 0.60 * career_similarity + 0.30 * skill_similarity + 0.10 * profile_similarity
```

### Stage 4 — Multi-Signal Weighted Scoring
Thirteen signals are normalized (MinMax) and combined into a single `final_score`:

```python
final_score = (
      0.42 * semantic_score_norm        # Core JD fit (career + skill + profile similarity)
    + 0.20 * career_score               # YOE, tenure, avg job duration, degree
    + 0.15 * behavior_score             # Recruiter response, interview rate, GitHub, recruiter saves, profile views
    + 0.10 * availability_score         # Open to work, willing to relocate, recent activity
    + 0.10 * skill_score                # Skill count, avg duration, endorsements
    + 0.10 * title_prior_norm           # Title relevance to the JD
    + 0.06 * location_score             # Preferred / supported hiring cities
    + 0.05 * industry_bonus_norm        # Product company vs. services/manufacturing
    + 0.05 * ir_bonus_guarded           # IR/retrieval keyword bonus (gated, see below)
    - 0.05 * integrity_penalty          # Job-hopping (>8 switches), inactivity
    - 0.05 * transition_penalty         # Non-technical title dressed up with AI buzzwords
    - 0.05 * career_company_penalty     # 100% career spent at pure-services companies
    - 0.05 * notice_penalty_norm        # Long notice period
    - 0.03 * profile_consistency_penalty  # Internal profile inconsistencies
)
```

**Why `ir_bonus_guarded`?** A raw IR-keyword bonus rewards anyone who lists "Pinecone" in their skills — including candidates with no real IR background. The guarded version only applies the bonus when `title_prior > 0 AND career_similarity > 0.30`, so keyword presence must be backed by an actually relevant title and career history.

### Stage 5 — Honeypot / Fraud Detection
Candidates whose stated years of experience diverge from their actual summed career-history months by more than 3 years receive a flat **−0.30** penalty on `final_score` — pushing statistically impossible profiles out of the top 100. This deterministic rule was chosen because every honeypot profile found during dataset exploration shared this exact signature.

### Stage 6 — Ranking & Reasoning
The top 100 candidates by `final_score` are selected and given sequential ranks. For each, `generate_reason()` builds a per-candidate explanation entirely from structured data fields:

- Current title, YOE, and company
- Specific IR/retrieval keywords found in career or skills text (not just the skills section)
- Product- vs. services-company signal
- Experience-band fit relative to the JD's target range
- Recruiter responsiveness and GitHub activity
- Inactivity / notice-period flags, where relevant
- Location fit and relocation willingness
- A rank-consistent closing statement (positive framing for top ranks, honest gap explanation for lower ranks)

Because reasoning is assembled from data fields rather than generated by an LLM, every claim maps directly to a value in the input record — there is no hallucination risk.

**Output (`submission.csv`):** `candidate_id, rank, score, reasoning`

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **`all-MiniLM-L6-v2`** | CPU-fast (~80MB), bundled locally so the pipeline runs fully offline — no network calls during ranking. |
| **Pre-filter before embedding** | 100K × 3 text fields would mean ~300K encode calls. Filtering to ~1,500 relevant candidates first brings this down to ~4,500 calls, cutting runtime from tens of minutes to under 3. |
| **Streaming JSONL read** | Candidates are read and filtered line-by-line so the full 100K-record file is never held in memory at once. |
| **Single-pass batched encoding** | Skill, career, and profile texts for all candidates are concatenated and encoded together in one `model.encode()` call, avoiding redundant forward passes. |
| **Gated IR bonus** | Prevents keyword-stuffed but irrelevant profiles from gaming the retrieval/ranking bonus. |
| **Deterministic honeypot rule** | A fixed YOE-gap threshold proved more reliable than a learned signal for catching the fraud pattern present in this dataset. |
| **Data-driven reasoning generator** | No LLM in the loop for explanations — every sentence is traceable to a specific field, eliminating hallucination risk in the audit-facing output. |

---

## Technologies Used

- **sentence-transformers** (`all-MiniLM-L6-v2`) — CPU-optimized semantic embeddings, bundled offline
- **scikit-learn** — `MinMaxScaler`, `cosine_similarity` for normalization and semantic matching
- **pandas / numpy** — vectorized feature engineering across the candidate pool
- **argparse** — single-command CLI for reproducibility
- **Google Colab** — zero-setup sandbox for evaluators

---

## Results (reference run)

- **Top 10:** all ML/AI or Recommendation Systems Engineers from product companies (Salesforce, LinkedIn, Zomato, Netflix, Microsoft, Krutrim, Haptik), YOE range 5.7–8.8 years — inside the JD's 5–9 year target band
- **Honeypots in top 100:** 0 (eliminated by YOE-gap detection)
- **Score spread across top 100:** 0.724 – 0.894
- **Runtime:** under 3 minutes on CPU for 100K candidates

---

## Requirements

```
pandas
numpy
scikit-learn
sentence-transformers
torch
tqdm
joblib
matplotlib
seaborn
jupyter
```

```bash
pip install -r requirements.txt
```

---

## Reproduce

```bash
python rank_final.py candidates.jsonl --out submission.csv
```

No GPU required. No internet required. No pre-computation step needed — this single command runs the full pipeline from raw input to final ranked CSV.

---

## Links

- **GitHub:** https://github.com/abhinavt1325/Redrob-candidate-ranking
- **Sandbox Demo (Colab):** https://colab.research.google.com/drive/1iY1G2Vz7iVfTe84aoJxTHPnU_LJibhGm

---

*BeyondCV | India Runs Data and AI Challenge — Redrob × Hack2Skill*


| Name | Role | Email |
| :--- | :--- | :--- |
| **Abhinav Thakur** | Team Lead & ML Engineer | abhinavt0613@gmail.com |
| **Anurudh Shrestha** | ML Engineer | anirudhshrestha28@gmail.com |
| **Aashika Kumari** | Data Engineer | aashikapandey10@gmail.com |
| **Ayushi Choudhary** | Data Engineer | choudharybinaykumar0@gmail.com |

---
*Developed by Team BeyondCV for the India Runs Data and AI Challenge 2026.*
