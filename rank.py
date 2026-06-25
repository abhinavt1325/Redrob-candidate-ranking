import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Load Data
# -------------------------------

print("Loading data...")

artifact_dir = Path(__file__).resolve().parent / "artifacts"

feature_files = [
    "dataset_features_v3.pkl",
    "dataset_features_v2.pkl",
    "dataset_features_v1.pkl",
]

rank_df = None
for filename in feature_files:
    feature_path = artifact_dir / filename
    if not feature_path.exists():
        continue
    try:
        print(f"Trying to load {filename}...")
        rank_df = pd.read_pickle(feature_path)
        print(f"Loaded feature file: {filename}")
        break
    except MemoryError:
        print(f"MemoryError while loading {filename}; trying next fallback.")
    except Exception as exc:
        print(f"Error loading {filename}: {exc}")
        raise

if rank_df is None:
    raise FileNotFoundError(
        f"Could not load any feature artifact from {artifact_dir}. "
        f"Tried: {', '.join(feature_files)}"
    )

with open(artifact_dir / "skill_embeddings.pkl","rb") as f:
    skill_embeddings = pickle.load(f)

with open(artifact_dir / "career_embeddings.pkl","rb") as f:
    career_embeddings = pickle.load(f)

with open(artifact_dir / "profile_embeddings.pkl","rb") as f:
    profile_embeddings = pickle.load(f)

print("Files loaded successfully")

# -------------------------------
# Load Model
# -------------------------------

print("Loading model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model loaded")

# -------------------------------
# Scaler
# -------------------------------

scaler = MinMaxScaler()

# -------------------------------
# JD Texts
# -------------------------------

jd_skill_text = """
Built and deployed production retrieval, ranking, recommendation, and search systems.

Experience with dense retrieval, hybrid retrieval, semantic search, vector search, and candidate matching.

Hands-on use of embeddings, sentence-transformers, BGE, E5, retrieval pipelines, and relevance optimization.

Worked with vector databases and search infrastructure including FAISS, Pinecone, Qdrant, Weaviate, Elasticsearch, and OpenSearch.

Designed ranking evaluation frameworks using NDCG, MAP, MRR, precision, recall, and offline relevance metrics.

Strong Python engineering, experimentation, debugging, and production machine learning practices.
"""

jd_career_text = """
Worked at product-focused technology companies building systems used by real customers.

Built and shipped recommendation systems, ranking systems, retrieval systems, search platforms, or candidate matching products.

Hands-on ownership of production machine learning systems rather than purely research or architecture roles.

Experience designing offline and online evaluation pipelines, running A/B tests, measuring ranking quality, and improving user outcomes.

Strong software engineering mindset with Python development, deployment, monitoring, experimentation, and iteration.

Preference for engineers who have shipped end-to-end systems and demonstrated measurable business impact.
"""

jd_profile_text = """
Senior machine learning engineer.

Recommendation systems engineer.

Retrieval and ranking engineer.

Product-company background.

Hands-on builder with strong software engineering fundamentals.

Experience operating production systems at scale.

Active candidate with strong engagement signals.

Open to relocation and capable of working in fast-moving startup environments.

Ownership mindset, execution focused, and comfortable shipping products.
"""

# -------------------------------
# JD Embeddings
# -------------------------------

print("Creating JD embeddings...")

jd_skill_embedding = model.encode(
    jd_skill_text,
    normalize_embeddings=True
)

jd_career_embedding = model.encode(
    jd_career_text,
    normalize_embeddings=True
)

jd_profile_embedding = model.encode(
    jd_profile_text,
    normalize_embeddings=True
)

print("JD embeddings created")

# -------------------------------
# Similarities
# -------------------------------

print("Computing similarities...")

rank_df["skill_similarity"] = cosine_similarity(
    skill_embeddings,
    jd_skill_embedding.reshape(1, -1)
)

rank_df["career_similarity"] = cosine_similarity(
    career_embeddings,
    jd_career_embedding.reshape(1, -1)
)

rank_df["profile_similarity"] = cosine_similarity(
    profile_embeddings,
    jd_profile_embedding.reshape(1, -1)
)

print("Similarities computed")

# -------------------------------
# Semantic Score
# -------------------------------

rank_df["semantic_score"] = (
      0.60 * rank_df["career_similarity"]
    + 0.30 * rank_df["skill_similarity"]
    + 0.10 * rank_df["profile_similarity"]
)

rank_df["semantic_score_norm"] = scaler.fit_transform(
    rank_df[["semantic_score"]]
)

# -------------------------------
# Career Score
# -------------------------------

career_cols = [
    'profile_years_of_experience',
    'current_company_tenure_months',
    'avg_job_duration_months',
    'highest_degree_score'
]

rank_df['career_score'] = scaler.fit_transform(
    rank_df[career_cols]
    .mean(axis=1)
    .values.reshape(-1,1)
)


# -------------------------------
# Behavior Score
# -------------------------------

behavior_cols = [
    'signal_recruiter_response_rate',
    'signal_interview_completion_rate',
    'signal_github_activity_score',
    'signal_saved_by_recruiters_30d',
    'signal_profile_views_received_30d'
]

# Save raw values before normalisation (used in generate_reason)
rank_df['_raw_recruiter_response_rate'] = rank_df['signal_recruiter_response_rate']
rank_df['_raw_github_activity_score']   = rank_df['signal_github_activity_score']

rank_df[behavior_cols] = scaler.fit_transform(
    rank_df[behavior_cols]
)

rank_df['behavior_score'] = (
    rank_df[behavior_cols]
    .mean(axis=1)
)


# -------------------------------
# Availability Score
# -------------------------------

availability_cols = [
    'signal_open_to_work_flag',
    'signal_willing_to_relocate'
]

rank_df['signal_last_active_date'] = pd.to_datetime(
    rank_df['signal_last_active_date']
)

today = pd.Timestamp("2026-06-23")

rank_df['days_since_last_active'] = (
    today - rank_df['signal_last_active_date']
).dt.days

rank_df['recent_activity_score'] = 1 - MinMaxScaler().fit_transform(
    rank_df[['days_since_last_active']]
)

rank_df['availability_score'] = (
      0.4*rank_df['signal_open_to_work_flag']
    + 0.2*rank_df['signal_willing_to_relocate']
    + 0.4*rank_df['recent_activity_score']
)


# -------------------------------
# Skill Score
# -------------------------------

skill_cols = [
    'num_skills',
    'avg_skill_duration_months',
    'avg_skill_endorsements'
]

rank_df[skill_cols] = scaler.fit_transform(
    rank_df[skill_cols]
)

rank_df['skill_score'] = (
    rank_df[skill_cols]
    .mean(axis=1)
)

# -------------------------------
# Title Priors
# -------------------------------

strong_positive = [
    'Recommendation Systems Engineer',
    'ML Engineer',
    'AI Specialist',
    'Machine Learning Engineer',
    'Senior Machine Learning Engineer',
    'Staff Machine Learning Engineer',
    'Applied ML Engineer',
    'AI Engineer',
    'Senior AI Engineer',
    'NLP Engineer',
    'Backend Engineer',
    'Software Engineer',
    'Senior Software Engineer',
    'Data Scientist',
    'Data Engineer',
    'Senior Data Engineer',
    'Analytics Engineer',
    'Cloud Engineer',
    'DevOps Engineer',
]

mild_positive = [
    'AI Research Engineer',
    '.NET Developer',
    'Java Developer',
    'Frontend Engineer',
    'Full Stack Developer',
    'Mobile Developer',
]

neutral = [
    'Business Analyst',
    'QA Engineer',
    'Project Manager',
    'Data Analyst',
]

strong_negative = [
    'HR Manager',
    'Marketing Manager',
    'Content Writer',
    'Graphic Designer',
    'Customer Support',
    'Sales Executive',
    'Accountant',
]

mild_negative = [
    'Mechanical Engineer',
    'Civil Engineer',
    'Operations Manager',
]

def title_prior(title):

    if title in strong_positive:
        return 1

    elif title in mild_positive:
        return 0.5

    elif title in strong_negative:
        return -1

    elif title in mild_negative:
        return -0.5

    else:
        return 0


rank_df['title_prior'] = (
    rank_df['profile_current_title']
    .apply(title_prior)
)
# -------------------------------
# Transition Penalty
# -------------------------------

non_ai_titles = [
    'HR Manager',
    'Marketing Manager',
    'Content Writer',
    'Graphic Designer',
    'Customer Support',
    'Sales Executive',
    'Accountant'
]

ai_words = [
    'rag',
    'langchain',
    'pinecone',
    'embeddings',
    'vector search',
    'genai',
    'llm',
    'openai'
]


def transition_penalty(row):

    title = str(row['profile_current_title']).lower()
    profile = str(row['profile_text']).lower()

    if (
        any(t.lower() == title for t in non_ai_titles)
        and
        any(word in profile for word in ai_words)
    ):
        return -1

    return 0


rank_df['transition_penalty'] = (
    rank_df.apply(
        transition_penalty,
        axis=1
    )
)
# -------------------------------
# Transition Penalty (Normalized)
# -------------------------------

rank_df['transition_penalty_norm'] = (
    -rank_df['transition_penalty']
)
# -------------------------------
# Integrity Penalty
# -------------------------------

rank_df['job_switch_penalty'] = (
    rank_df['job_switch_count'] > 8
).astype(int)

rank_df['inactive_penalty'] = 0.0
rank_df.loc[rank_df['days_since_last_active'] > 180, 'inactive_penalty'] = 1.0
rank_df.loc[
    (rank_df['days_since_last_active'] > 60) &
    (rank_df['days_since_last_active'] <= 180),
    'inactive_penalty'
] = 0.5

rank_df['integrity_penalty'] = (
    rank_df['job_switch_penalty'] + rank_df['inactive_penalty']
)
rank_df['integrity_penalty'] = (
    MinMaxScaler()
    .fit_transform(
        rank_df[['integrity_penalty']]
    )
)
# -------------------------------
# Industry Bonus
# -------------------------------
product_industries = [
    'Software',
    'Food Delivery',
    'Fintech',
    'E-commerce',
    'EdTech',
    'SaaS',
    'AI/ML',
    'HealthTech',
    'Gaming',
    'Conversational AI',
    'AdTech',
    'HealthTech AI'
]
negative_industries = [
    'IT Services',
    'Manufacturing',
    'Consulting'
]
def industry_bonus(industry):
    if industry in product_industries:
        return 1
    elif industry in negative_industries:
        return -1
    return 0

rank_df['industry_bonus'] = (
    rank_df['profile_current_industry']
    .apply(industry_bonus)
)
# -------------------------------
# Career Company Penalty
# -------------------------------

service_companies = [
    'TCS',
    'Infosys',
    'Wipro',
    'Accenture',
    'Cognizant',
    'Capgemini'
]

def career_company_penalty(career_history):

    companies = [
        str(job.get('company', ''))
        for job in career_history
    ]

    if len(companies) == 0:
        return 0

    count_service = sum(
        any(
            s.lower() in company.lower()
            for s in service_companies
        )
        for company in companies
    )

    if count_service == len(companies):
        return 1

    return 0


rank_df['career_company_penalty'] = (
    rank_df['career_history']
    .apply(career_company_penalty)
)
# -------------------------------
# Notice Penalty
# -------------------------------

rank_df['notice_penalty'] = 0.0

rank_df.loc[
    rank_df['signal_notice_period_days'] > 90,
    'notice_penalty'
] = 3.0

rank_df.loc[
    (rank_df['signal_notice_period_days'] > 60) &
    (rank_df['signal_notice_period_days'] <= 90),
    'notice_penalty'
] = 2.0

rank_df.loc[
    (rank_df['signal_notice_period_days'] > 30) &
    (rank_df['signal_notice_period_days'] <= 60),
    'notice_penalty'
] = 1.0

rank_df['notice_penalty_norm'] = rank_df['notice_penalty'] / 3.0
# -------------------------------
# Location Score
# -------------------------------

preferred_cities = [
    "Pune",
    "Noida"
]

supported_cities = [
    "Hyderabad",
    "Mumbai",
    "Delhi",
    "Delhi NCR",
    "New Delhi",
    "Gurgaon",
    "Gurugram",
    "Bangalore",
    "Bengaluru",
    "Chennai"
]


def location_score(row):

    city = str(row['profile_location'])
    country = str(row['profile_country'])

    relocate = bool(
        row['signal_willing_to_relocate']
    )

    city_lower = city.lower()

    # Preferred cities
    if any(
        c.lower() in city_lower
        for c in preferred_cities
    ):
        return 1.0

    # Explicitly welcomed cities
    if any(
        c.lower() in city_lower
        for c in supported_cities
    ):
        return 0.90

    # Outside India
    if country.lower() != "india":

        if relocate:
            return 0.50

        return 0.10

    # Other Indian city + willing to relocate
    if relocate:
        return 0.80

    # Other Indian city + not relocating
    return 0.65


rank_df['location_score'] = (
    rank_df.apply(
        location_score,
        axis=1
    )
)
# -------------------------------
# IR Bonus
# -------------------------------

ir_words = [
    'recommendation',
    'retrieval',
    'ranking',
    'semantic search',
    'information retrieval',
    'vector search',
    'faiss',
    'bm25',
    'sentence transformers',
    'pinecone',
    'qdrant',
    'weaviate',
    'elasticsearch',
    'opensearch'
]


def ir_score(row):

    text = (
        str(row['career_text']) + " " +
        str(row['skills_text'])
    ).lower()

    count = sum(
        1
        for word in ir_words
        if word in text
    )

    return min(
        count / 5,
        1.0
    )


rank_df['ir_score'] = (
    rank_df.apply(
        ir_score,
        axis=1
    )
)
rank_df['ir_bonus_guarded'] = (
    rank_df['ir_score']
    *
    (
        (rank_df['title_prior'] > 0)
        &
        (rank_df['career_similarity'] > 0.30)
    ).astype(float)
)

# -------------------------------
# Normalise
# -------------------------------
rank_df['title_prior_norm'] = (
    rank_df['title_prior'] + 1
) / 2

rank_df['industry_bonus_norm'] = (
    rank_df['industry_bonus'] + 1
) / 2

# -------------------------------
# Profile Consistency Penalty
# -------------------------------

rank_df['profile_consistency_penalty'] = 0

# Too many skills with low average usage duration
rank_df.loc[
    (rank_df['num_skills'] > 15)
    &
    (rank_df['avg_skill_duration_months'] < 6),
    'profile_consistency_penalty'
] += 1

# Experience mismatch between profile and career history
rank_df.loc[
    abs(
        rank_df['profile_years_of_experience']
        -
        rank_df['total_career_months']/12
    ) > 2,
    'profile_consistency_penalty'
] += 1

# Extremely high endorsements with low experience
rank_df.loc[
    (rank_df['avg_skill_endorsements'] > 80)
    &
    (rank_df['profile_years_of_experience'] < 3),
    'profile_consistency_penalty'
] += 1

rank_df['profile_consistency_penalty'] = (
    MinMaxScaler()
    .fit_transform(
        rank_df[['profile_consistency_penalty']]
    )
)
rank_df['final_score'] = (
      0.40 * rank_df['semantic_score_norm']
    + 0.20 * rank_df['career_score']
    + 0.15 * rank_df['behavior_score']
    + 0.10 * rank_df['availability_score']
    + 0.10 * rank_df['skill_score']
    - 0.05 * rank_df['integrity_penalty']
    + 0.10 * rank_df['title_prior_norm']
    - 0.05 * rank_df['transition_penalty_norm']
    + 0.05 * rank_df['industry_bonus_norm']
    - 0.05 * rank_df['career_company_penalty']
    - 0.06 * rank_df['notice_penalty_norm']
    + 0.07 * rank_df["location_score"]
    + 0.05 * rank_df['ir_bonus_guarded']
    - 0.03 * rank_df['profile_consistency_penalty']
)

def extract_keywords(row):

    text = (
        str(row['skills_text']) + " " +
        str(row['career_text'])
    ).lower()

    keyword_map = {
        'recommendation systems': 'Recommendation Systems',
        'semantic search': 'Semantic Search',
        'information retrieval': 'Information Retrieval',
        'vector search': 'Vector Search',
        'faiss': 'FAISS',
        'pinecone': 'Pinecone',
        'qdrant': 'Qdrant',
        'weaviate': 'Weaviate',
        'elasticsearch': 'Elasticsearch',
        'bm25': 'BM25',
        'sentence transformers': 'Sentence Transformers',
        'llm': 'LLMs',
        'fine-tuning': 'Fine-tuning',
        'mlops': 'MLOps',
        'langchain': 'LangChain',
        'rag': 'RAG',
        'nlp': 'NLP'
    }

    found = []

    for k, v in keyword_map.items():
        if k in text:
            found.append(v)

    return found[:3]


def generate_reason(row):

    parts = []

    title = str(row['profile_current_title'])
    yoe = round(row['profile_years_of_experience'], 1)
    industry = str(row['profile_current_industry'])

    rank = int(row['rank'])

    # -----------------------------
    # Opening (varied by rank)
    # -----------------------------
    if rank <= 10:

        parts.append(
            f"{title} with {yoe} years of experience appears strongly aligned with the role"
        )

    elif rank <= 30:

        parts.append(
            f"{title} bringing {yoe} years of experience in {industry}"
        )

    else:

        parts.append(
            f"{title} with {yoe} years of professional experience"
        )
    # -----------------------------
    # Current company
    # -----------------------------

    if rank <= 30:

        current_company = str(
            row.get("profile_current_company", "")
        )

        if current_company:

            parts.append(
                f"currently at {current_company}"
            )

    # -----------------------------
    # Retrieval / ranking evidence
    # -----------------------------
    kws = extract_keywords(row)

    retrieval_terms = {
        'Recommendation Systems',
        'Semantic Search',
        'Information Retrieval',
        'Vector Search',
        'FAISS',
        'Pinecone',
        'Qdrant',
        'Weaviate',
        'Elasticsearch',
        'BM25'
    }

    retrieval_hits = [
        k for k in kws
        if k in retrieval_terms
    ]

    if len(retrieval_hits) >= 2:

        parts.append(
            "demonstrates retrieval and ranking experience through "
            + ", ".join(retrieval_hits)
        )

    elif len(retrieval_hits) == 1:

        parts.append(
            "shows some exposure to retrieval systems via "
            + retrieval_hits[0]
        )

    elif len(kws) > 0:

        parts.append(
            "profile reflects adjacent AI expertise through "
            + ", ".join(kws)
        )

    # -----------------------------
    # Product company signal
    # IMPORTANT:
    # use ORIGINAL columns
    # -----------------------------
    if row['industry_bonus'] == 1:

        parts.append(
            "background aligns with product-oriented engineering environments"
        )

    elif row['industry_bonus'] == -1:

        parts.append(
            "industry background is less aligned with the target product-company profile"
        )

    # -----------------------------
    # Service-company concentration
    # IMPORTANT:
    # use ORIGINAL column
    # -----------------------------
    if row['career_company_penalty'] == 1:

        parts.append(
            "career history is concentrated in service organizations rather than product teams"
        )

    # -----------------------------
    # Experience band
    # -----------------------------
    if 6 <= yoe <= 9:
        parts.append("experience level falls within the JD's preferred 6-9 year range")
    elif 5 <= yoe < 6:
        parts.append("experience level is at the lower end of the JD's 5-9 year range")
    elif 4 <= yoe < 5:
        parts.append(
            "experience level of {:.1f} years is below the JD's stated 5-year minimum; "
            "exception warranted only if technical depth is strong".format(yoe)
        )
    elif yoe > 12:
        parts.append("experience level is significantly above the typical target range")
    elif yoe < 4:
        parts.append("experience level is well below the JD's usual target range")

    # -----------------------------
    # Recruiter responsiveness
    # -----------------------------
    response_rate = float(row['_raw_recruiter_response_rate'])

    if response_rate > 0.80:
        parts.append("high recruiter responsiveness indicates active engagement")
    elif response_rate < 0.30:
        parts.append("low recruiter responsiveness may indicate limited availability")

    # -----------------------------
    # Github activity
    # -----------------------------
    github_score = float(row['_raw_github_activity_score'])

    if github_score > 75:
        parts.append("strong GitHub activity suggests continued hands-on engineering work")
    elif github_score > 50:
        parts.append("moderate GitHub activity supports ongoing technical involvement")

    # -----------------------------
    # Notice period
    # Mention ONLY when relevant
    # -----------------------------
    days_inactive = int(row['days_since_last_active'])

    if days_inactive > 180:
        parts.append(
            "candidate has been inactive for more than six months — "
            "actual availability is uncertain"
        )
    elif days_inactive > 90:
        parts.append(
            f"last active {days_inactive} days ago; engagement level is lower than ideal"
        )
    elif days_inactive > 60:
        parts.append(
            f"last active {days_inactive} days ago; moderate engagement gap"
        )
    # ----------------------------------------------------------
    # Location note
    #---------------------------------------------------------
    city = str(row['profile_location'])
    country = str(row['profile_country'])
    city_lower = city.lower()

    if any(c.lower() in city_lower for c in preferred_cities):
        parts.append(f"located in preferred hiring city ({city})")

    elif any(c.lower() in city_lower for c in supported_cities):
        parts.append(f"located in a supported hiring location ({city})")

    elif country.lower() != "india":
        if row['signal_willing_to_relocate']:
            parts.append(
                f"currently based in {city}, {country} and open to relocation "
                f"— visa sponsorship is case-by-case per the JD"
            )
        else:
            parts.append(
                f"currently based in {city}, {country} and not willing to relocate "
                f"— JD does not sponsor work visas; this is a significant availability risk"
            )

    else:
        # Indian city, not in preferred/supported list
        if not row['signal_willing_to_relocate']:
            parts.append(
                f"based in {city} (outside preferred hiring cities) "
                f"and not willing to relocate — logistics risk for Pune/Noida role"
            )
        else:
            parts.append(f"based in {city}, willing to relocate to Pune/Noida")

    # -----------------------------
    # Explain lower ranking
    # -----------------------------

    if rank >= 80:

        gap_reasons = []

        if row['days_since_last_active'] > 120:
            gap_reasons.append(
                "limited recent activity"
            )

        if row['signal_recruiter_response_rate'] < 0.40:
            gap_reasons.append(
                "below-average recruiter responsiveness"
            )

        if int(row['signal_notice_period_days']) > 60:
            gap_reasons.append(
                f"{int(row['signal_notice_period_days'])}-day notice period"
            )

        if row['career_company_penalty'] == 1:
            gap_reasons.append(
                "career history concentrated in service organizations"
            )

        if row['profile_consistency_penalty'] > 0.30:
            gap_reasons.append(
                "profile consistency concerns"
            )

        # Fallback if no specific issue triggered
        if not gap_reasons:
            gap_reasons.append(
                "lower overall alignment with the retrieval, ranking, and production ML requirements"
            )

        parts.append(
            "lower ranking primarily reflects: "
            + ", ".join(gap_reasons)
        )
    

    # -----------------------------
    # Final conclusion
    # -----------------------------
    if rank <= 10:

        parts.append(
            "overall profile is a strong match for retrieval, ranking, and production ML responsibilities"
        )

    elif rank <= 30:

        parts.append(
            "overall profile aligns well with the role and should be considered a strong shortlist candidate"
        )

    elif rank <= 60:

        parts.append(
            "profile shows partial alignment but trails stronger candidates on key requirements"
        )

    else:

        parts.append(
            "profile contains several gaps relative to higher-ranked candidates"
        )

    return ". ".join(parts) + "."

top100 = (
    rank_df
    .sort_values(
        'final_score',
        ascending=False
    )
    .head(100)
    .copy()
)

top100['rank'] = range(1,101)

top100['reasoning'] = (
    top100.apply(
        generate_reason,
        axis=1
    )
)

top100['score'] = (
    top100['final_score']
    .round(6)
)

submission = top100[
    [
        'candidate_id',
        'rank',
        'score',
        'reasoning'
    ]
]

submission.to_csv(
    "submission_test6.csv",
    index=False
)

print("submission_test6.csv created successfully")