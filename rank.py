import pandas as pd
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Load Data
# -------------------------------

print("Loading data...")

rank_df = pd.read_pickle(
    "artifacts/dataset_features_v3.pkl"
)

with open("artifacts/skill_embeddings.pkl","rb") as f:
    skill_embeddings = pickle.load(f)

with open("artifacts/career_embeddings.pkl","rb") as f:
    career_embeddings = pickle.load(f)

with open("artifacts/profile_embeddings.pkl","rb") as f:
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
embeddings retrieval ranking recommendation systems
hybrid search vector databases
sentence-transformers BGE E5 FAISS Pinecone Weaviate Qdrant OpenSearch Elasticsearch
Python evaluation frameworks NDCG MAP MRR
"""

jd_career_text = """
product company experience
production ML systems
retrieval ranking recommendation search systems
evaluation frameworks
A/B testing
NDCG MAP MRR
Python engineering
hands-on coding
shipper mindset
"""

jd_profile_text = """
senior AI engineer
6 to 8 years experience
active candidate
open to relocation
startup mindset
fast moving engineering culture
hands-on coding
scalable systems
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
      0.70 * rank_df["career_similarity"]
    + 0.25 * rank_df["skill_similarity"]
    + 0.05 * rank_df["profile_similarity"]
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
    'AI Research Engineer',
    'AI Specialist',
    'Backend Engineer',
    'Software Engineer',
    'Senior Software Engineer',
    'Data Scientist',
    'Data Engineer',
    'Senior Data Engineer',
    'Analytics Engineer',
    'Cloud Engineer',
    'DevOps Engineer',
    'Machine Learning Engineer',
    'Senior Machine Learning Engineer',
    'Staff Machine Learning Engineer',
    'Applied ML Engineer',
    'AI Engineer',
    'Senior AI Engineer',
    'NLP Engineer'
]

mild_positive = [
    '.NET Developer',
    'Java Developer',
    'Frontend Engineer',
    'Full Stack Developer',
    'Mobile Developer'
]

neutral = [
    'Business Analyst',
    'QA Engineer',
    'Project Manager',
    'Data Analyst'
]

strong_negative = [
    'HR Manager',
    'Marketing Manager',
    'Content Writer',
    'Graphic Designer',
    'Customer Support',
    'Sales Executive',
    'Accountant'
]

mild_negative = [
    'Mechanical Engineer',
    'Civil Engineer',
    'Operations Manager'
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
# Integrity Penalty
# -------------------------------

rank_df['job_switch_penalty'] = (
    rank_df['job_switch_count'] > 8
).astype(int)

rank_df['inactive_penalty'] = (
    rank_df['days_since_last_active'] > 180
).astype(int)

rank_df['integrity_penalty'] = (
    rank_df['job_switch_penalty']
    +
    rank_df['inactive_penalty']
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
] = 2.0

rank_df.loc[
    (rank_df['signal_notice_period_days'] > 30)
    &
    (rank_df['signal_notice_period_days'] <= 90),
    'notice_penalty'
] = 1.0
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

def ir_bonus(row):

    text = (
        str(row['career_text']) + " " +
        str(row['skills_text'])
    ).lower()

    return int(
        any(word in text for word in ir_words)
    )


rank_df['ir_bonus'] = (
    rank_df.apply(
        ir_bonus,
        axis=1
    )
)

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
    + 0.10 * rank_df['title_prior']
    + 0.10 * rank_df['transition_penalty']
    + 0.05 * rank_df['industry_bonus']
    - 0.05 * rank_df['career_company_penalty']
    - 0.05 * rank_df['notice_penalty']
    + 0.05 * rank_df['ir_bonus']
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

    reasons = []

    title = row['profile_current_title']
    yoe = round(row['profile_years_of_experience'], 1)
    industry = row['profile_current_industry']

    reasons.append(
        f"{title} with {yoe} years of experience in {industry}"
    )

    # Skill / career evidence
    kws = extract_keywords(row)

    if len(kws) > 0:
        reasons.append(
            "profile shows exposure to " + ", ".join(kws)
        )

    # Engagement
    if row['signal_recruiter_response_rate'] > 0.8:
        reasons.append(
            "high recruiter responsiveness suggests active engagement"
        )

    # Github activity
    if row['signal_github_activity_score'] > 0.6:
        reasons.append(
            "strong external activity signal"
        )

    # Notice period
    notice_days = int(row['signal_notice_period_days'])

    if notice_days > 90:
        reasons.append(
            f"{notice_days}-day notice period is a concern"
        )

    elif notice_days > 30:
        reasons.append(
            f"{notice_days}-day notice period raises the bar slightly"
        )

    # Rank consistency
    rank = row['rank']

    if rank <= 20:
        reasons.append(
            "overall profile aligns strongly with the role requirements"
        )

    elif rank >= 80:
        reasons.append(
            "overall alignment is weaker than higher-ranked profiles"
        )

    return ". ".join(reasons) + "."
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
    "submission.csv",
    index=False
)

print("submission.csv created successfully")