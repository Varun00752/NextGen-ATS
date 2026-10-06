import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import re
import hashlib
import io
import os
import shutil
from datetime import datetime, date, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import plotly.express as px
import plotly.graph_objects as go

# Support both PyPDF2 and pypdf
try:
    from PyPDF2 import PdfReader
except ImportError:
    from pypdf import PdfReader

try:
    from docx import Document
except ImportError:
    Document = None

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="NextGen ATS - AI Resume Screening & Hiring Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_NAME = "ats_enterprise.db"
UPLOAD_DIR = "uploads"
EXPORT_DIR = "exports"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# =========================================================
# UTILITIES & HELPERS
# =========================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password, stored_hash):
    return hash_password(plain_password) == stored_hash

def safe_float(v, default=0.0):
    try:
        return float(v)
    except:
        return default

# =========================================================
# SESSION STATE MANAGEMENT
# =========================================================
def init_session():
    defaults = {
        "logged_in": False,
        "user_id": None,
        "user_name": "",
        "user_email": "",
        "user_role": "Recruiter",
        "user_department": "General",
        "theme_mode": "Light",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# =========================================================
# DATABASE LAYER
# =========================================================
def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False, timeout=30)

def execute_query(query, params=(), fetch=False, many=False):
    conn = get_connection()
    cur = conn.cursor()
    try:
        if many:
            cur.executemany(query, params)
        else:
            cur.execute(query, params)
        conn.commit()
        if fetch:
            return cur.fetchall()
    finally:
        conn.close()

def get_df(query, params=()):
    conn = get_connection()
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT,
        email TEXT UNIQUE,
        password TEXT,
        role TEXT,
        department TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_title TEXT,
        department TEXT,
        required_skills TEXT,
        experience_required TEXT,
        job_description TEXT,
        created_by TEXT,
        created_at TEXT,
        status TEXT DEFAULT 'Open'
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_name TEXT,
        candidate_email TEXT,
        phone TEXT,
        education TEXT,
        experience TEXT,
        skills TEXT,
        resume_text TEXT,
        job_role TEXT,
        score REAL,
        skill_score REAL,
        experience_score REAL,
        communication_score REAL,
        fake_resume_score REAL,
        confidence_level TEXT,
        recommendation TEXT,
        skill_gap TEXT,
        status TEXT,
        interview_status TEXT,
        applied_on TEXT,
        resume_file_path TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS interviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_name TEXT,
        candidate_email TEXT,
        job_role TEXT,
        interview_date TEXT,
        interview_time TEXT,
        mode TEXT,
        interviewer TEXT,
        status TEXT,
        feedback TEXT,
        created_at TEXT
    )
    """)

    conn.commit()

    # Schema Migrations
    cur.execute("PRAGMA table_info(candidates)")
    cand_cols = [r[1] for r in cur.fetchall()]
    if "resume_file_path" not in cand_cols:
        cur.execute("ALTER TABLE candidates ADD COLUMN resume_file_path TEXT")

    cur.execute("PRAGMA table_info(jobs)")
    job_cols = [r[1] for r in cur.fetchall()]
    if "status" not in job_cols:
        cur.execute("ALTER TABLE jobs ADD COLUMN status TEXT DEFAULT 'Open'")

    # Seed Default Accounts if missing
    default_users = [
        ("System Admin", "admin@ats.com", hash_password("admin123"), "Admin", "General"),
        ("Talent Recruiter", "recruiter@ats.com", hash_password("recruiter123"), "Recruiter", "HR"),
        ("Alex Johnson", "candidate@ats.com", hash_password("candidate123"), "Candidate", "General")
    ]
    for name, email, pwd, role, dept in default_users:
        cur.execute("SELECT id FROM users WHERE email=?", (email,))
        if not cur.fetchone():
            cur.execute("""
            INSERT INTO users (full_name, email, password, role, department, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (name, email, pwd, role, dept, now()))

    # Seed Sample Job if none exists
    cur.execute("SELECT count(*) FROM jobs")
    if cur.fetchone()[0] == 0:
        cur.execute("""
        INSERT INTO jobs (job_title, department, required_skills, experience_required, job_description, created_by, created_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "Senior Python & AI Engineer",
            "Data Science",
            "Python, Machine Learning, NLP, SQL, Docker, Scikit-Learn, Pandas, Git",
            "3+ years",
            "We are seeking an experienced AI/ML Engineer to build intelligent NLP pipelines, deploy scalable machine learning models, and develop predictive analytics solutions.",
            "admin@ats.com",
            now(),
            "Open"
        ))

    conn.commit()
    conn.close()

init_db()

# =========================================================
# DATA ACCESS FUNCTIONS
# =========================================================
def insert_user(full_name, email, password, role, department):
    try:
        execute_query("""
        INSERT INTO users (full_name, email, password, role, department, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (full_name.strip(), email.strip().lower(), hash_password(password), role, department, now()))
        return True
    except:
        return False

def login_user(email, password):
    email = email.strip().lower()
    rows = execute_query("SELECT * FROM users WHERE email=?", (email,), fetch=True)
    if rows:
        user = rows[0]
        stored_hash = user[3]
        if verify_password(password, stored_hash):
            return user
    return None

def get_users_df():
    return get_df("SELECT id, full_name, email, role, department, created_at FROM users ORDER BY id DESC")

def get_jobs_df(only_open=False):
    if only_open:
        return get_df("SELECT * FROM jobs WHERE status='Open' ORDER BY id DESC")
    return get_df("SELECT * FROM jobs ORDER BY id DESC")

def get_candidates_df(role=None):
    if role:
        return get_df("SELECT * FROM candidates WHERE job_role=? ORDER BY score DESC, id DESC", (role,))
    return get_df("SELECT * FROM candidates ORDER BY score DESC, id DESC")

def get_interviews_df():
    return get_df("SELECT * FROM interviews ORDER BY id DESC")

# =========================================================
# THEME & MODERN STYLING
# =========================================================
def apply_theme():
    if st.session_state.theme_mode == "Dark":
        bg = "#07111f"
        bg2 = "#0b1220"
        glass = "rgba(17, 24, 39, 0.76)"
        text = "#f8fafc"
        muted = "#94a3b8"
        border = "rgba(255,255,255,0.10)"
        sidebar_bg = "rgba(9, 14, 26, 0.94)"
        chip_bg = "rgba(59,130,246,0.18)"
    else:
        bg = "#f0f6ff"
        bg2 = "#f8fafc"
        glass = "rgba(255, 255, 255, 0.78)"
        text = "#0f172a"
        muted = "#475569"
        border = "rgba(15, 23, 42, 0.09)"
        sidebar_bg = "rgba(255, 255, 255, 0.88)"
        chip_bg = "rgba(37,99,235,0.10)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 10% 10%, rgba(59,130,246,0.14), transparent 24%),
            radial-gradient(circle at 85% 15%, rgba(168,85,247,0.12), transparent 22%),
            linear-gradient(135deg, {bg}, {bg2});
        color: {text};
    }}

    .block-container {{
        padding-top: 4.5rem !important;
        padding-bottom: 5rem !important;
        max-width: 96% !important;
    }}

    section[data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid {border};
    }}

    .brand-card {{
        padding: 16px 14px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(124,58,237,0.18));
        border: 1px solid rgba(255,255,255,0.16);
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        backdrop-filter: blur(14px);
        margin-bottom: 12px;
    }}

    .brand-title {{
        font-size: 22px;
        font-weight: 900;
        line-height: 1.2;
    }}

    .brand-sub {{
        font-size: 12px;
        color: {muted};
        margin-top: 3px;
    }}

    .hero-card {{
        width: 100%;
        padding: 24px 28px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(168,85,247,0.12)), {glass};
        border: 1px solid {border};
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(16px);
        margin-bottom: 22px;
    }}

    .hero-title {{
        font-size: 34px;
        font-weight: 900;
        color: {text};
        margin-bottom: 6px;
    }}

    .hero-subtitle {{
        font-size: 15px;
        color: {muted};
        line-height: 1.6;
    }}

    .glass-card {{
        background: {glass};
        border: 1px solid {border};
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        backdrop-filter: blur(16px);
        margin-bottom: 18px;
    }}

    .acceptance-card {{
        background: linear-gradient(135deg, rgba(16,185,129,0.10), rgba(59,130,246,0.10));
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 20px;
        padding: 22px;
        margin-bottom: 20px;
    }}

    [data-testid="stMetric"] {{
        background: {glass};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 14px 14px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
    }}

    .stButton > button {{
        border-radius: 12px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 8px 20px rgba(37,99,235,0.20);
    }}

    .skill-chip {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: {chip_bg};
        border: 1px solid rgba(59,130,246,0.22);
        margin: 3px 4px 3px 0;
        font-size: 12px;
        font-weight: 600;
        color: {text};
    }}

    .skill-chip-missing {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.28);
        margin: 3px 4px 3px 0;
        font-size: 12px;
        font-weight: 600;
        color: #ef4444;
    }}

    .badge-role {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 800;
        background: rgba(37,99,235,0.18);
        color: #2563eb;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

def render_hero(title, subtitle, badge="NextGen ATS"):
    st.markdown(f"""
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap;">
            <div>
                <div class="hero-title">{title}</div>
                <div class="hero-subtitle">{subtitle}</div>
            </div>
            <div style="
                padding:6px 14px;
                border-radius:999px;
                background:rgba(255,255,255,0.22);
                border:1px solid rgba(255,255,255,0.18);
                font-size:12px;
                font-weight:800;
            ">
                ✨ {badge}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_brand():
    st.markdown("""
    <div class="brand-card">
        <div class="brand-title">⚡ NextGen ATS</div>
        <div class="brand-sub">AI Resume Screening & Hiring Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

def render_skill_chips(skills_text, missing=False):
    if not skills_text:
        st.write("None")
        return
    skills = [s.strip() for s in str(skills_text).split(",") if s.strip()]
    css_class = "skill-chip-missing" if missing else "skill-chip"
    html = "".join([f'<span class="{css_class}">{skill}</span>' for skill in skills[:15]])
    st.markdown(html, unsafe_allow_html=True)

# =========================================================
# COMPREHENSIVE SKILLS & KNOWLEDGE BASE
# =========================================================
SKILL_CATALOG = [
    # Programming Languages
    "python", "java", "c++", "c#", "c", "javascript", "typescript", "go", "golang", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "dart", "matlab", "bash", "shell",
    # Web & Full Stack
    "html", "css", "react", "react.js", "angular", "vue", "vue.js", "node.js", "express",
    "django", "flask", "fastapi", "spring boot", "asp.net", "next.js", "tailwind", "bootstrap",
    # AI / ML / Data Science
    "machine learning", "deep learning", "nlp", "natural language processing", "computer vision",
    "scikit-learn", "tensorflow", "pytorch", "keras", "pandas", "numpy", "opencv", "scipy",
    "llm", "large language models", "generative ai", "transformers", "huggingface", "langchain",
    "rag", "bert", "gpt", "spacy", "nltk",
    # Data & Business Intelligence
    "sql", "power bi", "tableau", "excel", "data analysis", "data analytics", "data engineering",
    "big data", "apache spark", "spark", "hadoop", "kafka", "snowflake", "databricks", "etl",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "git", "github", "gitlab",
    "ci/cd", "linux", "unix", "terraform", "ansible", "jenkins", "devops",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "cassandra", "dynamodb",
    # Soft & Professional Skills
    "communication", "teamwork", "leadership", "problem solving", "agile", "scrum",
    "project management", "critical thinking", "collaboration", "adaptability"
]

# =========================================================
# DETERMINISTIC NLP & RESUME PARSING ENGINE
# =========================================================
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return "\n".join(text).strip()
    except Exception:
        return ""

def extract_text_from_docx(file):
    if Document is None:
        return ""
    try:
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs if p.text]).strip()
    except Exception:
        return ""

def extract_resume_text(uploaded_file):
    if uploaded_file is None:
        return ""
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif filename.endswith(".txt"):
        try:
            return uploaded_file.read().decode("utf-8", errors="ignore")
        except:
            return ""
    return ""

def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        clean_name = re.sub(r'[^a-zA-Z0-9_\.-]', '_', uploaded_file.name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{clean_name}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        uploaded_file.seek(0)
        with open(filepath, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)
        uploaded_file.seek(0)
        return filepath
    except Exception:
        return None

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0).lower() if match else ""

def extract_phone(text):
    patterns = [
        r'(\+?\d{1,3}[\s-]?)?\(?\d{3,5}\)?[\s.-]?\d{3,5}[\s.-]?\d{3,5}',
        r'(\+91[\-\s]?)?[6-9]\d{9}',
        r'\b\d{10}\b'
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            clean = m.group(0).strip()
            if len(re.sub(r'\D', '', clean)) >= 10:
                return clean
    return ""

def extract_name(text, default="Candidate"):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    invalid_keywords = [
        "resume", "curriculum vitae", "cv", "biodata", "profile", "contact",
        "email", "phone", "summary", "experience", "education", "skills",
        "objective", "page", "http", "www", "github", "linkedin"
    ]
    for line in lines[:10]:
        line_clean = line.strip()
        lower = line_clean.lower()
        if any(inv in lower for inv in invalid_keywords):
            continue
        if "@" in line_clean or re.search(r'\d', line_clean):
            continue
        words = line_clean.split()
        if 2 <= len(words) <= 4 and all(re.match(r'^[A-Za-z\.]+$', w) for w in words):
            return line_clean.title()
    return default

def extract_education(text):
    text_lower = text.lower()
    degrees = [
        (r'\b(ph\.?d|doctorate)\b', "Ph.D. / Doctorate"),
        (r'\b(m\.?tech|m\.?e|master of technology|master of engineering)\b', "M.Tech / M.E."),
        (r'\b(m\.?s|m\.?sc|master of science)\b', "M.S. / M.Sc."),
        (r'\b(m\.?b\.?a|master of business administration)\b', "MBA"),
        (r'\b(m\.?c\.?a|master of computer applications)\b', "MCA"),
        (r'\b(b\.?tech|b\.?e|bachelor of technology|bachelor of engineering)\b', "B.Tech / B.E."),
        (r'\b(b\.?s|b\.?sc|bachelor of science)\b', "B.S. / B.Sc."),
        (r'\b(b\.?c\.?a|bachelor of computer applications)\b', "BCA"),
        (r'\b(b\.?b\.?a|b\.?com|bachelor of commerce)\b', "Bachelor's Degree"),
        (r'\b(diploma|associate degree)\b', "Diploma / Associate"),
        (r'\b(bachelor|graduate|undergraduate)\b', "Bachelor's Degree")
    ]
    for pat, label in degrees:
        if re.search(pat, text_lower):
            return label
    return "Graduate"

def extract_skills_deterministic(text):
    text_clean = " " + text.lower() + " "
    found = set()
    for skill in SKILL_CATALOG:
        pattern = r'(?:\b|_)' + re.escape(skill) + r'(?:\b|_)'
        if re.search(pattern, text_clean):
            found.add(skill.title())
    return sorted(list(found))

def score_experience_deterministic(text, required_exp_str=""):
    text_lower = text.lower()
    matches = re.findall(r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?|year)\b', text_lower)
    exp_years = 0.0
    if matches:
        exp_years = max([float(m) for m in matches if float(m) < 40])
    else:
        # Check date range indicators (e.g. 2020 - 2024)
        year_matches = re.findall(r'\b(20\d{2}|19\d{2})\s*(?:-|–|to)\s*(20\d{2}|present|current)\b', text_lower)
        if year_matches:
            current_yr = datetime.now().year
            diffs = []
            for start, end in year_matches:
                s_val = int(start)
                e_val = current_yr if end in ["present", "current"] else int(end)
                if e_val >= s_val and (e_val - s_val) <= 30:
                    diffs.append(e_val - s_val)
            if diffs:
                exp_years = float(max(diffs))

    # Parse required experience
    req_years = 2.0
    req_match = re.search(r'(\d+(?:\.\d+)?)', str(required_exp_str))
    if req_match:
        req_years = float(req_match.group(1))

    if exp_years == 0.0:
        seniority_bonus = 0
        if any(term in text_lower for term in ["lead", "architect", "manager", "staff engineer"]):
            seniority_bonus = 35
        elif any(term in text_lower for term in ["senior", "specialist"]):
            seniority_bonus = 25
        elif any(term in text_lower for term in ["junior", "associate", "intern"]):
            seniority_bonus = 10
        score = 45.0 + seniority_bonus
    else:
        ratio = exp_years / max(1.0, req_years)
        score = min(100.0, 50.0 + (ratio * 45.0))

    return round(score, 1), f"{exp_years} Years" if exp_years > 0 else "Fresher / Entry Level"

def detect_fake_resume_deterministic(text):
    text_lower = text.lower()
    red_flags = [
        "lorem ipsum", "dolor sit amet", "[company name]", "[insert date]",
        "[your name]", "sample resume", "template provided by", "dummy text",
        "xyz university", "sample template", "placeholder text"
    ]
    detected_flags = sum(1 for flag in red_flags if flag in text_lower)
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    word_count = len(words)

    risk = 0.0
    if detected_flags > 0:
        risk += min(80.0, detected_flags * 30.0)
    if word_count < 60:
        risk += 40.0
    elif word_count < 120:
        risk += 15.0

    if word_count > 0:
        word_freq = pd.Series(words).value_counts()
        if not word_freq.empty and (word_freq.iloc[0] / word_count) > 0.12:
            risk += 25.0

    return min(100.0, round(risk, 1))

def score_communication_deterministic(text):
    words = re.findall(r'\b\w+\b', text)
    total_words = len(words)
    if total_words < 20:
        return 40.0
    unique_words = len(set(w.lower() for w in words))
    ttr = unique_words / total_words
    score = min(100.0, max(45.0, (ttr * 130.0)))
    return round(score, 1)

def compute_tfidf_similarity(resume_text, jd_text):
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(sim * 100), 2)
    except:
        return 0.0

# =========================================================
# PROFESSIONAL COMPANY ACCEPTANCE & INTERVIEWER INSIGHTS
# =========================================================
def generate_company_acceptance_insights(cand_data, company_name="the hiring company"):
    score = safe_float(cand_data.get("score", 50))
    skill_score = safe_float(cand_data.get("skill_score", 50))
    comm_score = safe_float(cand_data.get("communication_score", 50))
    fake_score = safe_float(cand_data.get("fake_resume_score", 0))
    job_role = cand_data.get("job_role", "Target Position")
    exp_str = str(cand_data.get("experience", ""))

    raw_skills = cand_data.get("skills", "")
    skills_list = [s.strip() for s in str(raw_skills).split(",") if s.strip()]
    gap_skills = [s.strip() for s in str(cand_data.get("skill_gap", "")).split(",") if s.strip()]

    # Deterministic Interview Acceptance Probability
    prob = int(min(98, max(15, (score * 0.75) + (comm_score * 0.15) + (skill_score * 0.10) - (fake_score * 0.2))))

    if prob >= 75:
        verdict = "Very High Probability of Interview Acceptance"
        verdict_color = "#10b981"
    elif prob >= 55:
        verdict = "Moderate to Strong Acceptance Probability"
        verdict_color = "#3b82f6"
    else:
        verdict = "Low Initial Acceptance (Optimization Recommended)"
        verdict_color = "#ef4444"

    company_acceptance_points = []
    if skills_list:
        top_skills = ", ".join(skills_list[:4])
        company_acceptance_points.append(
            f"**Verified Technical Competency**: Demonstrates concrete proficiency in **{top_skills}**, proving direct suitability for {company_name}'s production tech stack."
        )
    if "Years" in exp_str and "Fresher" not in exp_str:
        company_acceptance_points.append(
            f"**Demonstrated Industry Experience**: With **{exp_str}** of background, corporate interviewers will see immediate value and rapid onboarding potential."
        )
    else:
        company_acceptance_points.append(
            f"**High Adaptability & Strong Fundamentals**: Exhibits modern foundational engineering tools, making this candidate a high-potential hire adaptable to company workflows."
        )
    if comm_score >= 60:
        company_acceptance_points.append(
            f"**Clear Professional Presentation**: Structured, well-articulated documentation ({comm_score}% communication score) indicates effective cross-team stakeholder collaboration."
        )
    if score >= 65:
        company_acceptance_points.append(
            f"**Competitive Benchmark**: Overall profile ranks in the upper tier for **{job_role}**, making the application stand out in enterprise candidate screenings."
        )

    interviewer_prep_points = []
    if gap_skills:
        top_gaps = ", ".join(gap_skills[:3])
        interviewer_prep_points.append(
            f"**Bridge Target Skill Gaps**: The company requires **{top_gaps}**. Be ready to explain your conceptual familiarity, personal projects, or learning roadmap for these technologies."
        )
    interviewer_prep_points.append(
        "**Frame Projects Using the STAR Method**: Interviewers will look for measurable impact. Describe your work as Situation, Task, Action, and Quantifiable Results (e.g. latency reduced, users served, accuracy increased)."
    )
    interviewer_prep_points.append(
        "**Deep-Dive Technical Readiness**: Be prepared to explain technical trade-offs, debugging challenges, and architecture decisions made in your highlighted projects."
    )
    interviewer_prep_points.append(
        f"**Demonstrate Alignment with {company_name}**: Research current industry challenges in {job_role} and articulate how your specific skillset helps solve them."
    )

    resume_optimization_tips = []
    if gap_skills:
        resume_optimization_tips.append(f"Add projects highlighting experience with {', '.join(gap_skills[:2])}.")
    resume_optimization_tips.append("Quantify your achievements with numbers, percentages, and performance benchmarks.")
    resume_optimization_tips.append("Ensure GitHub repositories, live demo URLs, or portfolios are active and linked.")

    return {
        "acceptance_prob": prob,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "acceptance_points": company_acceptance_points,
        "interviewer_prep_points": interviewer_prep_points,
        "resume_optimization_tips": resume_optimization_tips
    }

# =========================================================
# END-TO-END RESUME ANALYSIS PIPELINE
# =========================================================
def analyze_resume_pipeline(resume_text, job_title, jd_skills, jd_description, jd_experience="2 years", uploaded_file_path=None):
    extracted_name = extract_name(resume_text)
    extracted_email = extract_email(resume_text)
    extracted_phone = extract_phone(resume_text)
    education = extract_education(resume_text)
    candidate_skills = extract_skills_deterministic(resume_text)
    exp_score, exp_str = score_experience_deterministic(resume_text, jd_experience)
    comm_score = score_communication_deterministic(resume_text)
    fake_score = detect_fake_resume_deterministic(resume_text)

    # Job description skill matching
    jd_skill_list = [s.strip().lower() for s in str(jd_skills).split(",") if s.strip()]
    cand_skill_lower = [s.lower() for s in candidate_skills]
    matched = [s for s in jd_skill_list if s in cand_skill_lower]
    missing = [s.title() for s in jd_skill_list if s not in cand_skill_lower]

    skill_score = round((len(matched) / max(1, len(jd_skill_list))) * 100, 2)
    jd_full_text = f"{job_title} {jd_skills} {jd_description}"
    tfidf_score = compute_tfidf_similarity(resume_text, jd_full_text)

    # Weighted Composite Score
    composite_score = round(
        (skill_score * 0.35) +
        (tfidf_score * 0.25) +
        (exp_score * 0.25) +
        (comm_score * 0.15),
        2
    )
    if fake_score > 40:
        composite_score = max(10.0, composite_score - (fake_score * 0.3))

    if composite_score >= 70:
        confidence = "High"
        recommendation = "Strongly Recommended"
        status = "Shortlisted"
    elif composite_score >= 50:
        confidence = "Medium"
        recommendation = "Recommended"
        status = "Under Review"
    else:
        confidence = "Low"
        recommendation = "Not Recommended"
        status = "Rejected"

    return {
        "candidate_name": extracted_name,
        "candidate_email": extracted_email if extracted_email else "unspecified@candidate.io",
        "phone": extracted_phone if extracted_phone else "Not Provided",
        "education": education,
        "experience": exp_str,
        "skills": ", ".join(candidate_skills),
        "resume_text": resume_text,
        "job_role": job_title,
        "score": composite_score,
        "skill_score": skill_score,
        "experience_score": exp_score,
        "communication_score": comm_score,
        "fake_resume_score": fake_score,
        "confidence_level": confidence,
        "recommendation": recommendation,
        "skill_gap": ", ".join(missing),
        "status": status,
        "interview_status": "Pending",
        "applied_on": now(),
        "resume_file_path": uploaded_file_path if uploaded_file_path else ""
    }

def save_candidate_record(cand):
    execute_query("""
    INSERT INTO candidates (
        candidate_name, candidate_email, phone, education, experience, skills, resume_text,
        job_role, score, skill_score, experience_score, communication_score, fake_resume_score,
        confidence_level, recommendation, skill_gap, status, interview_status, applied_on, resume_file_path
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        cand["candidate_name"], cand["candidate_email"], cand["phone"], cand["education"],
        cand["experience"], cand["skills"], cand["resume_text"], cand["job_role"],
        cand["score"], cand["skill_score"], cand["experience_score"], cand["communication_score"],
        cand["fake_resume_score"], cand["confidence_level"], cand["recommendation"],
        cand["skill_gap"], cand["status"], cand["interview_status"], cand["applied_on"],
        cand.get("resume_file_path", "")
    ))

# =========================================================
# COMPREHENSIVE PDF REPORT BUILDER (WITH ACCEPTANCE POINTS)
# =========================================================
def generate_candidate_pdf(candidate):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    # Page 1: Candidate Scorecard & Metadata
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "NextGen ATS: AI Resume Screening & Acceptance Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, 782, f"Evaluation Date: {now()} | NextGen ATS Intelligence")
    c.line(50, 774, 550, 774)

    insights = generate_company_acceptance_insights(candidate)

    fields = [
        ("Candidate Name", candidate.get('candidate_name', 'N/A')),
        ("Email Address", candidate.get('candidate_email', 'N/A')),
        ("Phone Number", candidate.get('phone', 'N/A')),
        ("Target Position", candidate.get('job_role', 'N/A')),
        ("Acceptance Probability", f"{insights['acceptance_prob']}% ({insights['verdict']})"),
        ("Composite Score", f"{candidate.get('score', 0)}%"),
        ("Skill Match", f"{candidate.get('skill_score', 0)}%"),
        ("Experience Score", f"{candidate.get('experience_score', 0)}% ({candidate.get('experience', '')})"),
        ("Communication Score", f"{candidate.get('communication_score', 0)}%"),
        ("Fake Risk Rating", f"{candidate.get('fake_resume_score', 0)}%"),
        ("Confidence Level", candidate.get('confidence_level', 'N/A')),
        ("AI Recommendation", candidate.get('recommendation', 'N/A')),
        ("Education", candidate.get('education', 'N/A')),
        ("Status", candidate.get('status', 'N/A')),
        ("Interview Status", candidate.get('interview_status', 'N/A')),
        ("Extracted Skills", candidate.get('skills', 'None')[:130]),
        ("Skill Gap (Missing)", candidate.get('skill_gap', 'None')[:130])
    ]

    y = 745
    for label, val in fields:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, y, f"{label}:")
        c.setFont("Helvetica", 10)
        c.drawString(185, y, str(val))
        y -= 22

    # Draw Section for Key Company Points on Page 1
    c.line(50, y - 6, 550, y - 6)
    y -= 22
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Company Acceptance Highlights:")
    y -= 18

    c.setFont("Helvetica", 9)
    for pt in insights['acceptance_points'][:3]:
        clean_pt = pt.replace("**", "").replace("*", "")
        c.drawString(55, y, f"• {clean_pt[:100]}")
        y -= 16

    c.line(50, y - 10, 550, y - 10)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, y - 22, "Confidential Candidate Screening Report - NextGen ATS Intelligence")

    # Page 2: Interviewer Preparation Dossier
    c.showPage()
    c.setFont("Helvetica-Bold", 15)
    c.drawString(50, 800, "Interviewer Preparation & Actionable Candidate Pointers")
    c.setFont("Helvetica", 10)
    c.drawString(50, 782, f"Target Role: {candidate.get('job_role', 'N/A')} | Candidate: {candidate.get('candidate_name', 'N/A')}")
    c.line(50, 774, 550, 774)

    y2 = 745
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y2, "1. What Will Impress the Company Interviewer:")
    y2 -= 20
    c.setFont("Helvetica", 9)
    for pt in insights['acceptance_points']:
        clean_pt = pt.replace("**", "").replace("*", "")
        c.drawString(60, y2, f"• {clean_pt[:105]}")
        y2 -= 18

    y2 -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y2, "2. Key Points to Prepare for the Interview:")
    y2 -= 20
    c.setFont("Helvetica", 9)
    for pt in insights['interviewer_prep_points']:
        clean_pt = pt.replace("**", "").replace("*", "")
        c.drawString(60, y2, f"• {clean_pt[:105]}")
        y2 -= 18

    y2 -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y2, "3. Resume Optimization Tips:")
    y2 -= 20
    c.setFont("Helvetica", 9)
    for pt in insights['resume_optimization_tips']:
        clean_pt = pt.replace("**", "").replace("*", "")
        c.drawString(60, y2, f"• {clean_pt[:105]}")
        y2 -= 18

    c.line(50, y2 - 15, 550, y2 - 15)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, y2 - 28, "Generated by NextGen ATS AI Engine | Page 2 of 2")

    c.save()
    buffer.seek(0)
    return buffer

# =========================================================
# STANDALONE RESUME & COMPANY ACCEPTANCE ANALYZER PAGE
# =========================================================
def resume_acceptance_review_page():
    render_hero(
        "🎯 Company Fit & Interviewer Acceptance Analyzer",
        "Evaluate any resume against company role expectations to get your exact Interviewer Acceptance Probability, key strengths, and interview prep pointers.",
        "Core AI Feature"
    )

    jobs_df = get_jobs_df(only_open=True)

    col_target, col_resume = st.columns([1, 1.2], gap="large")

    with col_target:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("1. Select Target Company & Role")

        mode = st.radio("Selection Mode", ["Choose Active Company Requisition", "Enter Custom Company & Job Details"], horizontal=True)

        if mode == "Choose Active Company Requisition" and not jobs_df.empty:
            sel_title = st.selectbox("Select Target Job Opening", jobs_df["job_title"].tolist())
            job_row = jobs_df[jobs_df["job_title"] == sel_title].iloc[0]
            target_role = job_row["job_title"]
            target_skills = job_row["required_skills"]
            target_exp = job_row["experience_required"]
            target_desc = job_row["job_description"]
            company_name = job_row.get("department", "The Company")

            st.markdown(f"**Department/Company:** {company_name}")
            st.markdown(f"**Experience Requirement:** {target_exp}")
            st.markdown("**Required Skills:**")
            render_skill_chips(target_skills)
        else:
            company_name = st.text_input("Company / Organization Name", value="Google / Enterprise Corp")
            target_role = st.text_input("Target Job Title", value="Senior AI / Python Engineer")
            target_exp = st.text_input("Required Experience", value="3+ years")
            target_skills = st.text_area("Required Technical Skills (Comma separated)", value="Python, Machine Learning, NLP, SQL, Docker, Scikit-Learn")
            target_desc = st.text_area("Job Description Summary", value="Build production NLP models, design robust microservices, and deliver scalable AI pipelines.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_resume:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("2. Upload or Paste Candidate Resume")

        up_file = st.file_uploader("Upload Resume File (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="rev_uploader")
        paste_text = st.text_area("Or Paste Raw Resume Text directly", height=150, placeholder="Paste resume contents here...")

        analyze_btn = st.button("🚀 Analyze for Company & Interviewer Acceptance", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_btn:
        resume_content = ""
        saved_path = None
        if up_file:
            saved_path = save_uploaded_file(up_file)
            resume_content = extract_resume_text(up_file)
        elif paste_text.strip():
            resume_content = paste_text.strip()

        if not resume_content:
            st.warning("Please upload a resume file or paste text to perform analysis.")
            return

        with st.spinner("Analyzing resume against corporate requirements..."):
            analysis = analyze_resume_pipeline(
                resume_text=resume_content,
                job_title=target_role,
                jd_skills=target_skills,
                jd_description=target_desc,
                jd_experience=target_exp,
                uploaded_file_path=saved_path
            )
            insights = generate_company_acceptance_insights(analysis, company_name=company_name)

        # RENDER FULL ACCEPTANCE DOSSIER
        st.markdown(f"""
        <div class="acceptance-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <h2 style="margin:0; font-size:28px;">🎯 Interviewer Acceptance Likelihood: {insights['acceptance_prob']}%</h2>
                    <p style="margin:4px 0 0 0; font-weight:700; color:{insights['verdict_color']}; font-size:16px;">{insights['verdict']}</p>
                </div>
                <div style="background:rgba(255,255,255,0.25); padding:8px 16px; border-radius:999px; font-weight:800; font-size:14px;">
                    Target: {target_role} ({company_name})
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Match Score", f"{analysis['score']}%")
        m2.metric("Skill Match", f"{analysis['skill_score']}%")
        m3.metric("Experience Match", f"{analysis['experience_score']}%")
        m4.metric("Communication Rating", f"{analysis['communication_score']}%")

        st.markdown("<br>", unsafe_allow_html=True)
        c_left, c_right = st.columns(2, gap="large")

        with c_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🏢 Professional Company Acceptance Points")
            st.caption("Specific strengths in this resume that will convince the company interviewer to accept the candidate:")
            for pt in insights['acceptance_points']:
                st.markdown(f"✅ {pt}")

            st.markdown("---")
            st.markdown("#### Matched Core Skills")
            render_skill_chips(analysis['skills'])
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 💡 Actionable Interviewer Preparation Pointers")
            st.caption("Key topics and presentation strategies to prepare before the interview to guarantee selection:")
            for pt in insights['interviewer_prep_points']:
                st.markdown(f"🎯 {pt}")

            if analysis.get('skill_gap'):
                st.markdown("---")
                st.markdown("#### Company Required Skills Currently Missing:")
                render_skill_chips(analysis['skill_gap'], missing=True)

            st.markdown("---")
            st.markdown("#### 📝 Resume Optimization Tips:")
            for tip in insights['resume_optimization_tips']:
                st.markdown(f"• {tip}")
            st.markdown('</div>', unsafe_allow_html=True)

        # DOWNLOAD PDF REPORT
        pdf_bytes = generate_candidate_pdf(analysis)
        st.download_button(
            label="📄 Download Official Company Acceptance & Interviewer Prep PDF",
            data=pdf_bytes,
            file_name=f"{analysis['candidate_name'].replace(' ', '_')}_Acceptance_Evaluation.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# =========================================================
# AUTHENTICATION PAGE
# =========================================================
def auth_page():
    render_hero(
        "⚡ NextGen ATS: AI Resume Screening & Hiring Intelligence",
        "Next-generation Applicant Tracking System powered by NLP, TF-IDF matching, deterministic scoring, and corporate interviewer acceptance intelligence.",
        "Production Ready"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NLP Modules", "10+ Live")
    c2.metric("User Roles", "Admin / Recruiter / Candidate")
    c3.metric("Company Acceptance", "Automated")
    c4.metric("Scoring", "Deterministic")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🚀 Platform Capabilities")
        st.markdown("""
        * **Company & Interviewer Acceptance Engine**: Analyzes resumes against company requirements and generates concrete professional points to get accepted.
        * **Deterministic AI Resume Parser**: Extracts skills, degrees, experience, and contact details with zero random bias.
        * **TF-IDF & N-gram Similarity**: Semantic matching of resumes directly against detailed Job Descriptions.
        * **Enterprise Role-Based Access Control**: Strict UI isolation for Administrators, Recruiters, and Candidates.
        * **Full Hiring Lifecycle**: From multi-resume batch upload to candidate pipeline, interview scheduling, and feedback.
        * **Intelligent Recruiter Assistant**: Dynamic query engine for instant candidate analytics and gap discovery.
        * **Exportable Reports**: Generate instant audit PDF scorecards and CSV exports.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Create Account"])

        with tab_login:
            st.markdown("#### Access Portal")
            email = st.text_input("Work Email", key="in_email", placeholder="admin@ats.com")
            password = st.text_input("Password", type="password", key="in_pwd", placeholder="••••••••")

            if st.button("Sign In to ATS", use_container_width=True):
                user = login_user(email, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user[0]
                    st.session_state.user_name = user[1]
                    st.session_state.user_email = user[2]
                    st.session_state.user_role = user[4]
                    st.session_state.user_department = user[5] or "General"
                    st.success(f"Welcome back, {user[1]}!")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

            st.markdown("---")
            st.caption("Default Demo Logins:")
            st.code("Admin:     admin@ats.com     / admin123\nRecruiter: recruiter@ats.com / recruiter123\nCandidate: candidate@ats.com / candidate123", language="text")

        with tab_register:
            st.markdown("#### New Account Registration")
            r_name = st.text_input("Full Name", key="reg_name")
            r_email = st.text_input("Email", key="reg_email")
            r_pwd = st.text_input("Create Password", type="password", key="reg_pwd")
            r_role = st.selectbox("Role", ["Candidate", "Recruiter", "HR Manager", "Admin"], key="reg_role")
            r_dept = st.selectbox("Department", ["General", "Engineering", "Data Science", "HR", "Sales", "Operations"], key="reg_dept")

            if st.button("Complete Registration", use_container_width=True):
                if r_name and r_email and r_pwd:
                    if insert_user(r_name, r_email, r_pwd, r_role, r_dept):
                        st.success("Account created successfully! You can now sign in.")
                    else:
                        st.error("Account already exists with this email address.")
                else:
                    st.warning("Please fill in all required fields.")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RECRUITER / ADMIN DASHBOARD
# =========================================================
def dashboard_page():
    render_hero(
        f"📊 Executive Dashboard — {st.session_state.user_name}",
        "Comprehensive live metrics, hiring funnel analytics, and candidate distribution."
    )

    cands = get_candidates_df()
    jobs = get_jobs_df()
    interviews = get_interviews_df()

    total_cands = len(cands)
    shortlisted = len(cands[cands["status"] == "Shortlisted"]) if not cands.empty else 0
    open_jobs = len(jobs[jobs["status"] == "Open"]) if not jobs.empty else 0
    scheduled_ints = len(interviews[interviews["status"] == "Scheduled"]) if not interviews.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applicants", total_cands)
    c2.metric("Shortlisted Profiles", shortlisted)
    c3.metric("Open Job Openings", open_jobs)
    c4.metric("Active Interviews", scheduled_ints)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Candidate Status Funnel")
        if not cands.empty:
            status_counts = cands["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig = px.pie(status_counts, names="Status", values="Count", hole=0.45,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No candidate applications recorded yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Top Ranked Candidates")
        if not cands.empty:
            top_df = cands.head(8)[["candidate_name", "job_role", "score"]]
            fig = px.bar(top_df, x="score", y="candidate_name", orientation="h",
                         color="score", color_continuous_scale="Viridis", text="score")
            fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No candidates scored yet.")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# JOB MANAGEMENT
# =========================================================
def job_management_page():
    render_hero("💼 Job Requisitions & Management", "Create, monitor, and update corporate openings and screening criteria.")

    with st.expander("➕ Create New Job Requisition", expanded=False):
        with st.form("create_job_form"):
            c1, c2 = st.columns(2)
            with c1:
                title = st.text_input("Job Title*", placeholder="e.g. Senior Machine Learning Engineer")
                dept = st.selectbox("Department", ["Data Science", "Software Engineering", "Product", "Operations", "Finance", "HR"])
                exp_req = st.text_input("Required Experience*", placeholder="e.g. 3+ years")
            with c2:
                req_skills = st.text_area("Required Skills (Comma separated)*", placeholder="Python, PyTorch, Docker, SQL, NLP")
                desc = st.text_area("Job Description*", placeholder="Briefly describe responsibilities and qualifications...")

            submit_job = st.form_submit_button("Publish Job Opening")
            if submit_job:
                if title and req_skills and desc:
                    execute_query("""
                    INSERT INTO jobs (job_title, department, required_skills, experience_required, job_description, created_by, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'Open')
                    """, (title, dept, req_skills, exp_req, desc, st.session_state.user_email, now()))
                    st.success(f"Job opening '{title}' published successfully.")
                    st.rerun()
                else:
                    st.warning("Please fill in all required fields.")

    jobs_df = get_jobs_df()
    if jobs_df.empty:
        st.info("No job openings posted yet.")
        return

    st.subheader(f"All Job Openings ({len(jobs_df)})")
    for _, job in jobs_df.iterrows():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c_title, c_stats, c_btn = st.columns([2.5, 1.2, 0.8])
        with c_title:
            st.markdown(f"### {job['job_title']} `({job.get('status', 'Open')})`")
            st.caption(f"Dept: {job['department']} | Experience: {job['experience_required']} | Posted: {job['created_at']}")
            st.write(job['job_description'])
            render_skill_chips(job['required_skills'])
        with c_stats:
            cands_count = len(get_df("SELECT id FROM candidates WHERE job_role=?", (job['job_title'],)))
            st.metric("Total Applicants", cands_count)
        with c_btn:
            new_status = "Closed" if job.get('status', 'Open') == "Open" else "Open"
            if st.button(f"Mark {new_status}", key=f"toggle_job_{job['id']}"):
                execute_query("UPDATE jobs SET status=? WHERE id=?", (new_status, job['id']))
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# BATCH RESUME SCREENING LAB
# =========================================================
def screening_lab_page():
    render_hero("📤 Batch AI Screening Lab", "Upload multiple resumes simultaneously and run automated NLP evaluations.")

    jobs_df = get_jobs_df(only_open=True)
    if jobs_df.empty:
        st.warning("No active job postings found. Please create a job first.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_job_title = st.selectbox("Select Target Job", jobs_df["job_title"].tolist())
        target_job = jobs_df[jobs_df["job_title"] == selected_job_title].iloc[0]

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"**Required Experience:** {target_job['experience_required']}")
        st.markdown("**Target Skills:**")
        render_skill_chips(target_job['required_skills'])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        uploaded_resumes = st.file_uploader(
            "Upload Resumes (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if st.button("🚀 Execute Batch Screening", use_container_width=True):
            if uploaded_resumes:
                progress_bar = st.progress(0)
                processed = []
                for i, file in enumerate(uploaded_resumes):
                    saved_path = save_uploaded_file(file)
                    text = extract_resume_text(file)
                    if text:
                        analysis = analyze_resume_pipeline(
                            resume_text=text,
                            job_title=target_job["job_title"],
                            jd_skills=target_job["required_skills"],
                            jd_description=target_job["job_description"],
                            jd_experience=target_job["experience_required"],
                            uploaded_file_path=saved_path
                        )
                        save_candidate_record(analysis)
                        processed.append(analysis)
                    progress_bar.progress((i + 1) / len(uploaded_resumes))

                st.success(f"Successfully screened and indexed {len(processed)} resumes!")
                if processed:
                    res_df = pd.DataFrame(processed)[["candidate_name", "score", "skill_score", "experience", "confidence_level", "recommendation", "status"]]
                    st.dataframe(res_df, use_container_width=True)
            else:
                st.warning("Please choose one or more resume files to analyze.")

# =========================================================
# CANDIDATE PIPELINE
# =========================================================
def candidate_pipeline_page():
    render_hero("📋 Candidate Pipeline & ATS Actions", "Manage candidate screening results, update statuses, view company acceptance points, and download evaluation reports.")

    all_cands = get_candidates_df()
    if all_cands.empty:
        st.info("No candidates registered in the pipeline yet.")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        status_filter = st.selectbox("Filter Status", ["All"] + sorted(all_cands["status"].dropna().unique().tolist()))
    with c2:
        roles = ["All"] + sorted(all_cands["job_role"].dropna().unique().tolist())
        role_filter = st.selectbox("Filter Job Role", roles)
    with c3:
        search_query = st.text_input("Search Candidate / Email", placeholder="Type name or email...")

    filtered_df = all_cands.copy()
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["status"] == status_filter]
    if role_filter != "All":
        filtered_df = filtered_df[filtered_df["job_role"] == role_filter]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["candidate_name"].str.lower().str.contains(q) |
            filtered_df["candidate_email"].str.lower().str.contains(q)
        ]

    st.write(f"Showing **{len(filtered_df)}** matching candidate profiles:")

    for _, cand in filtered_df.iterrows():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col_main, col_metrics, col_actions = st.columns([2.5, 1.2, 1.3])

        with col_main:
            st.markdown(f"### 👤 {cand['candidate_name']}")
            st.caption(f"📧 {cand['candidate_email']} | 📞 {cand['phone']} | Applied: {cand['applied_on']}")
            st.markdown(f"**Target Role:** {cand['job_role']} | **Degree:** {cand['education']} | **Experience:** {cand['experience']}")
            st.markdown("**Matched Skills:**")
            render_skill_chips(cand['skills'])
            if cand.get('skill_gap'):
                st.markdown("**Missing Skills:**")
                render_skill_chips(cand['skill_gap'], missing=True)

            # COMPANY ACCEPTANCE EXPANDER
            insights = generate_company_acceptance_insights(cand)
            with st.expander("🎯 Company Acceptance Points & Interviewer Guide"):
                st.markdown(f"**Interviewer Acceptance Probability:** `{insights['acceptance_prob']}%` ({insights['verdict']})")
                st.markdown("**Why Interviewers Will Accept This Profile:**")
                for pt in insights['acceptance_points']:
                    st.markdown(f"• {pt}")
                st.markdown("**Questions/Topics to Test in Interview:**")
                for pt in insights['interviewer_prep_points']:
                    st.markdown(f"• {pt}")

        with col_metrics:
            st.metric("Overall Score", f"{cand['score']}%")
            st.metric("Skill Match", f"{cand['skill_score']}%")
            st.metric("Fake Risk", f"{cand['fake_resume_score']}%")
            st.write(f"**Confidence:** {cand['confidence_level']}")
            st.write(f"**Recommendation:** {cand['recommendation']}")

        with col_actions:
            st.markdown(f"**Current Status:** `{cand['status']}`")
            st.markdown(f"**Interview:** `{cand['interview_status']}`")

            new_status = st.selectbox(
                "Change Status",
                ["Shortlisted", "Under Review", "Interview", "Rejected", "Hired"],
                index=["Shortlisted", "Under Review", "Interview", "Rejected", "Hired"].index(cand['status']) if cand['status'] in ["Shortlisted", "Under Review", "Interview", "Rejected", "Hired"] else 1,
                key=f"status_sel_{cand['id']}"
            )
            if st.button("Update Status", key=f"btn_upd_{cand['id']}"):
                execute_query("UPDATE candidates SET status=? WHERE id=?", (new_status, cand['id']))
                st.success("Status updated!")
                st.rerun()

            pdf_bytes = generate_candidate_pdf(cand)
            st.download_button(
                label="📄 PDF AI Report",
                data=pdf_bytes,
                file_name=f"{cand['candidate_name'].replace(' ', '_')}_Report.pdf",
                mime="application/pdf",
                key=f"dl_pdf_{cand['id']}",
                use_container_width=True
            )

            file_path = cand.get('resume_file_path')
            if file_path and os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    st.download_button(
                        label="📥 Original Resume",
                        data=f.read(),
                        file_name=os.path.basename(file_path),
                        key=f"dl_orig_{cand['id']}",
                        use_container_width=True
                    )

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# INTERVIEW SCHEDULING & FEEDBACK MANAGEMENT
# =========================================================
def interview_page():
    render_hero("📅 Interview Scheduling & Feedback Hub", "Coordinate technical rounds, log interviewer remarks, and finalize hiring outcomes.")

    tab_sched, tab_manage = st.tabs(["🗓️ Schedule Interview", "📝 Manage Feedback & Results"])

    with tab_sched:
        eligible = get_df("SELECT * FROM candidates WHERE status IN ('Shortlisted', 'Under Review', 'Interview') ORDER BY score DESC")
        if eligible.empty:
            st.info("No candidates currently eligible for scheduling.")
        else:
            cand_names = eligible["candidate_name"].tolist()
            selected_cand = st.selectbox("Select Candidate", cand_names)
            cand_row = eligible[eligible["candidate_name"] == selected_cand].iloc[0]

            with st.form("schedule_interview_form"):
                c1, c2 = st.columns(2)
                with c1:
                    i_date = st.date_input("Interview Date", value=date.today())
                    i_mode = st.selectbox("Interview Mode", ["Online (Google Meet)", "Online (Zoom)", "In-Person", "Telephonic"])
                with c2:
                    i_time = st.time_input("Interview Time", value=time(10, 30))
                    i_interviewer = st.text_input("Lead Interviewer", placeholder="e.g. Senior Tech Lead")

                schedule_submit = st.form_submit_button("Confirm Interview Schedule")
                if schedule_submit:
                    execute_query("""
                    INSERT INTO interviews (candidate_name, candidate_email, job_role, interview_date, interview_time, mode, interviewer, status, feedback, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'Scheduled', '', ?)
                    """, (cand_row["candidate_name"], cand_row["candidate_email"], cand_row["job_role"],
                          str(i_date), str(i_time), i_mode, i_interviewer, now()))
                    execute_query("UPDATE candidates SET interview_status='Scheduled', status='Interview' WHERE candidate_email=?", (cand_row["candidate_email"],))
                    st.success(f"Interview scheduled for {cand_row['candidate_name']}!")
                    st.rerun()

    with tab_manage:
        int_df = get_interviews_df()
        if int_df.empty:
            st.info("No scheduled interviews recorded.")
        else:
            st.dataframe(int_df, use_container_width=True)
            st.markdown("---")
            st.subheader("Log Interviewer Feedback & Decision")
            int_options = [f"ID {r['id']}: {r['candidate_name']} ({r['job_role']}) - {r['interview_date']}" for _, r in int_df.iterrows()]
            selected_int_str = st.selectbox("Select Interview Record", int_options)
            int_id = int(selected_int_str.split(":")[0].replace("ID", "").strip())
            current_int = int_df[int_df["id"] == int_id].iloc[0]

            with st.form("feedback_form"):
                new_int_status = st.selectbox(
                    "Interview Outcome",
                    ["Scheduled", "Completed - Passed", "Completed - On Hold", "Rejected", "Cancelled"],
                    index=["Scheduled", "Completed - Passed", "Completed - On Hold", "Rejected", "Cancelled"].index(current_int['status']) if current_int['status'] in ["Scheduled", "Completed - Passed", "Completed - On Hold", "Rejected", "Cancelled"] else 0
                )
                feedback_text = st.text_area("Detailed Interview Feedback & Observations", value=current_int['feedback'] or "")

                if st.form_submit_button("Save Feedback & Update Candidate"):
                    execute_query("UPDATE interviews SET status=?, feedback=? WHERE id=?", (new_int_status, feedback_text, int_id))
                    cand_status_sync = "Shortlisted" if "Passed" in new_int_status else ("Rejected" if "Rejected" in new_int_status else "Interview")
                    execute_query("UPDATE candidates SET interview_status=?, status=? WHERE candidate_email=?", (new_int_status, cand_status_sync, current_int['candidate_email']))
                    st.success("Interview feedback saved successfully!")
                    st.rerun()

# =========================================================
# RECRUITER AI QUERY ASSISTANT
# =========================================================
def recruiter_assistant_page():
    render_hero("💬 Recruiter AI Assistant", "Ask questions about candidates, missing skills, score distribution, and hiring pipeline metrics.")

    col_q, col_quick = st.columns([2, 1])
    with col_q:
        query = st.text_input("Ask a recruitment query:", placeholder="e.g. Show top candidates for Python Engineer")
    with col_quick:
        st.write("**Quick Insights:**")
        q_pick = st.selectbox("Pick Preset Query", [
            "",
            "Top 5 highest scoring candidates",
            "Candidates with skill gaps",
            "Show scheduled interviews",
            "Summarize applicant pipeline"
        ])
        if q_pick:
            query = q_pick

    if st.button("Generate AI Insights", use_container_width=True) or query:
        if not query:
            st.info("Enter or select a question to see insights.")
            return

        q = query.lower()
        cand_df = get_candidates_df()

        if cand_df.empty:
            st.warning("No candidate data available to analyze.")
            return

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if "top" in q or "best" in q or "highest" in q:
            top_c = cand_df.sort_values("score", ascending=False).head(5)
            st.success("🎯 Top Ranked Candidate Profiles:")
            st.dataframe(top_c[["candidate_name", "job_role", "score", "skill_score", "experience", "recommendation"]], use_container_width=True)

        elif "gap" in q or "missing" in q:
            st.info("🔍 Candidate Skill Gaps Analysis:")
            gaps = cand_df[cand_df["skill_gap"].str.len() > 0][["candidate_name", "job_role", "skill_gap", "score"]]
            st.dataframe(gaps, use_container_width=True)

        elif "interview" in q:
            st.info("📅 Scheduled Interviews Summary:")
            st.dataframe(get_interviews_df(), use_container_width=True)

        elif "summary" in q or "pipeline" in q or "stat" in q:
            st.success("📊 Pipeline Executive Summary:")
            total = len(cand_df)
            short = len(cand_df[cand_df["status"] == "Shortlisted"])
            avg_score = round(cand_df["score"].mean(), 1)
            st.write(f"* **Total Candidates Processed:** {total}")
            st.write(f"* **Shortlisting Rate:** {round(short / total * 100, 1)}% ({short} candidates)")
            st.write(f"* **Mean Match Score:** {avg_score}%")
            st.write(f"* **High Confidence Profiles:** {len(cand_df[cand_df['confidence_level'] == 'High'])}")

        else:
            matches = cand_df[
                cand_df["job_role"].str.lower().str.contains(q) |
                cand_df["skills"].str.lower().str.contains(q) |
                cand_df["candidate_name"].str.lower().str.contains(q)
            ]
            if not matches.empty:
                st.success(f"Found {len(matches)} matching candidates:")
                st.dataframe(matches[["candidate_name", "job_role", "score", "skills", "status"]], use_container_width=True)
            else:
                st.info("💡 Recommendation: Filter candidates by required skill or check candidates with High Confidence levels in the Pipeline tab.")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# CANDIDATE SELF-SERVICE PORTAL
# =========================================================
def candidate_portal_page():
    render_hero(
        f"🧑‍🎓 Candidate Career Portal — {st.session_state.user_name}",
        "Browse open opportunities, submit your resume for automated screening, inspect your company acceptance points, and track your interview status."
    )

    tab_apply, tab_my_apps, tab_my_ints = st.tabs(["💼 Open Jobs & Apply", "📄 My Applications & Acceptance Points", "📅 My Interviews"])

    with tab_apply:
        open_jobs = get_jobs_df(only_open=True)
        if open_jobs.empty:
            st.info("No open positions currently available. Please check back soon!")
        else:
            for _, job in open_jobs.iterrows():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"### {job['job_title']}")
                st.caption(f"Department: {job['department']} | Experience Required: {job['experience_required']}")
                st.write(job['job_description'])
                st.write("**Required Skills:**")
                render_skill_chips(job['required_skills'])

                with st.expander(f"Apply for {job['job_title']}"):
                    cand_name = st.text_input("Full Name", value=st.session_state.user_name, key=f"c_name_{job['id']}")
                    cand_email = st.text_input("Email", value=st.session_state.user_email, key=f"c_email_{job['id']}")
                    cand_phone = st.text_input("Phone Number", placeholder="+91 9876543210", key=f"c_phone_{job['id']}")
                    cand_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key=f"c_file_{job['id']}")

                    if st.button("Submit Application", key=f"apply_btn_{job['id']}"):
                        if cand_file and cand_name and cand_email:
                            saved_path = save_uploaded_file(cand_file)
                            text = extract_resume_text(cand_file)
                            analysis = analyze_resume_pipeline(
                                resume_text=text,
                                job_title=job["job_title"],
                                jd_skills=job["required_skills"],
                                jd_description=job["job_description"],
                                jd_experience=job["experience_required"],
                                uploaded_file_path=saved_path
                            )
                            analysis["candidate_name"] = cand_name
                            analysis["candidate_email"] = cand_email
                            analysis["phone"] = cand_phone if cand_phone else analysis["phone"]

                            save_candidate_record(analysis)
                            st.success("🎉 Application submitted and analyzed successfully!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.warning("Please upload your resume and complete all required fields.")
                st.markdown('</div>', unsafe_allow_html=True)

    with tab_my_apps:
        my_cands = get_df("SELECT * FROM candidates WHERE candidate_email=? ORDER BY id DESC", (st.session_state.user_email,))
        if my_cands.empty:
            st.info("You haven't submitted any job applications yet.")
        else:
            for _, app in my_cands.iterrows():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                c_inf, c_score = st.columns([2, 1])
                insights = generate_company_acceptance_insights(app)

                with c_inf:
                    st.markdown(f"### {app['job_role']}")
                    st.caption(f"Applied on: {app['applied_on']} | Status: `{app['status']}`")
                    st.markdown(f"**Extracted Education:** {app['education']} | **Experience:** {app['experience']}")
                    st.markdown("**Your Identified Skills:**")
                    render_skill_chips(app['skills'])
                    if app.get('skill_gap'):
                        st.markdown("**Recommended Skills to Learn:**")
                        render_skill_chips(app['skill_gap'], missing=True)

                    # PROMINENT ACCEPTANCE INSIGHTS
                    st.markdown(f"""
                    <div style="background:rgba(16,185,129,0.12); border-left:4px solid #10b981; padding:12px; border-radius:10px; margin-top:14px;">
                        <h4 style="margin:0 0 6px 0;">🎯 Interviewer Acceptance Likelihood: {insights['acceptance_prob']}%</h4>
                        <p style="margin:0; font-size:13px; color:{insights['verdict_color']}; font-weight:700;">{insights['verdict']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("🏢 View Professional Company Points for this Resume", expanded=True):
                        st.markdown("**Why the Company Interviewer Will Like Your Profile:**")
                        for pt in insights['acceptance_points']:
                            st.markdown(f"✅ {pt}")
                        st.markdown("**What to Prepare for the Interview:**")
                        for pt in insights['interviewer_prep_points']:
                            st.markdown(f"🎯 {pt}")

                with c_score:
                    st.metric("Match Score", f"{app['score']}%")
                    st.write(f"**Recommendation:** {app['recommendation']}")
                    st.write(f"**Interview Status:** `{app['interview_status']}`")

                    pdf_bytes = generate_candidate_pdf(app)
                    st.download_button(
                        "📄 Download Evaluation & Prep Report",
                        data=pdf_bytes,
                        file_name=f"My_Evaluation_{app['job_role'].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        key=f"my_dl_{app['id']}"
                    )
                st.markdown('</div>', unsafe_allow_html=True)

    with tab_my_ints:
        my_ints = get_df("SELECT * FROM interviews WHERE candidate_email=? ORDER BY id DESC", (st.session_state.user_email,))
        if my_ints.empty:
            st.info("No interviews currently scheduled.")
        else:
            st.dataframe(my_ints[["job_role", "interview_date", "interview_time", "mode", "interviewer", "status", "feedback"]], use_container_width=True)

# =========================================================
# ADMIN CONTROL PANEL
# =========================================================
def admin_panel_page():
    if st.session_state.user_role != "Admin":
        st.error("⛔ Access Denied. Administrator privileges required.")
        return

    render_hero("🛡️ Administrator Control Center", "Manage user credentials, roles, database status, and system operations.")

    users_df = get_users_df()
    st.subheader(f"Platform Users ({len(users_df)})")
    st.dataframe(users_df, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Update User Role")
        user_emails = users_df["email"].tolist()
        u_sel = st.selectbox("Select User", user_emails)
        new_role = st.selectbox("New Role", ["Admin", "Recruiter", "HR Manager", "Candidate"])

        if st.button("Apply Role Change"):
            execute_query("UPDATE users SET role=? WHERE email=?", (new_role, u_sel))
            st.success(f"Updated role for {u_sel} to {new_role}!")
            st.rerun()

    with col2:
        st.markdown("### Reset User Password")
        p_email = st.selectbox("Select Account for Reset", user_emails, key="reset_usr")
        p_new = st.text_input("New Password", type="password", key="reset_new_pwd")

        if st.button("Reset Password"):
            if p_new:
                execute_query("UPDATE users SET password=? WHERE email=?", (hash_password(p_new), p_email))
                st.success(f"Password reset for {p_email} successfully!")
            else:
                st.warning("Enter a valid password.")

# =========================================================
# REPORTS & EXPORT CENTER
# =========================================================
def reports_center_page():
    render_hero("📊 Reports & Export Center", "Download structured hiring analytics, applicant registries, and interview data.")

    cands_df = get_candidates_df()
    jobs_df = get_jobs_df()
    ints_df = get_interviews_df()

    t1, t2, t3 = st.tabs(["📄 Candidate Master", "💼 Requisitions", "📅 Interview Ledger"])

    with t1:
        st.dataframe(cands_df, use_container_width=True)
        if not cands_df.empty:
            csv_data = cands_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Candidates CSV", csv_data, "Candidates_Export.csv", "text/csv")

    with t2:
        st.dataframe(jobs_df, use_container_width=True)
        if not jobs_df.empty:
            csv_data = jobs_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Jobs CSV", csv_data, "Jobs_Export.csv", "text/csv")

    with t3:
        st.dataframe(ints_df, use_container_width=True)
        if not ints_df.empty:
            csv_data = ints_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Interviews CSV", csv_data, "Interviews_Export.csv", "text/csv")

# =========================================================
# SETTINGS & ABOUT
# =========================================================
def settings_page():
    render_hero("⚙️ Platform Settings", "Customize visual themes and session preferences.")
    theme = st.selectbox("UI Theme", ["Light", "Dark"], index=0 if st.session_state.theme_mode == "Light" else 1)
    if theme != st.session_state.theme_mode:
        st.session_state.theme_mode = theme
        st.rerun()

def about_page():
    render_hero("ℹ️ About NextGen ATS", "NextGen ATS: AI Resume Screening & Interview Acceptance Engine.")
    st.markdown("""
    ### Project Purpose & Core AI Capabilities
    An intelligent, enterprise-grade Applicant Tracking System (ATS) combining deterministic NLP heuristics,
    TF-IDF document vectorization, and cosine similarity with **Company & Interviewer Acceptance Analytics**.

    #### What Makes This System Unique:
    * **Interviewer Acceptance Likelihood**: Mathematically models the probability of an applicant passing company screening.
    * **Professional Company Acceptance Points**: Pinpoints the exact technical and experiential highlights in the resume that impress hiring panels.
    * **Actionable Interview Preparation Pointers**: Identifies missing skills and coaches candidates on project presentation techniques to guarantee selection.
    * **Full Role-Based Access Control**: Strict UI isolation for Administrators, Recruiters, and Candidates.
    * **Deterministic Scoring**: Eliminates random noise and biases from resume evaluations.
    """)

# =========================================================
# APPLICATION ENTRYPOINT & ROUTING
# =========================================================
if not st.session_state.logged_in:
    auth_page()
else:
    role = st.session_state.user_role

    with st.sidebar:
        render_brand()
        st.markdown(f"**Logged in:** {st.session_state.user_name}")
        st.caption(f"{st.session_state.user_email}")
        st.markdown(f'<span class="badge-role">{role}</span>', unsafe_allow_html=True)
        st.markdown("---")

        # Dynamic Role-Based Menu (with dedicated Company Acceptance feature)
        if role == "Admin":
            nav_items = [
                "📊 Dashboard",
                "🎯 Company Acceptance Review",
                "🛡️ Admin Panel",
                "💼 Job Management",
                "📤 Screening Lab",
                "📋 Candidate Pipeline",
                "📅 Interview Hub",
                "💬 AI Recruiter Assistant",
                "📊 Reports Center",
                "⚙️ Settings",
                "ℹ️ About"
            ]
        elif role in ["Recruiter", "HR Manager"]:
            nav_items = [
                "📊 Dashboard",
                "🎯 Company Acceptance Review",
                "💼 Job Management",
                "📤 Screening Lab",
                "📋 Candidate Pipeline",
                "📅 Interview Hub",
                "💬 AI Recruiter Assistant",
                "📊 Reports Center",
                "⚙️ Settings",
                "ℹ️ About"
            ]
        else: # Candidate
            nav_items = [
                "🧑‍🎓 Career Portal",
                "🎯 Company Acceptance Review",
                "⚙️ Settings",
                "ℹ️ About"
            ]

        choice = st.radio("Navigation", nav_items)
        st.markdown("---")

        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.user_name = ""
            st.session_state.user_email = ""
            st.session_state.user_role = "Recruiter"
            st.rerun()

    # Route Page
    if "Dashboard" in choice:
        dashboard_page()
    elif "Company Acceptance Review" in choice:
        resume_acceptance_review_page()
    elif "Admin Panel" in choice:
        admin_panel_page()
    elif "Job Management" in choice:
        job_management_page()
    elif "Screening Lab" in choice:
        screening_lab_page()
    elif "Candidate Pipeline" in choice:
        candidate_pipeline_page()
    elif "Interview Hub" in choice:
        interview_page()
    elif "AI Recruiter Assistant" in choice:
        recruiter_assistant_page()
    elif "Reports Center" in choice:
        reports_center_page()
    elif "Career Portal" in choice:
        candidate_portal_page()
    elif "Settings" in choice:
        settings_page()
    elif "About" in choice:
        about_page()