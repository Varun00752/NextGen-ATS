# ⚡ NextGen ATS: AI Resume Screening & Hiring Intelligence

An intelligent, full-featured Applicant Tracking System powered by Natural Language Processing (NLP), TF-IDF semantic vectorization, and deterministic candidate scoring with **Company & Interviewer Acceptance Analytics**.

Built with Python and Streamlit.

---

## 🌟 Key Features

* **Interviewer Acceptance Likelihood & Company Fit**:
  * Calculates mathematically grounded **Interviewer Acceptance Probability (%)**.
  * Generates concrete **Professional Company Acceptance Points** detailing why the corporate interviewer will accept the candidate.
  * Provides actionable **Interview Preparation Pointers** and question strategies.
* **Deterministic AI Resume Parser**: Extracts contact information (email, phone), degrees/education, work experience duration, and 180+ technical skills without random scoring bias.
* **TF-IDF & N-Gram Job Matching**: Computes cosine similarity between candidate resumes and specific Job Descriptions using unigrams and bigrams.
* **Role-Based Access Control (RBAC)**:
  * **Admin**: User account management, role reassignments, password resets, and platform monitoring.
  * **Recruiter / HR Manager**: Job requisition creation, batch resume screening, interactive candidate pipeline, interview scheduling, and AI assistant.
  * **Candidate**: Dedicated self-service career portal to explore job openings, 1-click apply, track application status, and view interview invitations.
* **Batch Screening Lab**: Upload and evaluate multiple resumes (PDF, DOCX, TXT) simultaneously with real-time ranking.
* **Candidate Pipeline & ATS Actions**: Filter by role/status, search by candidate, one-click status transitions (Shortlisted, Under Review, Interview, Rejected, Hired), and instant PDF report download.
* **Interview Lifecycle & Feedback**: Schedule interviews across modes (Google Meet, Zoom, In-Person, Telephonic) and log interviewer remarks and decision outcomes.
* **Recruiter AI Assistant**: Query-based hiring intelligence engine providing instantaneous statistical breakdowns and candidate recommendations.
* **Auditing & Reports**: On-demand downloadable 2-page PDF scorecards & interview dossiers via ReportLab and CSV exports for candidate rosters, jobs, and interviews.

---

## 🔐 Demo Credentials

Use any of the following pre-configured credentials to test different role interfaces:

| Role | Email | Password |
| :--- | :--- | :--- |
| **Admin** | `admin@ats.com` | `admin123` |
| **Recruiter** | `recruiter@ats.com` | `recruiter123` |
| **Candidate** | `candidate@ats.com` | `candidate123` |

---

## 🚀 Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Varun00752/NextGen-ATS.git
cd NextGen-ATS
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🌐 Deploying to Streamlit Community Cloud

This project is configured for 1-click deployment on [Streamlit Community Cloud](https://share.streamlit.io):

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **"New app"**.
3. Select repository: `Varun00752/NextGen-ATS`.
4. Set **Main file path**: `app.py`.
5. Click **"Deploy!"**.

---

## 🛠️ Tech Stack

* **Frontend & UI**: Streamlit, Plotly Express
* **NLP & Matching**: Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity), Regex
* **Document Parsing**: PyPDF2, pypdf, python-docx
* **Reporting**: ReportLab (PDF Generation), Pandas
* **Database**: SQLite3 (Embedded relational storage)

---

## 📁 Project Structure

```text
├── app.py                         # Core application and Streamlit multi-page interface
├── requirements.txt               # Production dependencies
├── README.md                      # Project documentation and guide
├── NextGen_ATS_Documentation.docx # Technical architecture document
├── .gitignore                     # Environment and cache ignore rules
├── uploads/                       # Persistent uploaded resume storage
├── exports/                       # Exported reports directory
└── ats_enterprise.db              # SQLite relational database
```
