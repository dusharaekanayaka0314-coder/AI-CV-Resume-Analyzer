import os
import re

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CV Match Coach",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

LLM_MODEL = "gemini-3.8-flash"          # change here if your model name differs
EMBED_MODEL = "models/gemini-embedding-001"


# --------------------------------------------------
# 2. Styling
# --------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
    --ink: #e8edf7;
    --muted: #9aa6bd;
    --teal: #2fc4a8;
    --teal-soft: rgba(47, 196, 168, 0.12);
    --sky: #eef4fb;
    --line: #2a3550;
    --good: #34d399;
    --warn: #fbbf24;
    --bad: #f87171;
}

html, body, [class*="css"], .stMarkdown, p, li, label, textarea, input {
    font-family: 'Source Sans 3', sans-serif;
}

h1, h2, h3, .hero-title, .tab-title {
    font-family: 'Bricolage Grotesque', sans-serif !important;
    color: var(--ink);
    letter-spacing: -0.01em;
}

/* app background */
.stApp {
    background: linear-gradient(180deg, #121b33 0%, #0b1020 340px);
}

#MainMenu, footer { visibility: hidden; }

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    max-width: 1100px;
}

/* ---------- readable text (the main fix) ---------- */
.stMarkdown p,
.stMarkdown li,
.stMarkdown strong,
.stMarkdown b,
.stMarkdown em,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: var(--ink) !important;
}
.stMarkdown li::marker { color: var(--teal); }
[data-testid="stText"], [data-testid="stText"] * { color: var(--ink) !important; }

/* ---------- hero ---------- */
.hero {
    padding: 8px 0 22px 0;
}
.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 10px 0;
    color: var(--ink);
}
.hero-title span.mark {
    background: linear-gradient(transparent 62%, rgba(47, 196, 168, 0.35) 62%);
    padding: 0 4px;
}
.hero-sub {
    font-size: 1.12rem;
    color: var(--muted);
    max-width: 620px;
    line-height: 1.55;
}

/* ---------- input cards ---------- */
.step-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.step-dot {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: var(--teal);
    color: #0b1020;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.9rem;
}
.step-label {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--ink);
}
.step-hint {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 0 0 10px 38px;
}

[data-testid="stFileUploader"] section {
    background: #141b2e;
    border: 2px dashed #3a4764;
    border-radius: 16px;
    padding: 22px;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--teal);
    background: var(--teal-soft);
}

.stTextArea textarea {
    background: #141b2e;
    border: 1.5px solid var(--line);
    border-radius: 14px;
    font-size: 1rem;
    color: var(--ink);
}
.stTextArea textarea:focus {
    border-color: var(--teal);
    box-shadow: 0 0 0 3px rgba(47, 196, 168, 0.25);
}

/* ---------- buttons ---------- */
.stButton > button[kind="primary"],
.stDownloadButton > button {
    background: var(--teal);
    color: #0b1020;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.4rem;
    font-weight: 600;
    font-size: 1.05rem;
    transition: background 0.15s ease, transform 0.1s ease;
}
.stButton > button[kind="primary"] p,
.stDownloadButton > button p {
    color: #0b1020 !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button:hover {
    background: #5fe0c8;
    color: #0b1020;
    border: none;
}
.stButton > button[kind="primary"]:active { transform: scale(0.99); }

/* ---------- score card ---------- */
.score-wrap {
    position: relative;
    width: 190px;
    height: 190px;
    margin: 0 auto;
}
.score-ring {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), #2a3550 0);
    -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 18px), #000 calc(100% - 17px));
            mask: radial-gradient(farthest-side, transparent calc(100% - 18px), #000 calc(100% - 17px));
}
.score-center {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.score-num {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 3.6rem;
    font-weight: 800;
    color: var(--ink);
    line-height: 1;
}
.score-of { color: var(--muted); font-size: 0.95rem; margin-top: 4px; }
.verdict {
    display: inline-block;
    margin-top: 14px;
    padding: 5px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
}

/* ---------- result containers ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--line);
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 14px;
    border-radius: 10px 10px 0 0;
    font-weight: 600;
}
.stTabs [data-baseweb="tab"] p {
    color: var(--muted) !important;
}
.stTabs [data-baseweb="tab"]:hover p {
    color: var(--ink) !important;
}
.stTabs [aria-selected="true"] p {
    color: var(--teal) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--teal) !important;
}

.tab-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin: 6px 0 10px 0;
}

.stMarkdown li { margin-bottom: 6px; line-height: 1.55; }
.stMarkdown p { line-height: 1.6; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #111829;
    border-right: 1px solid var(--line);
}
.side-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 700;
    font-size: 1.25rem;
    color: var(--ink);
    margin-bottom: 4px;
}
.side-note {
    color: var(--muted);
    font-size: 0.95rem;
    line-height: 1.5;
}
.tip {
    background: #172033;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 10px;
    font-size: 0.93rem;
    color: var(--ink);
}

/* ---------- dark-mode fixes ---------- */
[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stHeader"] *, [data-testid="stToolbar"] * {
    color: var(--ink) !important;
}
.stTextArea textarea {
    color: var(--ink) !important;
}
.stTextArea textarea::placeholder {
    color: #6f7c97 !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
    color: var(--muted) !important;
}
[data-testid="stFileUploader"] button {
    background: #1f2a44 !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
}
[data-testid="stFileUploader"] button:hover {
    border-color: var(--teal) !important;
}
[data-testid="stAlert"] * {
    color: inherit;
}

@media (max-width: 640px) {
    .hero-title { font-size: 2rem; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------
# 3. API key
# --------------------------------------------------

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY was not found. Add it to your .env file and restart the app.")
    st.stop()


# --------------------------------------------------
# 4. Prompt
# --------------------------------------------------

PROMPT = ChatPromptTemplate.from_template("""
You are an AI-powered CV and Job Match Analyzer.

Analyze the candidate's CV against the given Job Description.

IMPORTANT RULES:

- Only make claims based on the CV and Job Description.
- If a skill is required by the Job Description but is NOT found
  in the CV, say "Not found in CV".
- Do NOT say that the candidate does not know a skill simply because
  it is not mentioned in the CV.
- Give a realistic match score.
- Keep the analysis clear and professional.
- Do not invent experience.

JOB DESCRIPTION:

{job_description}


RELEVANT CV INFORMATION:

{cv_context}


Provide the analysis using exactly these sections:

## 1. Overall Match Score

Give a score out of 100 (write it like "72/100") and briefly explain why.

## 2. Matching Skills

List skills from the Job Description that are supported by the CV.

## 3. Strengths

List the candidate's strongest qualifications, technical skills,
projects and achievements.

## 4. Requirements Not Found in CV

List important Job Description requirements that were not found
in the CV.

Use "Not found in CV" rather than claiming the candidate does not
possess the skill.

## 5. Experience & Projects

Explain the most relevant projects or experience for this job.

## 6. Improvement Suggestions

Give practical suggestions to improve the CV for this specific job.

## 7. Recommended Skills

Suggest useful skills the candidate could learn next.

## 8. Interview Questions

Give 5 likely interview questions based on the candidate's CV
and the Job Description.
""")


# --------------------------------------------------
# 5. Helper functions
# --------------------------------------------------

TABS = [
    # (section number, tab label, title shown inside the tab)
    (2, "✅ Matching skills", "Skills that match the job"),
    (3, "💪 Strengths", "Where you shine"),
    (4, "🔍 Not found", "Requirements not found in your CV"),
    (5, "🧩 Experience", "Your most relevant experience"),
    (6, "✏️ Improve CV", "How to improve your CV for this job"),
    (7, "📚 Learn next", "Skills worth learning next"),
    (8, "🎤 Interview", "Questions you may be asked"),
]


def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def response_to_text(content) -> str:
    """Gemini can return a plain string or a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def split_sections(analysis: str) -> dict:
    """Turn '## 3. Strengths ...' blocks into {3: 'text'}."""
    sections = {}
    pattern = re.compile(r"^\s*#{2,3}\s*(\d+)\s*[\.\)]\s*.*$", re.MULTILINE)
    matches = list(pattern.finditer(analysis))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(analysis)
        sections[int(m.group(1))] = analysis[start:end].strip()
    return sections


def extract_score(text: str):
    if not text:
        return None
    m = re.search(r"(\d{1,3})\s*(?:/|out of)\s*100", text)
    if not m:
        m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        return max(0, min(100, int(m.group(1))))
    return None


def score_style(score: int):
    if score >= 75:
        return "#34d399", "rgba(52, 211, 153, 0.16)", "Strong match"
    if score >= 50:
        return "#fbbf24", "rgba(251, 191, 36, 0.16)", "Decent match, room to grow"
    return "#f87171", "rgba(248, 113, 113, 0.16)", "Needs work for this role"


def score_card(score: int, explanation: str):
    color, soft, verdict = score_style(score)
    with st.container(border=True):
        left, right = st.columns([1, 2], vertical_alignment="center")
        with left:
            st.markdown(
                f"""
                <div class="score-wrap">
                    <div class="score-ring" style="--p:{score}; --c:{color};"></div>
                    <div class="score-center">
                        <div class="score-num">{score}</div>
                        <div class="score-of">out of 100</div>
                    </div>
                </div>
                <div style="text-align:center;">
                    <span class="verdict" style="background:{soft}; color:{color};">{verdict}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown('<div class="tab-title">Your job match</div>', unsafe_allow_html=True)
            st.markdown(explanation)


def run_analysis(uploaded_file, job_description: str, status):
    status.update(label="Reading your CV...", state="running")
    cv_text = extract_pdf_text(uploaded_file)
    if not cv_text.strip():
        return None, None, "Could not read any text from this PDF. It may be a scanned image. Try another file."

    status.update(label="Finding the parts of your CV that matter...", state="running")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(cv_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GEMINI_API_KEY,
    )
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(job_description)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    status.update(label="Comparing your CV with the job...", state="running")
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)
    response = llm.invoke(
        PROMPT.format(job_description=job_description, cv_context=context)
    )
    analysis = response_to_text(response.content)
    return analysis, retrieved_docs, None


# --------------------------------------------------
# 6. Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown('<div class="side-title">💼 CV Match Coach</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-note">Check how well your CV fits a job before you apply.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown('<div class="side-title" style="font-size:1.05rem;">For better results</div>', unsafe_allow_html=True)
    st.markdown('<div class="tip">Use a text-based PDF, not a scanned photo.</div>', unsafe_allow_html=True)
    st.markdown('<div class="tip">Paste the full job post, including the requirements list.</div>', unsafe_allow_html=True)
    st.markdown('<div class="tip">Run it again for each job you apply to. Scores change per role.</div>', unsafe_allow_html=True)
    st.write("")
    st.caption("Your CV is processed only for this analysis and is not stored by the app.")


# --------------------------------------------------
# 7. Hero
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Does your CV fit <span class="mark">the job?</span></div>
        <div class="hero-sub">
            Upload your CV, paste the job description, and get an honest match score
            with clear next steps to improve your chances.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 8. Inputs
# --------------------------------------------------

col_cv, col_jd = st.columns(2, gap="large")

with col_cv:
    st.markdown(
        '<div class="step-head"><span class="step-dot">1</span>'
        '<span class="step-label">Upload your CV</span></div>'
        '<div class="step-hint">PDF only</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "CV file",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        st.success(f"Ready: {uploaded_file.name}")

with col_jd:
    st.markdown(
        '<div class="step-head"><span class="step-dot">2</span>'
        '<span class="step-label">Paste the job description</span></div>'
        '<div class="step-hint">Include the requirements section</div>',
        unsafe_allow_html=True,
    )
    job_description = st.text_area(
        "Job description",
        height=210,
        label_visibility="collapsed",
        placeholder=(
            "Example:\n"
            "We are looking for a Junior Software Engineer...\n"
            "Requirements:\n"
            "- Python\n"
            "- SQL\n"
            "- Git\n"
            "- REST APIs"
        ),
    )

st.write("")
btn_col, _ = st.columns([1, 2])
with btn_col:
    analyze_button = st.button("Analyze my CV", type="primary", use_container_width=True)


# --------------------------------------------------
# 9. Run analysis
# --------------------------------------------------

if analyze_button:
    if uploaded_file is None:
        st.warning("Upload your CV PDF first.")
    elif not job_description.strip():
        st.warning("Paste a job description to compare against.")
    else:
        with st.status("Starting...", expanded=True) as status:
            try:
                analysis, docs, error = run_analysis(uploaded_file, job_description, status)
            except Exception as e:
                analysis, docs, error = None, None, f"Something went wrong: {e}"

            if error:
                status.update(label="Analysis stopped", state="error")
                st.error(error)
            else:
                status.update(label="Analysis ready", state="complete", expanded=False)
                st.session_state["analysis"] = analysis
                st.session_state["docs"] = [d.page_content for d in docs]
                st.session_state["file_name"] = uploaded_file.name


# --------------------------------------------------
# 10. Show results (kept in session so downloads don't wipe them)
# --------------------------------------------------

if "analysis" in st.session_state:
    analysis = st.session_state["analysis"]
    docs = st.session_state.get("docs", [])
    sections = split_sections(analysis)

    st.write("")
    st.markdown("## Your results")

    if not sections:
        # fallback if the model ignored the format
        with st.container(border=True):
            st.markdown(analysis)
    else:
        score_text = sections.get(1, "")
        score = extract_score(score_text)

        if score is not None:
            explanation = re.sub(r"\*{0,2}\d{1,3}\s*(?:/|out of)\s*100\*{0,2}[\.:,]?", "", score_text, count=1).strip()
            score_card(score, explanation or score_text)
        else:
            with st.container(border=True):
                st.markdown("### Overall match")
                st.markdown(score_text)

        st.write("")

        available = [t for t in TABS if t[0] in sections]
        if available:
            tab_objs = st.tabs([t[1] for t in available])
            for tab, (num, _label, title) in zip(tab_objs, available):
                with tab:
                    st.markdown(f'<div class="tab-title">{title}</div>', unsafe_allow_html=True)
                    st.markdown(sections[num])

    st.write("")
    dl_col, _ = st.columns([1, 2])
    with dl_col:
        st.download_button(
            "Download analysis",
            data=analysis,
            file_name="cv-analysis.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with st.expander("See which parts of your CV were used"):
        st.caption("These sections of your CV were the closest match to the job description.")
        for i, chunk in enumerate(docs, 1):
            with st.container(border=True):
                st.markdown(f"**CV section {i}**")
                st.write(chunk)