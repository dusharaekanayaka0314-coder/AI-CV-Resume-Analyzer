# 💼 AI CV & Resume Analyzer

An AI-powered CV and Job Description Analyzer built using **RAG, FAISS, Google Gemini, LangChain, and Streamlit**.

The application analyzes a candidate's CV against a given Job Description and provides a detailed assessment of skills, strengths, missing requirements, improvement suggestions, recommended skills, and interview questions.

## 🚀 Live Demo

**Try the live application:**  
https://ai-cv-resume-analyzer-p2trrkjmfbixjt2yimd9cd.streamlit.app/

> The application is deployed using Streamlit Community Cloud.

## 📌 Project Overview

Finding out whether a CV matches a specific job description can be time-consuming.

This project uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from a CV and then uses **Google Gemini AI** to analyze the CV against the provided Job Description.

### How it works

**CV PDF + Job Description → Text Extraction → Text Chunking → Gemini Embeddings → FAISS Vector Store → Relevant CV Retrieval → Gemini AI Analysis → Results**

## ✨ Features

- 📄 Upload a CV as a PDF
- 📝 Enter a Job Description
- 🔍 Extract text from the CV
- ✂️ Split CV text into smaller chunks
- 🧠 Generate embeddings using Google Gemini
- 📚 Store and retrieve CV information using FAISS
- 🤖 Analyze CV and Job Description using Gemini AI
- 📊 Generate an overall CV match score
- ✅ Identify matching skills
- 💪 Identify candidate strengths
- 🔎 Identify job requirements not found in the CV
- 💼 Analyze relevant experience and projects
- 📝 Provide CV improvement suggestions
- 🎯 Recommend useful skills to learn
- 🎤 Generate potential interview questions

## 🧠 RAG Pipeline

```text
CV PDF
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Gemini Embeddings
   ↓
FAISS Vector Store
   ↓
Relevant CV Information Retrieval
   ↓
Gemini AI
   ↓
CV & Job Description Analysis
```

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **Google Gemini AI**
- **LangChain**
- **FAISS**
- **PyPDF**

## 🔬 Prototype & Testing

Google Colab was used during the development process to prototype and test the RAG pipeline.

The main components tested in Colab included:

- PDF text extraction
- CV text chunking
- Gemini embeddings
- FAISS vector store
- Relevant document retrieval
- Gemini-based CV and Job Description analysis

After validating the core RAG workflow in Google Colab, the solution was implemented as a complete **Streamlit web application**.

## 💻 Local Setup

### 1. Clone the repository

```bash
git clone YOUR-GITHUB-REPOSITORY-URL
```

### 2. Open the project folder

```bash
cd AI-CV-Resume-Analyzer
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

Create a `.env` file in the project folder:

```env
GEMINI_API_KEY=your_api_key_here
```

### 5. Run the application

```bash
streamlit run app.py
```

## 🔐 Security

The Gemini API key is stored using environment variables and Streamlit Cloud Secrets.

The `.env` file is excluded from the GitHub repository using `.gitignore`.

**Never upload or expose your API key publicly.**

## 📂 Project Structure

```text
AI-CV-Resume-Analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
└── .env
```

> `.env` and `.venv` are kept locally and are not uploaded to GitHub.

## 🎯 Future Improvements

- Support for multiple CV formats
- Resume section-wise scoring
- More advanced job recommendation
- ATS-style analysis
- Downloadable analysis reports
- Multiple Job Description comparison
- Improved interview preparation features

## 👩‍💻 Project

Built as an AI/RAG learning and portfolio project to explore how **Retrieval-Augmented Generation can be applied to real-world recruitment and career-related tasks**.
