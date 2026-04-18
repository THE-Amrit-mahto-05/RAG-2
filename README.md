# 🎓 Edulevel — AI-Powered RAG Textbook Tutor

> Upload any textbook chapter as a PDF. Get instant, context-grounded AI explanations with matching textbook figures.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **PDF Intelligence** | Upload any textbook chapter — text, diagrams and figures are all extracted automatically |
| 🧠 **RAG Pipeline** | Hybrid semantic + keyword retrieval with re-ranking for highly accurate, grounded answers |
| 🖼️ **Figure Carousel** | Up to 3 contextually relevant textbook figures shown per answer, matched by page context |
| 📚 **Chapter Navigator** | Auto-generated Table of Contents from the PDF for one-click topic exploration |
| 🛡️ **Hallucination Guard** | Every answer is validated against source chunks — no making things up |
| ⚡ **Fast Startup** | Lazy-loaded embeddings, CPU-only PyTorch — runs on free-tier hosting without timeouts |
| 🔑 **Multi-Key Rotation** | Automatic Groq API key rotation to handle rate limits gracefully |
| 💾 **Semantic Cache** | Repeated questions return instantly without hitting the LLM again |

---

## 🏗️ Architecture

```
Student uploads PDF
        ↓
  [PDF Processor]  →  Text chunks (with page metadata)
  [Image Processor] →  Extracted figures + captions
        ↓
  [Embedding Service]  (sentence-transformers / all-MiniLM-L6-v2)
        ↓
  [FAISS Vector Store]  (per-topic index on disk)
        ↓
Student asks question
        ↓
  [Retrieval Engine]  →  Top-K chunks (semantic + keyword + page boost)
  [Image Matcher]    →  Top-3 figures (page-context boosted)
        ↓
  [LLM Service]  (Groq / llama-3.1-8b-instant)
        ↓
  Grounded Answer + Textbook Figures → Student UI
```

---

## 🛠️ Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com) — API framework
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io) — PDF text & image extraction
- [sentence-transformers](https://sbert.net) — Local embeddings (no API key needed)
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Groq](https://console.groq.com) — Ultra-fast LLM inference

**Frontend**
- [React 18](https://react.dev) + [TypeScript](https://typescriptlang.org)
- [Vite](https://vitejs.dev) — Build tool
- [Framer Motion](https://framer.com/motion) — Animations
- [Axios](https://axios-http.com) — HTTP client

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.9+
- Node.js 18+ & npm
- A free [Groq API key](https://console.groq.com)

### 1. Clone & Configure
```bash
git clone https://github.com/THE-Amrit-mahto-05/RAG-2.git
cd RAG-2

# Create your environment file
cp .env.example .env
# Fill in your GROQ_API_KEY in .env
```

### 2. Start the Backend
```bash
# From the project root
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt

uvicorn backend.main:app --reload
# → Runs at http://localhost:8000
```

### 3. Start the Frontend
```bash
cd frontend
npm install
npm run dev
# → Runs at http://localhost:3000
```

### 4. Use It
1. Open [http://localhost:3000](http://localhost:3000)
2. Upload any NCERT or textbook chapter PDF
3. Click any topic in the **Chapter Navigator**
4. Ask follow-up questions in the **AI Tutor Chat**

---

## ☁️ Deploy to Render (Production)

### Backend (Web Service)
| Setting | Value |
|---|---|
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python main.py` |
| Environment | Set all keys from `.env.example` |

### Frontend
Deploy to [Vercel](https://vercel.com) or [Netlify](https://netlify.com):
```bash
cd frontend
npm run build
# deploy the 'dist' folder
```
Set `VITE_API_URL` to your Render backend URL.

---

## 🔑 Environment Variables

See [`.env.example`](.env.example) for a full list. The minimum required:

```env
GROQ_API_KEY=your_key_here
EMBEDDING_PROVIDER=local
```

---

## 📄 License
MIT License — Built for the EduLevel Product Engineering Intern Assignment.
