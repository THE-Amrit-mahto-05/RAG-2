# Edulevel: AI-Powered RAG Textbook Tutor

Edulevel is a premium, full-stack AI application designed to transform static textbook PDFs into interactive learning experiences. Using a state-of-the-art **Retrieval-Augmented Generation (RAG)** architecture, Edulevel allows students to upload chapters, ask complex questions, and receive context-grounded answers accompanied by relevant textbook diagrams.

![Edulevel UI](https://github.com/THE-Amrit-mahto-05/RAG-2/raw/main/screenshot.png) *(Placeholder)*

## 🚀 Features

- **Semantic PDF Processing**: Intelligent chunking and text extraction with page-level metadata.
- **Advanced RAG Pipeline**: Hybrid semantic and keyword-based retrieval with re-ranking for maximum accuracy.
- **Visual Learning**: Automated textbook figure extraction and semantic image matching to explain concepts visually.
- **Multimodal AI**: Supports OpenAI, Google Gemini, Groq, and local Ollama for flexible, high-speed inference.
- **Hallucination Guard**: Built-in groundedness validation ensures every answer is backed by the source material.
- **Premium UI**: Modern dark-mode React interface with glassmorphism, smooth animations, and interactive citations.

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.9+)
- **Frontend**: React + Vite + TypeScript + TailwindCSS
- **AI/ML**: 
  - `sentence-transformers` (Local Embeddings)
  - `faiss-cpu` (Vector Database)
  - `PyMuPDF` (PDF Intelligence)
  - `Groq / OpenAI / Gemini` (LLM Inference)
- **Animations**: Framer Motion
- **Icons**: Lucide React

## 📦 Setup & Installation

### 1. Prerequisites
- Python 3.9+
- Node.js & NPM
- API Keys for Groq, OpenAI, or Gemini

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
uvicorn main:app --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🔒 Security & Privacy
- Local FAISS index for document privacy.
- Supports local LLMs via Ollama for zero-data-leakage environments.
- Comprehensive `.gitignore` for secret management.

## 📄 License
MIT License - Developed as part of the Edulevel AI Initiative.
