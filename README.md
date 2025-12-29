# Law-And-Judgement-Model-

An advanced machine learning system for **legal document analysis**, **case similarity search**, and **judgment outcome prediction**, powered by the **InLegalBERT** model and **FAISS-based semantic retrieval**.  

---

## **System Overview**

Law-And-Judgement-Model enables:
- Automatic **legal document ingestion**, **vectorization**, and **indexing**.
- Intelligent **similar case retrieval** via FAISS.
- **Outcome prediction** and judgment summary extraction.
- **Interactive visualization** through Streamlit frontend.

---

## **Tech Stack**

| Component          | Technology                           |
|--------------------|--------------------------------------|
| **Frontend**       | Streamlit                            |
| **Backend API**    | FastAPI                              |
| **Model Pipeline** | PyTorch + Transformers (InLegalBERT) |
| **Vector Store**   | FAISS (CPU)                          |
| **Orchestration**  | Docker + Docker Compose              |
| **Data Format**    | JSON + PDF (parsed via pdfplumber)   |

---

## **System Requirements**

### **Base System**
- Python 3.11+
- Docker & Docker Compose (for containerized setup)

### **System Dependencies (auto-installed in Docker)**
- `libgl1`, `poppler-utils`, `build-essential`, `dos2unix`  
*(for local setup: install these manually if not using Docker)*

---

## **Project Structure**

```
├── Common/
│   ├── device_utils.py # GPU/CPU detection utilities
│   ├── logger_utils.py # Logging utilities
│   └── requirements.txt
│
├── Core/
│   ├── pipeline/
│   │   └── pipeline.py # Full preprocessing → embedding → training pipeline
│   ├── inference/
│   │   └── run_inference.py # Unified inference + similarity + prediction
│   └── requirements.txt
│
├── Server/
│   ├── server.py # FastAPI backend with /health, /train, /analyze routes
│   ├── Dockerfile # Backend Docker build (auto FAISS detection)
│   └── requirements.txt
│
├── Frontend/
│   ├── app.py # Streamlit web UI for case analysis
│   └── requirements.txt
│
├── Data/
│   ├── Indian Laws & Constitution/
│   ├── Old Cases Dataset/
│   │   ├── 1950 to 1960/
│   │   ├── ...
│   │   └── 2021 to 2025/
│   ├── Json/
│   ├── Normalized Data/
│   └── Test/
│
├── Output/
│   ├── legal_index.faiss
│   └── legal_index_metadata.pkl
│
├── docker-compose.yml
└── README.md
```

---

## **Dataset Requirements**

### **Data Source**
- Supreme Court Dataset (Kaggle or equivalent)
- Store inside `Data/Old Cases Dataset/`  
  with decade-wise subfolders (automatically detected by pipeline).

### **Supported Formats**
- PDF: directly processed via backend.
- JSON: normalized preprocessed case data.

**Example JSON format:**
```json
{
  "case_number": "12345",
  "date": "2023-01-01",
  "petitioner": "John Doe",
  "respondent": "State",
  "judgment_text": "...",
  "judges": ["Judge A", "Judge B"],
  "judgment_outcome": "Allowed"
}
```

---
## **Running the System**

### **Backend (FastAPI)**
Run:
```
python -m uvicorn Server.server:app --reload --host 0.0.0.0 --port 8000
```

### **Frontend (Streamlit)**
Run:
```
python -m streamlit run Frontend/app.py
```

---
## **Output:**
- Backend (FastAPI docs): http://localhost:8000/docs
- Frontend (Streamlit UI): http://localhost:8501

---

## **Install dependencies:**

```
pip install -r requirements.txt
```

---

## **Pipeline Automation:**
Located in Core/pipeline/pipeline.py

### **Functions**
- Checks if `Output/legal_index.faiss and legal_index_metadata.pkl` exist.
- If missing or outdated → reprocesses data automatically.
- Steps: `preprocessing → feature_extraction → embedding_model → FAISS index.`
- Keeps same folder structure inside `Data/Old Cases Dataset/.`