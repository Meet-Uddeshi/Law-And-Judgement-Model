import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI, UploadFile, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import tempfile
import uvicorn
import pdfplumber
from Common.logger_utils import get_logger
from Core.inference.run_inference import analyze_text, retrieve_similar, force_retrain

file_path = os.path.abspath(__file__)
server_dir = os.path.dirname(file_path)
root_dir = os.path.dirname(server_dir)
os.chdir(root_dir)

API_SECRET = os.getenv("API_SECRET_KEY", "SuperSecureSecretKey123")

app = FastAPI(title="LegalAI Backend", version="1.0.0")
logger = get_logger("Server")

# CORS for local Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    # Allow health without key
    if request.url.path == "/health":
        return await call_next(request)
    key = request.headers.get("X-API-KEY")
    if key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await call_next(request)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train(force: bool = False):
    return force_retrain(force=force)

@app.post("/analyze")
async def analyze(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = None
):
    if file and file.filename.lower().endswith(".pdf"):
        # Use delete=False so we can close it and reopen it with pdfplumber
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                buf = await file.read()
                tmp.write(buf)
                tmp_path = tmp.name  # Save the path
            
            # The file is now CLOSED, so pdfplumber can access it
            with pdfplumber.open(tmp_path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n".join(pages)
        finally:
            # Clean up the temporary file manually
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    elif file and not text:
        raise HTTPException(status_code=400, detail="Only PDF files are supported or submit 'text'.")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted. Provide text or a valid PDF.")

    logger.info("Analyze request received")
    pred_block = analyze_text(text)
    if "error" in pred_block:
        raise HTTPException(status_code=500, detail=pred_block["error"])

    retrieval = retrieve_similar(text, top_k=10, threshold=0.15)
    return {
        **pred_block,
        "similar_cases": retrieval.get("results", []), # Changed from retrieval.get("similar_cases")
        "similar_cases_total": retrieval.get("total", 0),
        "input_text": text
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",    
        host="0.0.0.0",  
        port=8000,       
        reload=True,     
        log_level="info" 
    )