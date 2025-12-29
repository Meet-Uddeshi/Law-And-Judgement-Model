import faiss
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from Common.logger_utils import get_logger
from Common.device_utils import get_device
from Common.config import FAISS_FILE, META_FILE

logger = get_logger("Core.models.comparison")
device = get_device()

def load_index_and_metadata():
    index = faiss.read_index(str(FAISS_FILE))
    with open(META_FILE, 'rb') as f:
        metadata = pickle.load(f)
    logger.info(f"FAISS loaded with {index.ntotal} docs; metadata={len(metadata)}")
    return index, metadata

def load_inlegalbert_model(model_name: str = "law-ai/InLegalBERT"):
    logger.info(f"Loading encoder on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()
    return tok, mdl

def _clean(text: str) -> str:
    import re
    if not text:
        return ""
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', text)
    text = re.sub(r'\b\d{1,2}(st|nd|rd|th)\s+\w+\s+\d{4}\b', '', text)
    for m in ['january','february','march','april','may','june','july','august','september','october','november','december']:
        text = re.sub(r'\b' + m + r'\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s+[A-Z]+\s+\d+\b', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _encode(text: str, tokenizer, model) -> np.ndarray:
    if not text.strip():
        raise ValueError("Empty text for embedding")
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype('float32')

def search_similar_cases(query_text: str, top_k: int = 10, similarity_threshold: float = 0.15):
    index, metadata = load_index_and_metadata()
    tokenizer, model = load_inlegalbert_model()
    cleaned = _clean(query_text)
    qv = _encode(cleaned, tokenizer, model)
    faiss.normalize_L2(qv)
    sims, idxs = index.search(qv, top_k)
    results = []
    for i, (score, idx) in enumerate(zip(sims[0], idxs[0])):
        if idx < len(metadata) and score >= similarity_threshold:
            case_info = metadata[idx]
            content_source = "content"
            display_content = case_info.get("content","")
            js = case_info.get("judgement_summury","")
            if js:
                content_source = "judgement_summury"; display_content = js
            elif case_info.get("cleaned_content"):
                content_source = "cleaned_content"; display_content = case_info["cleaned_content"]
            case_name = case_info.get("metadata",{}).get("case_name","") or \
                        (case_info.get("content_preview","")[:100] + "..." if len(case_info.get("content_preview",""))>100 else case_info.get("content_preview",""))
            results.append({
                "rank": i+1,
                "case_name": case_name or "Unknown Case",
                "year": case_info.get("metadata",{}).get("year","N/A"),
                "document_type": case_info.get("document_type","unknown"),
                "summary": (display_content[:300] + "...") if len(display_content)>300 else display_content,
                "content_source": content_source,
                "has_judgement_summury": bool(js),
                "outcome": case_info.get("outcome","unknown"),
                "similarity_score": float(score),
                "similarity_percentage": round(float(score)*100,1),
                "legal_concepts": case_info.get("legal_concepts",[]),
                "court": case_info.get("metadata",{}).get("court","Unknown"),
                "content_preview": case_info.get("content_preview",""),
                "source_file": case_info.get("source_file","")
            })
    return results
