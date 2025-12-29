import os
import re
import pickle
from typing import List, Dict
import faiss
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from Common.logger_utils import get_logger
from Common.device_utils import get_device
from Common.config import FAISS_FILE, META_FILE

logger = get_logger("Core.models.prediction")
device = get_device()

class PredictionModel(nn.Module):
    def __init__(self, input_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

def predict(embedding: torch.Tensor) -> torch.Tensor:
    mdl = PredictionModel().to(device)
    mdl.eval()
    with torch.no_grad():
        out = mdl(embedding.to(device))
        pred = torch.argmax(out, dim=1)
    return pred.cpu()

tokenizer = None
model = None
faiss_index = None
document_metadata: List[Dict] = []

def initialize_model() -> torch.device:
    global tokenizer, model
    try:
        logger.info(f"Initializing InLegalBERT on {device}")
        model_name = "law-ai/InLegalBERT"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=False).to(device)
        model.eval()
        return device
    except Exception as e:
        logger.error(f"InLegalBERT load failed: {e}", exc_info=True)
        return _initialize_fallback()

def _initialize_fallback():
    global tokenizer, model
    try:
        fb = torch.device("cpu")
        logger.info("Fallback: bert-base-uncased (CPU)")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased").to(fb)
        model.eval()
        return fb
    except Exception as e:
        logger.exception(f"Fallback load failed: {e}")
        return None

def load_index_files() -> bool:
    global faiss_index, document_metadata
    try:
        if os.path.exists(FAISS_FILE):
            faiss_index = faiss.read_index(str(FAISS_FILE))
        else:
            logger.error(f"FAISS missing: {FAISS_FILE}"); return False
        if os.path.exists(META_FILE):
            with open(META_FILE, "rb") as f:
                document_metadata = pickle.load(f)
        else:
            logger.error(f"Metadata missing: {META_FILE}"); return False
        logger.info(f"Index docs: {faiss_index.ntotal}; metadata: {len(document_metadata)}")
        return True
    except Exception as e:
        logger.exception(f"Index load error: {e}")
        return False

def _clean(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', text)
    text = re.sub(r'\b\d{1,2}(st|nd|rd|th)\s+\w+\s+\d{4}\b', '', text)
    for m in ['january','february','march','april','may','june','july','august','september','october','november','december']:
        text = re.sub(r'\b' + m + r'\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _embed(text: str) -> np.ndarray:
    global tokenizer, model
    if not text or text.strip() == "": return np.zeros(768, dtype=np.float32)
    if tokenizer is None or model is None:
        logger.error("Call initialize_model() first.")
        return np.zeros(768, dtype=np.float32)
    try:
        enc = tokenizer(_clean(text), padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
        return emb[0].astype(np.float32)
    except Exception as e:
        logger.exception(f"Embedding error: {e}")
        return np.zeros(768, dtype=np.float32)

def search_similar_cases_for_prediction(query_embedding: np.ndarray, top_k: int = 50, similarity_threshold: float = 0.15) -> List[Dict]:
    global faiss_index, document_metadata
    if faiss_index is None or document_metadata is None:
        logger.error("FAISS or metadata not loaded"); return []
    q = query_embedding.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    sims, idxs = faiss_index.search(q, top_k)
    results = []
    for i, (sim, idx) in enumerate(zip(sims[0], idxs[0])):
        if idx < len(document_metadata) and sim >= similarity_threshold:
            dm = document_metadata[idx]
            if dm.get('document_type') != 'case_judgment': continue
            js = _extract_js(dm)
            cp = dm.get('content_preview','')
            if not js and cp: js = cp
            results.append({
                "rank": i+1,
                "similarity_score": float(sim),
                "similarity_percentage": round(float(sim)*100,1),
                "outcome": dm.get('outcome','unknown'),
                "case_name": dm.get('metadata',{}).get('case_name','Unknown Case'),
                "year": dm.get('metadata',{}).get('year','Unknown'),
                "court": dm.get('metadata',{}).get('court','Unknown'),
                "content_preview": cp,
                "legal_concepts": dm.get('legal_concepts',[]),
                "judgement_summury": js,
                "has_judgement_summury": bool(js and len(js.strip())>0),
                "source_file": dm.get('source_file','')
            })
    logger.info(f"Similar case_judgment docs: {len(results)} (th={similarity_threshold})")
    return results

def _extract_js(doc_meta: Dict) -> str:
    if not isinstance(doc_meta, dict): return ""
    if doc_meta.get('judgement_summury'): return doc_meta['judgement_summury']
    md = doc_meta.get('metadata')
    if isinstance(md, dict) and md.get('judgement_summury'): return md['judgement_summury']
    for f in ['judgement','judgment','judgement_summary','judgment_summary']:
        if doc_meta.get(f): return doc_meta[f]
        if isinstance(md, dict) and md.get(f): return md[f]
    for f in ['content','cleaned_content','content_preview','text']:
        if doc_meta.get(f): return doc_meta[f]
    return ""

def _bucket(c: float) -> str:
    if c >= 80: return "Very High Confidence"
    if c >= 70: return "High Confidence"
    if c >= 60: return "Moderate Confidence"
    if c >= 50: return "Low Confidence"
    return "Very Low Confidence"

def predict_case_outcome(new_case_text: str, top_k: int = 50, similarity_threshold: float = 0.15) -> Dict:
    if model is None or faiss_index is None:
        return {'prediction':'unknown','confidence':0.0,'confidence_level':'System not initialized',
                'reason':'AI model or database not loaded','total_cases_found':0,
                'known_outcome_cases':0,'cases_with_judgement_summury':0}
    q = _embed(new_case_text)
    sim = search_similar_cases_for_prediction(q, top_k=top_k, similarity_threshold=similarity_threshold)

    known = [c for c in sim if c.get('outcome') in ['allowed','dismissed']]
    with_js = sum(1 for c in known if c.get('has_judgement_summury', False))

    if not known:
        return {'prediction':'unknown','confidence':0.0,'confidence_level':'No similar cases with known outcomes',
                'reason':'No similar cases with known outcomes found','total_cases_found':len(sim),
                'known_outcome_cases':0,'cases_with_judgement_summury':0}

    a_score = sum(c['similarity_percentage'] for c in known if c['outcome']=='allowed')
    d_score = sum(c['similarity_percentage'] for c in known if c['outcome']=='dismissed')
    a_cnt = sum(1 for c in known if c['outcome']=='allowed')
    d_cnt = sum(1 for c in known if c['outcome']=='dismissed')

    if a_score + d_score > 0:
        if a_score > d_score:
            pred = 'allowed'; conf = (a_score/(a_score+d_score))*100
        else:
            pred = 'dismissed'; conf = (d_score/(a_score+d_score))*100
    else:
        pred = 'unknown'; conf = 0.0

    return {
        'prediction': pred, 'confidence': round(conf,1), 'confidence_level': _bucket(conf),
        'total_cases_found': len(sim), 'known_outcome_cases': len(known),
        'cases_with_judgement_summury': with_js, 'allowed_cases': a_cnt, 'dismissed_cases': d_cnt,
        'top_similar_cases': known[:5]
    }
