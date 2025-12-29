import logging
import pickle
import faiss
import numpy as np
import torch
from pathlib import Path
from Common.config import FAISS_FILE, META_FILE
from Common.device_utils import get_device
from Core.pipeline.pipeline import build_index_from_data, get_text_embedding

device = get_device()
logger = logging.getLogger(__name__)

def ensure_artifacts():
    faiss_path = FAISS_FILE
    pkl_path = META_FILE
    if not faiss_path.exists() or not pkl_path.exists():
        logger.warning("Artifacts missing. Rebuilding index...")
        build_index_from_data()
    return faiss_path, pkl_path

def analyze_text(input_text: str, top_k: int = 10):
    """
    Logic Flow: Load -> Embed -> Search -> Format
    """
    try:
        # 1. Ensure files exist and load them
        faiss_path, pkl_path = ensure_artifacts()
        index = faiss.read_index(str(faiss_path))
        with open(pkl_path, "rb") as f:
            metadata = pickle.load(f)

        # 2. GENERATE THE VECTOR (This was the missing line causing the error)
        # Using the function from pipeline.py as imported
        query_vec = get_text_embedding(input_text)

        # 3. CONVERT TO FAISS FORMAT (Must be 2D float32)
        if isinstance(query_vec, torch.Tensor):
            query_vec = query_vec.detach().cpu().numpy()
        
        # Ensure it is a 2D array: (1, 768)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        
        query_vec = query_vec.astype('float32')

        # 4. SEARCH
        # Now 'query_vec' is defined and safe to use
        distances, indices = index.search(query_vec, k=top_k)

        # 5. FORMAT RESULTS
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(metadata):
                case_info = metadata[idx].copy()
                
                # 1. HEURISTIC CASE NAME EXTRACTION
                # Check for 'case_name', then fallback to 'source_file', 
                # then try to parse the first line of 'content_preview'
                raw_name = case_info.get("case_name")
                if not raw_name:
                    raw_name = case_info.get("source_file", "Unknown Case")
                
                # Clean up filenames (e.g., '1993_tax_case.pdf' -> '1993 Tax Case')
                clean_name = str(raw_name).replace(".pdf", "").replace("_", " ").title()
                
                # 2. OVERRIDE WITH PREVIEW IF IT LOOKS LIKE A LEGAL TITLE
                # Most legal docs start with 'Petitioner vs Respondent'
                preview = case_info.get("content_preview", "")
                if " vs " in preview[:100] or " VS " in preview[:100]:
                    # Extract the part before the date or citations
                    clean_name = preview.split(" on ")[0].split("\n")[0].strip()

                case_info["case_name"] = clean_name
                
                # 3. CALCULATE SIMILARITY
                dist = float(distances[0][rank])
                # If using IndexFlatIP with normalized vectors, dist is Cosine Similarity
                case_info["similarity_percentage"] = round(dist * 100, 2) if dist <= 1 else round((1/(1+dist))*100, 2)

                results.append(case_info)
        # 6. OUTPUT FOR SERVER.PY
        return {
            "prediction": results[0].get("outcome", "unknown") if results else "unknown",
            "confidence": results[0].get("similarity_percentage", 0) if results else 0,
            "confidence_level": "High" if (results and results[0].get("similarity_percentage", 0) > 80) else "Medium",
            "results": results,
            "total": len(results)
        }

    except Exception as e:
        logger.error(f"Inference Error: {e}")
        return {"error": str(e), "results": [], "total": 0}

def retrieve_similar(text: str, top_k: int = 10, threshold: float = 0.15):
    """Alias for compatibility with server.py"""
    return analyze_text(text, top_k=top_k)

def force_retrain(force: bool = False):
    """Admin function to rebuild artifacts"""
    build_index_from_data()
    return {"status": "success", "message": "Index rebuilt."}