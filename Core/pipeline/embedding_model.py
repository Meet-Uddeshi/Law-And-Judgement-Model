import os, json, pickle, torch, faiss, logging
from typing import List, Dict, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from Common.logger_utils import get_logger
from Common.device_utils import get_device
from Common.config import FAISS_FILE, META_FILE

logger = get_logger("Core.pipeline.embedding_model")
device = get_device()

# Global instances for reuse
tokenizer = None
model = None
faiss_index = None
document_metadata: List[Dict] = []

def initialize_inlegalbert(model_name: str = "law-ai/InLegalBERT"):
    """
    Initializes and caches the InLegalBERT model and tokenizer on the detected device.
    """
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info(f"[MODEL] Loading {model_name} onto {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
    return device

def generate_single_embedding(text: str) -> np.ndarray:
    """
    Converts a single string of text into a 768-dimension vector.
    This is used by the Inference engine and Server.
    """
    global tokenizer, model
    # Ensure model is loaded
    if tokenizer is None or model is None:
        initialize_inlegalbert()
    
    # Preprocess text (BERT limit is 512 tokens)
    inputs = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean Pooling strategy: Average the last hidden states
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    # Return as flat float32 array for FAISS compatibility
    return embeddings.flatten().astype('float32')

def load_normalized_data(filepath: str) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} normalized documents from {filepath}")
    return data

def _content_for_embed(doc: Dict) -> str:
    for field in ["cleaned_content","content"]:
        v = doc.get(field,"")
        if v and v.strip():
            return v
    md = doc.get("metadata",{})
    return md.get("judgement_summury","")

def batch_generate_embeddings(documents: List[Dict], batch_size: int = 8) -> Tuple[np.ndarray, List[Dict]]:
    global tokenizer, model
    if tokenizer is None or model is None:
        initialize_inlegalbert()
        
    all_embeddings = []
    metadata_list = []
    logger.info(f"Generating embeddings for {len(documents)} docs | batch_size={batch_size}")
    
    for i in tqdm(range(0, len(documents), batch_size)):
        batch_docs = documents[i:i+batch_size]
        texts = [_content_for_embed(doc)[:10000] for doc in batch_docs]
        try:
            enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**enc)
                batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            all_embeddings.extend(batch_emb)
            for doc in batch_docs:
                content = doc.get("content", "")
                first_line = content.split('\n')[0][:100] if content else "Unknown Case"
                m = {
                    "document_id": doc.get("document_id"),
                    "case_name": doc.get("metadata", {}).get("case_name") or first_line,
                    "source_file": doc.get("source_file"),
                    "court": doc.get("metadata", {}).get("court", "Unknown"),
                    "year": doc.get("metadata", {}).get("year", "N/A"),
                    "content_preview": content[:300],
                    "legal_concepts": doc.get("legal_concepts", [])
                }
                metadata_list.append(m)
        except Exception as e:
            logger.exception(f"Batch {i//batch_size} failed: {e}")
            
    return np.array(all_embeddings).astype('float32'), metadata_list

def extract_case_outcome(text: str) -> str:
    if not text: return "unknown"
    tl = text.lower()
    allowed = ['appeal allowed','petition allowed','allowed.','writ petition allowed','is allowed']
    dismissed = ['appeal dismissed','petition dismissed','dismissed.','is dismissed']
    
    # Check the last few paragraphs where the verdict usually is
    paras = tl.split('\n')
    last_segments = paras[-5:] if len(paras)>5 else paras
    for p in last_segments:
        if any(ph in p for ph in allowed): return "allowed"
        if any(ph in p for ph in dismissed): return "dismissed"
    return "unknown"

def add_outcomes_to_metadata(metadata_list: List[Dict], normalized_data: List[Dict]) -> List[Dict]:
    logger.info("Extracting outcomes for metadata...")
    for dm in metadata_list:
        content = dm.get("judgement_summury", "") or dm.get("content_preview", "")
        dm["outcome"] = extract_case_outcome(content)
    return metadata_list

def build_faiss_index(embeddings: np.ndarray, metadata: List[Dict]):
    global faiss_index, document_metadata
    logger.info("Building FAISS index (Inner Product for Similarity)")
    
    # Normalize vectors for Cosine Similarity (using IndexFlatIP)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings.astype(np.float32))
    document_metadata = metadata
    logger.info(f"Index created with {faiss_index.ntotal} vectors.")

def save_faiss_index():
    if faiss_index is None: raise ValueError("Index is empty.")
    os.makedirs(FAISS_FILE.parent, exist_ok=True)
    faiss.write_index(faiss_index, str(FAISS_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(document_metadata, f)
    logger.info("Index and Metadata saved successfully.")

def analyze_embedding_statistics(emb: np.ndarray):
    logger.info(f"Stats: shape={emb.shape} | mean={np.mean(emb):.4f}")