import re
import logging
from pathlib import Path
from typing import List
import torch 
from Common.logger_utils import get_logger
from Common.config import DATA_DIR, JSON_DIR, NORMALIZED_DIR, FAISS_FILE, META_FILE
from Core.pipeline.preprocessing import run_pdf_folder_to_json
from Core.pipeline.feature_extraction import normalize_all_files, save_normalized
from Core.pipeline.embedding_model import (
    initialize_inlegalbert, load_normalized_data, batch_generate_embeddings,
    add_outcomes_to_metadata, analyze_embedding_statistics, build_faiss_index, 
    save_faiss_index, generate_single_embedding # Ensure this is imported
)

logger = get_logger("Core.pipeline.pipeline")

def get_text_embedding(text: str, year: str = "N/A", filename: str = "N/A"):
    """
    FIXED: This function must return a NUMPY VECTOR for FAISS.
    We use the model initialized in embedding_model.py.
    """
    # 1. Ensure the model is loaded
    initialize_inlegalbert()
    
    # 2. Call the actual mathematical embedding logic
    # This usually comes from your embedding_model.py
    vector = generate_single_embedding(text)
    return vector

def extract_metadata_from_text(text: str, year: str, filename: str):
    """
    REFACTORED: Moved your regex logic here. 
    This is for metadata, not embeddings.
    """
    md = {
        'year': year, 'filename': filename, 'case_summary': '',
        'judgement': '', 'parties_involved': '', 'court': ''
    }
    # ... (Your existing regex logic for court and parties remains here) ...
    lines = text.split('\n')
    if len(lines) > 10:
        md['case_summary'] = ' '.join(lines[:10])
        md['judgement'] = ' '.join(lines[-20:])
    return md

def _default_json_inputs() -> List[str]:
    files = [
        JSON_DIR / "constitution_qa.json",
        JSON_DIR / "crpc_qa.json",
        JSON_DIR / "ipc_qa.json",
        JSON_DIR / "case_1950 to 1960.json",
        JSON_DIR / "case_1961 to 1970.json",
        JSON_DIR / "case_1971 to 1980.json",
        JSON_DIR / "case_1981 to 1990.json",
        JSON_DIR / "case_1991 to 2000.json",
        JSON_DIR / "case_2001 to 2010.json",
        JSON_DIR / "case_2011 to 2020.json",
        JSON_DIR / "case_2021 to 2025.json",
    ]
    return [str(p) for p in files if p.exists()]

def build_index_from_data():
    logger.info("=== PIPELINE START ===")
    from pathlib import Path
    faiss_path = Path("Output/legal_index.faiss")
    pkl_path = Path("Output/legal_index_metadata.pkl")

    if faiss_path.exists() and pkl_path.exists():
        logging.info("Output already exist → skipping pipeline rebuild.")
        return
    # Generate a catch-all JSON from any PDFs present (non-destructive)
    if (DATA_DIR / "Old Cases Dataset").exists():
        out_json = JSON_DIR / "case_latest.json"
        run_pdf_folder_to_json(DATA_DIR / "Old Cases Dataset", out_json)

    # Gather JSON inputs
    json_inputs = _default_json_inputs()
    latest_json = JSON_DIR / "case_latest.json"
    if latest_json.exists():
        json_inputs.append(str(latest_json))

    if not json_inputs:
        logger.warning("No JSON files found under Data/Json. Proceeding with empty set.")

    # Normalize
    docs = normalize_all_files(json_inputs)
    normalized_json_path = NORMALIZED_DIR / "normalized_all_json_files.json"
    save_normalized(docs, out_json=str(normalized_json_path))

    # Embeddings + index
    initialize_inlegalbert()
    documents = load_normalized_data(str(normalized_json_path))
    embeddings, metadata_list = batch_generate_embeddings(documents, batch_size=16)
    metadata_list = add_outcomes_to_metadata(metadata_list, documents)
    analyze_embedding_statistics(embeddings)

    build_faiss_index(embeddings, metadata_list)
    save_faiss_index()
    logger.info(f"Artifacts saved: {FAISS_FILE.name}, {META_FILE.name}")
    logger.info("=== PIPELINE END ===")
