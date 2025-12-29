import json
import os
import re
import uuid
from typing import Dict, List
import pandas as pd
import torch

from Common.logger_utils import get_logger
from Common.device_utils import get_device

logger = get_logger("Core.pipeline.feature_extraction")
device = get_device()

def extract_features(preprocessed_data: torch.Tensor) -> torch.Tensor:
    try:
        preprocessed_data = preprocessed_data.to(device)
        features = preprocessed_data.mean(dim=0, keepdim=True)
        return features.cpu()
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise

def clean_legal_text(text: str) -> str:
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

def extract_legal_concepts(text: str) -> List[str]:
    if not text:
        return []
    tl = text.lower()
    legal_terms = [
        'murder','assault','robbery','theft','fraud','bail','arrest','investigation',
        'evidence','witness','trial','conviction','sentence','offense','crime','criminal',
        'penal','code','contract','agreement','breach','damages','compensation','tort',
        'negligence','property','ownership','possession','lease','rent','landlord','tenant',
        'fundamental','rights','liberty','equality','freedom','constitution','article',
        'amendment','writ','petition','judicial','review','worker','employee','employer',
        'wages','salary','termination','factory','industrial','dispute','strike','union',
        'labor','appeal','court','judge','judgment','order','decree','verdict','plaintiff',
        'defendant','accused','respondent','petitioner','jurisdiction','procedure',
        'substantive','law'
    ]
    return [t for t in legal_terms if t in tl]

def _pick_content_field(case_data: Dict) -> str:
    for field in ['cleaned_text','judgement','judgement_summury','judgement_summary','case_summary','summary','case_text','text']:
        if case_data.get(field):
            return field
    return "unknown"

def _case_name(case_data: Dict) -> str:
    if case_data.get("filename"):
        n = case_data["filename"].replace(".PDF","").replace(".pdf","")
        n = ' '.join([w for w in n.split('_') if not w.isdigit()])
        return n
    for k in ["case_summary", "judgement_summury", "judgement_summary"]:
        if case_data.get(k):
            v = case_data[k]
            return v.split(" vs ")[0] if " vs " in v else v[:100]
    return ""

def _refs(text: str) -> str:
    patterns = [r'article\s+(\d+[A-Z]*)', r'section\s+(\d+[A-Z]*)', r'schedule\s+(\w+)', r'act\s+of\s+(\d{4})']
    refs = []
    for p in patterns:
        refs.extend(re.findall(p, text or "", re.IGNORECASE))
    return ", ".join(refs) if refs else ""

def create_qa_document(qa_data: Dict, law_type: str, file_path: str) -> Dict:
    q, a = qa_data.get("question",""), qa_data.get("answer","")
    content = f"Question: {q}\nAnswer: {a}"
    return {
        "document_id": str(uuid.uuid4()),
        "document_type": "law_qa",
        "content": content,
        "cleaned_content": clean_legal_text(content),
        "legal_concepts": extract_legal_concepts(content),
        "metadata": {
            "law_type": law_type,
            "article_section": _refs(q),
            "question_length": len(q),
            "answer_length": len(a)
        },
        "source_file": os.path.basename(file_path)
    }

def create_case_document(case_data: Dict, file_path: str) -> Dict:
    content_field = _pick_content_field(case_data)
    content = case_data.get(content_field, "")
    return {
        "document_id": str(uuid.uuid4()),
        "document_type": "case_judgment",
        "content": content,
        "cleaned_content": clean_legal_text(content),
        "legal_concepts": extract_legal_concepts(content),
        "metadata": {
            "year": case_data.get("year",""),
            "case_name": _case_name(case_data),
            "court": case_data.get("court",""),
            "parties": case_data.get("parties_involved", case_data.get("parties","")),
            "filename": case_data.get("filename",""),
            "content_length": len(content),
            "has_judgement_summary": bool(case_data.get("judgement_summury") or case_data.get("judgement_summary")),
            "content_source": content_field
        },
        "source_file": os.path.basename(file_path)
    }

def _process_qa_file(file_path: str, data: Dict) -> List[Dict]:
    out = []
    law_type = ("constitution" if "constitution" in file_path else
                "crpc" if "crpc" in file_path else
                "ipc" if "ipc" in file_path else "unknown_law")
    if isinstance(data, dict):
        if "question" in data and "answer" in data:
            out.append(create_qa_document(data, law_type, file_path))
        else:
            for v in data.values():
                if isinstance(v, dict) and "question" in v and "answer" in v:
                    out.append(create_qa_document(v, law_type, file_path))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                out.append(create_qa_document(item, law_type, file_path))
    return out

def _process_case_file(file_path: str, data: Dict) -> List[Dict]:
    out = []
    keys = ['year','case_summary','judgement','cleaned_text','judgement_summury','judgement_summary',
            'summary','case_text','text','parties_involved','court']
    if isinstance(data, dict):
        if any(k in data for k in keys):
            out.append(create_case_document(data, file_path))
        else:
            for v in data.values():
                if isinstance(v, dict) and any(k in v for k in keys):
                    out.append(create_case_document(v, file_path))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and any(k in item for k in keys):
                out.append(create_case_document(item, file_path))
    return out

def normalize_all_files(json_files: List[str]) -> List[Dict]:
    docs = []
    for file_path in json_files:
        try:
            logger.info(f"Processing {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if any(t in file_path for t in ["constitution","crpc","ipc"]):
                part = _process_qa_file(file_path, data)
            elif "case" in file_path.lower():
                part = _process_case_file(file_path, data)
            else:
                logger.warning(f"Unknown file type {file_path}")
                part = []
            logger.info(f"Extracted {len(part)} documents")
            docs.extend(part)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
    return docs

def save_normalized(docs: List[Dict], out_json: str = "./Data/Normalized Data/normalized_all_json_files.json",
                    out_csv: str = "./Data/Normalized Data/normalized_all_json_files.csv"):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        import json as _json
        _json.dump(docs, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(docs)} documents → {out_json}")

    df = pd.DataFrame(docs)
    meta = pd.json_normalize(df['metadata'])
    df2 = pd.concat([df.drop(['metadata'], axis=1), meta], axis=1)
    df2.to_csv(out_csv, index=False, encoding='utf-8')
    logger.info(f"Saved CSV → {out_csv}")
    return out_json
