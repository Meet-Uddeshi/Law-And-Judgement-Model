import os
import json
import re
from collections import defaultdict
from pathlib import Path
import pdfplumber
import torch

from Common.logger_utils import get_logger
from Common.device_utils import get_device

logger = get_logger("Core.pipeline.preprocessing")
device = get_device()

def preprocess(data):
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    processed = data_tensor * 0.5
    return processed.cpu()

def analyze_pdf_structure(pdf_path: str):
    common_footers = set()
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        footer_candidates = defaultdict(int)
        for page in pdf.pages[:min(5, total_pages)]:
            footer_bbox = (0, page.height * 0.85, page.width, page.height)
            footer_area = page.within_bbox(footer_bbox)
            footer_text = footer_area.extract_text()
            if footer_text:
                for line in footer_text.strip().split("\n"):
                    clean_line = line.strip()
                    if clean_line and len(clean_line) < 100:
                        footer_candidates[clean_line] += 1
        for footer, count in footer_candidates.items():
            if count >= 2:
                common_footers.add(footer)
    return common_footers

def clean_extracted_text(text: str, common_footers):
    if not text:
        return ""
    for footer in common_footers:
        text = text.replace(footer, "")
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_case_metadata(text: str, year: str, filename: str):
    md = {
        'year': year, 'filename': filename, 'case_summary': '',
        'judgement': '', 'parties_involved': '', 'court': ''
    }
    court_patterns = [
        r'IN THE (SUPREME COURT|HIGH COURT) OF ([A-Z\s]+)',
        r'BEFORE THE (SUPREME COURT|HIGH COURT) OF ([A-Z\s]+)',
        r'([A-Z\s]+HIGH COURT|[A-Z\s]+SUPREME COURT)'
    ]
    for pattern in court_patterns:
        m = re.search(pattern, text[:2000])
        if m:
            md['court'] = m.group(0)
            break

    parties_pattern = r'([A-Z][A-Za-z\s,]+)\s+(?:v|vs|versus|v\.)\s+([A-Z][A-Za-z\s,]+)'
    pm = re.search(parties_pattern, text[:3000])
    if pm:
        md['parties_involved'] = f"{pm.group(1)} vs {pm.group(2)}"

    lines = text.split('\n')
    if len(lines) > 10:
        md['case_summary'] = ' '.join(lines[:10])
        md['judgement'] = ' '.join(lines[-20:])
    return md

def process_single_pdf(pdf_path: str, year: str, filename: str):
    try:
        logger.info(f"Processing PDF: {filename}")
        common_footers = analyze_pdf_structure(pdf_path)
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                content_bbox = (0, 0, page.width, page.height * 0.85)
                content_area = page.within_bbox(content_bbox)
                page_text = content_area.extract_text() or ""
                cleaned_text = clean_extracted_text(page_text, common_footers)
                full_text += cleaned_text + "\n\n"
        meta = extract_case_metadata(full_text, year, filename)
        meta['cleaned_text'] = full_text[:50000]
        logger.info(f"Completed: {filename}")
        return meta
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}", exc_info=True)
        return {
            'year': year, 'filename': filename, 'error': str(e),
            'case_summary': '', 'judgement': '', 'parties_involved': '', 'court': ''
        }

def run_pdf_folder_to_json(base_directory: Path, output_json_path: Path):
    logger.info(f"Scanning PDFs under: {base_directory}")
    all_cases = []
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                year_match = re.search(r'\b(19|20)\d{2}\b', folder_name) or re.search(r'\b(19|20)\d{2}\b', file)
                year = year_match.group(0) if year_match else "Unknown"
                all_cases.append(process_single_pdf(pdf_path, year, file))
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(all_cases)} records → {output_json_path}")
    return all_cases
