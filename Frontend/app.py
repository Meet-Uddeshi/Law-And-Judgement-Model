import os
import sys
import glob
import tempfile
import base64
import zipfile
import logging
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple, List, Dict

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors

import streamlit as st
import requests

# ---------------------------
# Path & Environment Setup
# ---------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
os.chdir(root_dir)

# ---------------------------
# Configuration & Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frontend")

# Backend URL & Auth (override via env when running with Docker)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "SuperSecureSecretKey123")

PDF_BASE_PATH_DEFAULT = "./Data/Old Cases Dataset"

# Headers for backend (do NOT set content-type when sending multipart)
AUTH_HEADER = {"X-API-KEY": API_SECRET_KEY}

# ---------------------------
# Streamlit Setup & CSS
# ---------------------------
st.set_page_config(
    page_title="Law & Judgement Model",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
.case-card { background-color: #000000; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border-left: 5px solid #1f77b4; }
.similarity-high { color: #00cc96; font-weight: bold; }
.similarity-medium { color: #ffa15c; font-weight: bold; }
.similarity-low { color: #ef553b; font-weight: bold; }
.outcome-allowed { color: #00cc96; font-weight: bold; }
.outcome-dismissed { color: #ef553b; font-weight: bold; }
.outcome-unknown { color: #636efa; font-weight: bold; }
.download-btn { background-color: #000000; color: white; padding: 0.5rem 1rem; border: solid; border-color: white; border-radius: 5px; cursor: pointer; margin: 0.2rem; text-decoration: none; display: inline-block; }
.download-btn:hover { background-color: #155a8a; color: white; }
.judgement-summary { background-color: #1e1e1e; padding: 1rem; border-radius: 5px; margin-top: 0.5rem; border-left: 3px solid #1f77b4; font-size: 0.9em; max-height: 200px; overflow-y: auto; }
.summary-badge { background-color: #1f77b4; color: white; padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8em; margin-left: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Backend helpers
# ---------------------------
def ping_backend() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", headers=AUTH_HEADER, timeout=10)
        ok = r.ok and r.json().get("status") == "ok"
        if ok:
            st.success("Backend system initialized successfully")
        else:
            st.warning("Backend responded but not healthy")
        return ok
    except Exception as e:
        logger.warning(f"Backend health failed: {e}")
        st.info("Running in limited mode - backend features unavailable")
        return False

def analyze_via_backend(text: Optional[str] = None, pdf_file=None) -> Optional[Dict]:
    try:
        if pdf_file is not None:
            files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
            r = requests.post(f"{BACKEND_URL}/analyze", headers=AUTH_HEADER, files=files, timeout=180)
        else:
            if not text or not text.strip():
                st.warning("Please provide case text or upload a PDF.")
                return None
            data = {"text": text}
            r = requests.post(f"{BACKEND_URL}/analyze", headers=AUTH_HEADER, data=data, timeout=180)

        if r.ok:
            return r.json()
        else:
            st.error(f"Backend error {r.status_code}: {r.text}")
            return None
    except Exception as e:
        logger.exception("Analyze call failed")
        st.error(f"Analyze error: {e}")
        return None

# ---------------------------
# Local filesystem helpers
# ---------------------------
def get_decade_folders(base_path: str) -> List[str]:
    try:
        if not os.path.exists(base_path):
            return []
        return [os.path.join(base_path, x) for x in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, x))]
    except Exception as e:
        logger.exception("Folder listing error")
        st.error(f"Folder error: {str(e)}")
        return []

def find_case_pdf(base_path: str, case_name: str, case_year: Optional[str] = None) -> Optional[str]:
    try:
        decade_folders = get_decade_folders(base_path)
        if not decade_folders:
            return None

        clean_name = case_name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")

        if case_year:
            try:
                year = int(case_year)
                decade_start = (year // 10) * 10
                decade_end = decade_start + 10
                candidate_dir = os.path.join(base_path, f"{decade_start} to {decade_end}")
                if os.path.exists(candidate_dir):
                    for pattern in [
                        f"{candidate_dir}/{clean_name}.pdf",
                        f"{candidate_dir}/*{clean_name}*.pdf",
                        f"{candidate_dir}/{case_name.replace(' ', '*')}.pdf",
                        f"{candidate_dir}/*{clean_name[:20]}*.pdf",
                    ]:
                        hits = glob.glob(pattern)
                        if hits:
                            return hits[0]
            except (ValueError, TypeError):
                pass

        all_pdfs = []
        for d in decade_folders:
            all_pdfs.extend(glob.glob(f"{d}/*.pdf"))

        case_lower = case_name.lower()
        for pdf in all_pdfs:
            filename = os.path.basename(pdf).lower().replace(".pdf", "").replace("_", " ")
            if filename == case_lower or case_lower in filename or filename in case_lower:
                return pdf

        return None
    except Exception as e:
        logger.exception("PDF search error")
        return None

def get_download_link(pdf_path: str, filename: str) -> str:
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-btn">Download Case PDF</a>'
    except Exception as e:
        logger.exception("PDF read error")
        return ""

def create_case_summary_pdf(case: Dict, original_case_info: Optional[Dict] = None) -> Optional[bytes]:
    try:
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'], fontSize=16,
            spaceAfter=30, alignment=TA_CENTER, textColor=colors.HexColor('#1f77b4')
        )
        normal_style = ParagraphStyle(
            'CustomNormal', parent=styles['Normal'], fontSize=10,
            spaceAfter=6, alignment=TA_JUSTIFY
        )

        story = []
        story.append(Paragraph("CASE ANALYSIS SUMMARY", title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph(f"Case Name: {case.get('case_name', 'N/A')}", normal_style))
        story.append(Paragraph(f"Court: {case.get('court', 'N/A')}", normal_style))
        story.append(Paragraph(f"Outcome: {case.get('outcome', 'N/A').upper()}", normal_style))
        story.append(Paragraph(f"Similarity Score: {case.get('similarity_percentage', 'N/A')}%", normal_style))

        doc.build(story)
        pdf_bytes = buf.getvalue()
        buf.close()
        return pdf_bytes
    except Exception as e:
        logger.exception("Summary PDF error")
        return None

# ---------------------------
# UI components
# ---------------------------
def display_case_card(case: Dict, rank: int, pdf_base_path: str, original_case_info: Optional[Dict] = None):
    sim = float(case.get('similarity_percentage', 0))
    similarity_class = "similarity-high" if sim >= 70 else "similarity-medium" if sim >= 50 else "similarity-low"
    legal_concepts = ', '.join(case.get('legal_concepts', [])[:5]) if case.get('legal_concepts') else 'None identified'
    pdf_path = find_case_pdf(pdf_base_path, case.get('case_name',''), case.get('year'))

    st.markdown(
        f"""
        <div class="case-card">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4>#{rank} {case.get('case_name','Unknown Case')}</h4>
            <div class="{similarity_class}">Similarity: {sim}%</div>
          </div>
          <p><strong>Court:</strong> {case.get('court','Unknown')} | <strong>Year:</strong> {case.get('year','N/A')}</p>
          <p><strong>Legal Concepts:</strong> {legal_concepts}</p>
          <p><strong>Preview:</strong> {(case.get('content_preview','')[:200] + '...') if case.get('content_preview') else ''}</p>
        """, unsafe_allow_html=True
    )

    js = case.get('judgement_summury')
    if case.get('has_judgement_summury') and js:
        with st.expander("View Judgement Summary", expanded=False):
            st.markdown('<div class="judgement-summary">', unsafe_allow_html=True)
            st.write(js)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 10px;">', unsafe_allow_html=True)
    if pdf_path:
        filename = f"case_{rank}_{case.get('case_name','')[:50]}.pdf".replace(" ", "_").replace("/", "_")
        st.markdown(get_download_link(pdf_path, filename), unsafe_allow_html=True)
    else:
        pdf_bytes = create_case_summary_pdf(case, original_case_info)
        if pdf_bytes:
            filename = f"case_{rank}_{case.get('case_name','')[:50]}_summary.pdf".replace(" ", "_").replace("/", "_")
            b64 = base64.b64encode(pdf_bytes).decode()
            st.markdown(
                f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-btn">Download Summary PDF</a>',
                unsafe_allow_html=True
            )
    st.markdown("</div></div>", unsafe_allow_html=True)

def render_result_payload(result: Dict, original_case_text: str, pdf_base_path: str, original_case_info: Optional[Dict] = None):
    if not isinstance(result, dict):
        st.error("Invalid response payload from backend")
        return

    prediction = result.get("prediction", "unknown")
    confidence = result.get("confidence", 0)
    confidence_level = result.get("confidence_level", "Unknown")
    
    if prediction != "unknown":
        outcome_class = "outcome-allowed" if prediction == "allowed" else "outcome-dismissed" if prediction == "dismissed" else "outcome-unknown"
        st.markdown(
            f"""
            <div style='text-align: center; padding: 1rem; background-color: #000000; border-radius: 10px;'>
              <h3>Outcome Prediction</h3>
              <p style='font-size: 1.5rem;' class='{outcome_class}'>{prediction.upper()}
                <span style='font-size: 1rem;'>({confidence}% confidence)</span></p>
              <p><small>{confidence_level}</small></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    similar_cases = result.get("similar_cases", []) or []
    total = result.get("similar_cases_total", len(similar_cases))
    st.subheader(f"Similar Cases Found ({total})")
    if len(similar_cases) > 0:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Download All Available Case PDFs", use_container_width=True):
                with st.spinner("Searching across all decade folders..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                        with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
                            pdf_count = 0
                            # Prefer first 10
                            for i, case in enumerate(similar_cases[:10]):
                                pdf_path = find_case_pdf(pdf_base_path, case.get('case_name',''), case.get('year'))
                                if pdf_path and os.path.exists(pdf_path):
                                    filename = f"case_{i+1}_{case.get('case_name','')[:50]}.pdf".replace(" ", "_").replace("/", "_")
                                    zipf.write(pdf_path, filename)
                                    pdf_count += 1

                            # If no PDFs found, add generated summaries for first 5
                            if pdf_count == 0:
                                for i, case in enumerate(similar_cases[:5]):
                                    pdf_bytes = create_case_summary_pdf(case, original_case_info)
                                    if pdf_bytes:
                                        filename = f"case_{i+1}_{case.get('case_name','')[:50]}_summary.pdf".replace(" ", "_").replace("/", "_")
                                        zipf.writestr(filename, pdf_bytes)
                                        pdf_count += 1
    for i, case in enumerate(similar_cases[:10]):
        display_case_card(case, i + 1, pdf_base_path, original_case_info)

# ---------------------------
# Screens
# ---------------------------
def screen_upload_pdf(pdf_base_path: str):
    st.header("Upload Case PDF")
    uploaded_file = st.file_uploader("Choose a PDF file containing your legal case", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Analyzing PDF via backend..."):
            result = analyze_via_backend(pdf_file=uploaded_file)

        if result:
            original_case_info = {
                'case_name': result.get('case_name', uploaded_file.name),
                'input_method': 'PDF Upload',
                'file_name': uploaded_file.name
            }
            render_result_payload(result, result.get("input_text","") or "", pdf_base_path, original_case_info)

def screen_enter_text(pdf_base_path: str):
    st.header("Enter Case Details")
    case_text = st.text_area("Enter case facts and details:", height=220)
    
    col1, col2 = st.columns(2)
    with col1:
        case_name = st.text_input("Case Name (Optional)")
    with col2:
        court_name = st.text_input("Court (Optional)")

    if st.button("Find Similar Cases", type="primary"):
        if not case_text.strip():
            st.warning("Please enter some case text to analyze.")
            return

        with st.spinner("Analyzing text via backend..."):
            result = analyze_via_backend(text=case_text)

        if result:
            original_case_info = {
                'case_name': case_name or "Manual Input Case",
                'court': court_name or "Not specified",
                'input_method': 'Manual Text Input'
            }
            render_result_payload(result, case_text, pdf_base_path, original_case_info)

def main():
    st.markdown('<div class="main-header">Law & Judgement Model</div>', unsafe_allow_html=True)

    pdf_base_path = st.sidebar.text_input("PDF Base Folder Path", value=PDF_BASE_PATH_DEFAULT)
    get_decade_folders(pdf_base_path)

    ping_backend()

    # Choose input mode
    app_mode = st.radio("Input Method", ["Upload PDF Case", "Enter Case Text"], label_visibility="collapsed")
    if app_mode == "Upload PDF Case":
        screen_upload_pdf(pdf_base_path)
    else:
        screen_enter_text(pdf_base_path)

if __name__ == "__main__":
    main()