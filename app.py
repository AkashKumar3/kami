from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pdf2image import convert_from_bytes
from io import BytesIO
from model_utils import split_tall_image, ocr_image
from fpdf import FPDF

app = FastAPI()

TALL_PAGE_HEIGHT = 10000
SPLIT_CHUNK_HEIGHT = 9000

def text_to_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(0, 10, line.encode('latin-1', 'replace').decode('latin-1'), ln=1)
    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    file_bytes = await file.read()
    images = convert_from_bytes(file_bytes, dpi=300, fmt='png')
    all_text = []
    for img in images:
        if img.height > TALL_PAGE_HEIGHT:
            for chunk in split_tall_image(img, SPLIT_CHUNK_HEIGHT):
                all_text.append(ocr_image(chunk))
        else:
            all_text.append(ocr_image(img))
    extracted = "\n\n".join(all_text)
    pdf_bytes = text_to_pdf(extracted)
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=extracted_text.pdf"}
    )
