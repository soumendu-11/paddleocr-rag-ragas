import os
import numpy as np
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from langchain_core.documents import Document
from typing import List, Dict, Tuple


def pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = 200) -> np.ndarray:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    doc.close()
    if pix.n == 4:  # RGBA → RGB
        img = img[:, :, :3]
    return img


def run_ocr_on_page(
    img: np.ndarray, ocr_engine: PaddleOCR
) -> List[Tuple[List[List[float]], str, float]]:
    """
    Run PaddleOCR 3.x on a single page image.
    Returns a list of (poly, text, confidence) where poly is a list of [x, y] points.
    """
    result = ocr_engine.predict(img)
    if not result:
        return []

    page_result = result[0]
    texts  = page_result.get("rec_texts", [])
    polys  = page_result.get("rec_polys", [])
    scores = page_result.get("rec_scores", [])

    lines = []
    for text, poly, score in zip(texts, polys, scores):
        # poly is a numpy ndarray of shape (N, 2); convert to plain python list
        poly_list = np.asarray(poly).tolist()
        lines.append((poly_list, text.strip(), float(score)))
    return lines


def group_lines_into_blocks(
    ocr_lines: List[Tuple[List[List[float]], str, float]],
    gap_threshold: int = 20,
) -> List[Tuple[List[List[List[float]]], str]]:
    """
    Group OCR lines into paragraph blocks based on vertical proximity.

    Returns list of (bboxes, combined_text) where bboxes is a list of polygon
    point-lists (one per source OCR line).
    """
    if not ocr_lines:
        return []

    # Sort by top y-coord of each poly
    def top_y(poly):
        return min(pt[1] for pt in poly)

    def bottom_y(poly):
        return max(pt[1] for pt in poly)

    sorted_lines = sorted(ocr_lines, key=lambda x: top_y(x[0]))

    blocks = []
    current_texts = [sorted_lines[0][1]]
    current_bboxes = [sorted_lines[0][0]]
    prev_bottom = bottom_y(sorted_lines[0][0])

    for poly, text, _ in sorted_lines[1:]:
        if top_y(poly) - prev_bottom <= gap_threshold:
            current_texts.append(text)
            current_bboxes.append(poly)
        else:
            blocks.append((current_bboxes, " ".join(current_texts)))
            current_texts = [text]
            current_bboxes = [poly]
        prev_bottom = bottom_y(poly)

    blocks.append((current_bboxes, " ".join(current_texts)))
    return blocks


def extract_documents_from_pdf(
    pdf_path: str, dpi: int = 200, gap_threshold: int = 20
) -> Tuple[List[Document], Dict[int, np.ndarray]]:
    """
    Run PaddleOCR (3.x API) on each page of a PDF and return LangChain Documents
    with polygon metadata plus a page_images dict for display.

    Returns:
        documents: List of Documents. Each has metadata:
                   { 'page': int, 'bboxes': list of polygon point-lists, 'source': str }
        page_images: dict mapping page_num → numpy RGB image array
    """
    ocr_engine = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    documents: List[Document] = []
    page_images: Dict[int, np.ndarray] = {}

    print(f"Processing {num_pages} page(s) from {os.path.basename(pdf_path)}...")
    for page_num in range(num_pages):
        img = pdf_page_to_image(pdf_path, page_num, dpi=dpi)
        page_images[page_num] = img

        ocr_lines = run_ocr_on_page(img, ocr_engine)
        blocks = group_lines_into_blocks(ocr_lines, gap_threshold=gap_threshold)

        for bboxes, text in blocks:
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": page_num,
                            "bboxes": bboxes,   # list of polygon point-lists
                            "source": pdf_path,
                        },
                    )
                )

    print(f"Extracted {len(documents)} text blocks total.")
    return documents, page_images
