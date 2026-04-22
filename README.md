# PaddleOCR RAG Pipeline with RAGAS Evaluation

A Retrieval-Augmented Generation (RAG) pipeline that uses **PaddleOCR 3.x** to extract text and bounding boxes from PDF documents, generates answers via **Azure OpenAI (GPT-4o)**, visualises the source regions on the original PDF pages, and evaluates answer quality using **RAGAS** metrics.

---

## Features

- **PaddleOCR extraction** — converts each PDF page to an image and runs OCR to obtain text blocks with precise bounding boxes. Uses the PaddleOCR 3.x defaults for `lang="en"`: `PP-OCRv5_server_det` for detection and `en_PP-OCRv5_mobile_rec` for recognition
- **Hybrid retrieval** — combines BM25 (keyword) and semantic vector search, then reranks with a cross-encoder model
- **Azure OpenAI answer generation** — uses GPT-4o via your `.env` credentials
- **Bounding box visualisation** — highlights the exact source regions on the PDF page that contributed to each answer
- **RAGAS evaluation** — scores every answer on Faithfulness and Answer Relevancy

---

## Project Structure

```
PaddleOCR/
├── .env                        # Azure OpenAI credentials (not committed)
├── requirements.txt
├── README.md
├── document/
│   └── RAGAS.pdf               # Input PDF document
├── codes/
│   ├── __init__.py
│   ├── ocr_pipeline.py         # PDF → PaddleOCR → Documents with bbox metadata
│   ├── rag_pipeline.py         # Hybrid retrieval + reranking + answer generation
│   └── ragas_eval.py           # RAGAS faithfulness + answer relevancy evaluation
├── notebooks/
│   └── RAGAS_v1.ipynb          # End-to-end notebook: OCR → RAG → bbox viz → eval
└── results/                    # Output CSVs written here at runtime
    ├── rag_results_output.csv
    └── ragas_evaluation_results.csv
```

---

## Setup

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Key pinned versions (see `requirements.txt` for the full list):

- `paddlepaddle==3.3.1`, `paddleocr==3.5.0`
- `PyMuPDF==1.25.1`
- `langchain==0.3.25` (plus `langchain-community`, `langchain-core`, `langchain-huggingface`, `langchain-openai`)
- `sentence-transformers==4.1.0`
- `ragas==0.2.15`, `datasets==2.12.0`
- `urllib3<2` (botocore pin — required for `datasets` / `ragas` to import)

> **macOS / Apple Silicon note:** Install the CPU version of PaddlePaddle first if the above fails:
> ```bash
> pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/mac/cpu/stable.html
> pip install paddleocr
> ```

### 3. Configure Azure OpenAI credentials

Create a `.env` file in the project root:

```env
AZURE_API_KEY=<your-azure-api-key>
AZURE_ENDPOINT=https://<your-resource>.openai.azure.com/
DEPLOYMENT_NAME=gpt-4o
API_VERSION=2025-01-01-preview
```

---

## Usage

Open and run `notebooks/RAGAS_v1.ipynb` top-to-bottom. The notebook is self-contained and will:

| Cell | Action |
|------|--------|
| 1 — Setup | Auto-installs PaddleOCR if missing; configures paths and questions |
| 2 — OCR | Extracts text blocks + bounding boxes from `document/RAGAS.pdf` |
| 3 — RAG | Runs hybrid retrieval, reranking, and GPT-4o answer generation; saves `results/rag_results_output.csv` |
| 4 — Bbox viz | Displays each PDF page with retrieved source regions highlighted in red/yellow |
| 5 — Evaluation | Runs RAGAS scoring; saves `results/ragas_evaluation_results.csv` |
| 6 — Summary | Prints mean Faithfulness and Answer Relevancy scores |

### Using the Python modules directly

```python
from codes.ocr_pipeline import extract_documents_from_pdf
from codes.rag_pipeline  import run_rag_pipeline
from codes.ragas_eval    import run_ragas_evaluation

documents, page_images = extract_documents_from_pdf("document/RAGAS.pdf")

questions = ["What metrics does RAGAS propose?"]
rag_df = run_rag_pipeline(documents, questions, output_dir="results")

eval_df = run_ragas_evaluation("results/rag_results_output.csv", output_dir="results")
print(eval_df[["question", "faithfulness", "answer_relevancy"]])
```

---

## Module Reference

### `codes/ocr_pipeline.py`

| Function | Description |
|----------|-------------|
| `extract_documents_from_pdf(pdf_path, dpi=200, gap_threshold=20)` | Runs PaddleOCR on every page; returns `(List[Document], Dict[int, np.ndarray])` — documents carry `page`, `bboxes`, and `source` metadata; the dict maps page index to the rendered image array. Instantiates `PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)` |
| `pdf_page_to_image(pdf_path, page_num, dpi=200)` | Renders a single PDF page to a numpy RGB array via PyMuPDF |
| `run_ocr_on_page(img, ocr_engine)` | Runs PaddleOCR 3.x on one page image and returns `List[(poly, text, confidence)]` by reading `rec_texts` / `rec_polys` / `rec_scores` from the predict result |
| `group_lines_into_blocks(ocr_lines, gap_threshold=20)` | Groups vertically adjacent OCR lines into paragraph blocks |

### `codes/rag_pipeline.py`

| Function / Class | Description |
|------------------|-------------|
| `run_rag_pipeline(documents, questions, output_dir="results")` | Full pipeline: embeddings (`sentence-transformers/all-MiniLM-L6-v2`) → BM25+vector hybrid retrieval → cross-encoder reranking → GPT-4o answers → CSV |
| `SimpleEnsembleRetriever` | Merges BM25 and vector results, deduplicates, returns top 10 |
| `RerankRetriever` | Applies `cross-encoder/ms-marco-MiniLM-L-6-v2` and returns top 3 |

### `codes/ragas_eval.py`

| Function | Description |
|----------|-------------|
| `run_ragas_evaluation(rag_csv_path, output_dir="results")` | Evaluates every row on Faithfulness and Answer Relevancy; retries rows that scored NaN once with a 5-second pause; saves results CSV |

---

## Output Files

| File | Columns |
|------|---------|
| `results/rag_results_output.csv` | `question`, `answer`, `retrieved_texts`, `page_nums`, `bboxes` |
| `results/ragas_evaluation_results.csv` | above + `faithfulness`, `answer_relevancy` |

---

## Retrieval Pipeline

```
PDF pages
   │
   ▼ PaddleOCR (per page image)
Text blocks + bounding boxes
   │
   ├──▶ BM25 retriever (top 5)  ─┐
   └──▶ Vector store (top 5)    ─┴──▶ SimpleEnsembleRetriever (top 10, deduped)
                                          │
                                          ▼ CrossEncoder reranker
                                       Top 3 chunks  ──▶  GPT-4o  ──▶  Answer
```

---

## Evaluation Metrics

| Metric | What it measures | Range |
|--------|-----------------|-------|
| **Faithfulness** | Whether every claim in the answer is supported by the retrieved context | 0 – 1 |
| **Answer Relevancy** | How well the answer addresses the original question | 0 – 1 |

Both metrics use Azure GPT-4o for scoring. Answer Relevancy additionally uses `sentence-transformers/all-MiniLM-L6-v2` embeddings (local, no extra Azure deployment required).
