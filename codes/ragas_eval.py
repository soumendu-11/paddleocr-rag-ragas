import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

load_dotenv()


class StripMarkdownCallback(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                if hasattr(gen, "text"):
                    text = gen.text
                    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
                    text = re.sub(r"^```\s*$", "", text, flags=re.MULTILINE)
                    gen.text = text.strip()


def _make_llm(azure_cfg: dict) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        openai_api_version=azure_cfg["api_version"],
        azure_endpoint=azure_cfg["azure_endpoint"],
        azure_deployment=azure_cfg["azure_deployment"],
        api_key=azure_cfg["api_key"],
        model="gpt-4o",
        validate_base_url=False,
        temperature=0,
        max_retries=5,
        request_timeout=120,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _parse_chunks(chunks_text: str):
    chunks = []
    for part in str(chunks_text).split("Chunk "):
        if part.strip():
            content = part.split(":", 1)[-1].strip() if ":" in part else part.strip()
            if content:
                chunks.append(content)
    return chunks or ["No context available"]


def run_ragas_evaluation(
    rag_csv_path: str,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Evaluate RAG results using RAGAS faithfulness and answer_relevancy.

    Uses Azure OpenAI (gpt-4o) for LLM-based scoring and local
    sentence-transformers embeddings for answer relevancy.

    Args:
        rag_csv_path: Path to the CSV produced by run_rag_pipeline().
        output_dir:   Folder to write ragas_evaluation_results.csv.

    Returns:
        DataFrame with original columns plus faithfulness and answer_relevancy.
    """
    os.makedirs(output_dir, exist_ok=True)

    azure_cfg = {
        "api_key": os.getenv("AZURE_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
        "api_version": os.getenv("API_VERSION"),
        "azure_deployment": os.getenv("DEPLOYMENT_NAME"),
    }

    print("Loading RAG results...")
    df = pd.read_csv(rag_csv_path)
    print(f"  {len(df)} rows loaded.")

    contexts_list = df["retrieved_texts"].apply(_parse_chunks).tolist()

    dataset = Dataset.from_dict(
        {
            "question": df["question"].tolist(),
            "answer": df["answer"].tolist(),
            "contexts": contexts_list,
        }
    )

    azure_llm = _make_llm(azure_cfg)
    # Use local embeddings — avoids needing a separate Azure embedding deployment
    local_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # evaluate() wraps LangChain LLMs/embeddings in LangchainLLMWrapper / LangchainEmbeddingsWrapper
    # and only injects them when metric.llm / metric.embeddings are None. Assigning raw
    # AzureChatOpenAI to metric.llm skips that wrap and breaks metric.init (set_run_config).
    faithfulness.llm = None
    answer_relevancy.llm = None
    answer_relevancy.embeddings = None

    print("Running RAGAS evaluation (this may take a few minutes)...")
    time.sleep(2)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=azure_llm,
        embeddings=local_embeddings,
        raise_exceptions=False,
    )

    results_df = results.to_pandas()

    # Retry any NaN scores individually
    for idx in range(len(results_df)):
        needs_retry = pd.isna(results_df.loc[idx, "faithfulness"]) or pd.isna(
            results_df.loc[idx, "answer_relevancy"]
        )
        if not needs_retry:
            continue

        print(f"  Retrying question {idx + 1}...")
        time.sleep(5)
        try:
            single_ds = Dataset.from_dict(
                {
                    "question": [dataset[idx]["question"]],
                    "answer": [dataset[idx]["answer"]],
                    "contexts": [dataset[idx]["contexts"]],
                }
            )
            retry_llm = _make_llm(azure_cfg)
            retry_metrics = []
            if pd.isna(results_df.loc[idx, "faithfulness"]):
                faithfulness.llm = None
                retry_metrics.append(faithfulness)
            if pd.isna(results_df.loc[idx, "answer_relevancy"]):
                answer_relevancy.llm = None
                answer_relevancy.embeddings = None
                retry_metrics.append(answer_relevancy)

            r = evaluate(
                dataset=single_ds,
                metrics=retry_metrics,
                llm=retry_llm,
                embeddings=local_embeddings,
                raise_exceptions=False,
            ).to_pandas()

            for col in ["faithfulness", "answer_relevancy"]:
                if col in r.columns and not pd.isna(r.loc[0, col]):
                    results_df.loc[idx, col] = r.loc[0, col]
        except Exception as exc:
            print(f"    Retry failed: {exc}")

    final_df = df.copy()
    for col in ["faithfulness", "answer_relevancy"]:
        if col in results_df.columns:
            final_df[col] = results_df[col].values

    out_path = os.path.join(output_dir, "ragas_evaluation_results.csv")
    final_df.to_csv(out_path, index=False)
    print(f"\nSaved evaluation results → {out_path}")

    valid_f = final_df["faithfulness"].dropna()
    valid_r = final_df["answer_relevancy"].dropna()
    print(f"  Faithfulness:     {valid_f.mean():.4f}  ({len(valid_f)}/{len(final_df)} valid)")
    print(f"  Answer Relevancy: {valid_r.mean():.4f}  ({len(valid_r)}/{len(final_df)} valid)")

    return final_df
