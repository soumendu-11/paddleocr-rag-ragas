import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import List

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_openai import AzureChatOpenAI
from sentence_transformers import CrossEncoder

load_dotenv()


class SimpleEnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        all_docs = []
        for retriever in self.retrievers:
            all_docs.extend(retriever.invoke(query))
        seen, unique = set(), []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)
        return unique[:10]


class RerankRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    rerank_model: CrossEncoder
    top_n: int = 3

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.rerank_model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def run_rag_pipeline(
    documents: List[Document],
    questions: List[str],
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Build hybrid BM25 + vector retrieval with cross-encoder reranking,
    generate answers via Azure OpenAI, and save results CSV.

    Args:
        documents: OCR-extracted Documents with bbox metadata.
        questions:  List of question strings.
        output_dir: Folder to write rag_results_output.csv.

    Returns:
        DataFrame with columns: question, answer, retrieved_texts, page_nums, bboxes.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings)

    print("Initialising BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = SimpleEnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    print("Loading cross-encoder reranker...")
    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    compression_retriever = RerankRetriever(
        base_retriever=ensemble_retriever,
        rerank_model=rerank_model,
        top_n=3,
    )

    print("Initialising Azure OpenAI LLM...")
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        temperature=0,
    )

    template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": compression_retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\nProcessing {len(questions)} question(s)...")
    rows = []
    for idx, question in enumerate(questions, 1):
        print(f"  [{idx}/{len(questions)}] {question[:70]}...")
        retrieved_docs = compression_retriever.invoke(question)
        answer = rag_chain.invoke(question)

        rows.append(
            {
                "question": question,
                "answer": answer,
                "retrieved_texts": "\n\n".join(
                    f"Chunk {i+1}: {d.page_content}"
                    for i, d in enumerate(retrieved_docs)
                ),
                # JSON-serialise so the CSV is portable; notebook deserialises for display
                "page_nums": json.dumps([d.metadata.get("page", 0) for d in retrieved_docs]),
                "bboxes": json.dumps([d.metadata.get("bboxes", []) for d in retrieved_docs]),
            }
        )

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "rag_results_output.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved RAG results → {out_path}")
    return df
