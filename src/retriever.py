import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import CrossEncoder

load_dotenv()

class Retriever:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):

        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},  # Change to 'cuda' if GPU available
        )

        # Initialize semantic chunker
        self.chunker = SemanticChunker(self.embeddings)

        # Initialize re-ranker
        self.reranker = CrossEncoder(reranker_model_name, device="cpu")

        self.vectorstore: Optional[FAISS] = None

    def chunk_documents(self, documents: List[str]) -> List[Document]:
       
        chunks = []
        for doc in documents:
            doc_chunks = self.chunker.create_documents([doc])
            chunks.extend(doc_chunks)
        return chunks

    def build_index(self, documents: List[str]):
       
        chunks = self.chunk_documents(documents)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

    def retrieve(
        self, query: str, top_k_dense: int = 20, top_k_rerank: int = 5
    ) -> List[Document]:
        
        if self.vectorstore is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Stage 1: Dense vector search
        dense_results = self.vectorstore.similarity_search(query, k=top_k_dense)

        # Stage 2: Re-ranking with cross-encoder
        query_doc_pairs = [(query, doc.page_content) for doc in dense_results]
        rerank_scores = self.reranker.predict(query_doc_pairs)

        # Sort by scores descending and select top_k_rerank
        sorted_indices = sorted(
            range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True
        )[:top_k_rerank]

        return [dense_results[i] for i in sorted_indices]