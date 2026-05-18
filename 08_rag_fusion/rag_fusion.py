import logging
import os
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_fusion.log")

logger = logging.getLogger("rag_fusion_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _file_handler = logging.FileHandler(_log_path)
    _file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_file_handler)


class SubQuerySchema(BaseModel):
    
    sub_queries: list[str] = Field(..., description="List of sub-queries to be generated from the main query.")



class RAGFusion:
    """
    RAG Fusion is a retrieval-augmented generation (RAG) system that combines
    multiple-retrievers or Sub-Queries to enhance the retrieval process. It uses an ensemble of
    retrievers or Sub-Queries to retrieve relevant documents and then fuses the results to generate
    a final response.
    """
    
    def __init__(self,
                retriever: BaseRetriever,
                llm_chain: Runnable | None = None,
                num_subqueries: int = 3,
                k: int = 5):

        self.retriever = retriever
        self.llm_chain = llm_chain
        self.num_subqueries = num_subqueries
        self.k = k


    @classmethod
    def from_retrievers(cls,
                        base_retrievers: list[BaseRetriever],
                        weights: list[float] | None = None,
                        k: int = 5):

        if not base_retrievers:
            raise ValueError("At least one retriever must be provided.")

        if not isinstance(base_retrievers, list):
            raise ValueError("base_retrievers must be a list of retriever instances.")

        ensemble = EnsembleRetriever(retrievers=base_retrievers, weights=weights)

        return cls(retriever=ensemble, k=k)
    
    
    @classmethod
    def from_llm(cls,
                llm,
                retriever: BaseRetriever,
                num_subqueries: int = 3,
                k: int = 5):

        # create the prompt template for generating sub-queries
        prompt = ChatPromptTemplate(
            messages=[
                ("system", "You are a helpful assistant that generates sub-queries from a main query to enhance retrieval."),
                ("user", "Given the main query: '{main_query}', generate {num_subqueries} sub-queries that can be used to retrieve relevant documents.")
            ],
            input_variables=["main_query", "num_subqueries"]
        )

        # use the llm with structured output to generate sub-queries
        structured_llm = llm.with_structured_output(SubQuerySchema)

        # create the chain
        llm_chain = prompt | structured_llm

        return cls(retriever=retriever, llm_chain=llm_chain, num_subqueries=num_subqueries, k=k)
    

    def _retrieve_documents(self, query: str) -> list[Document]:
        """
        Internal method to retrieve documents using the retriever.
        """
        return self.retriever.invoke(query)
    
    
    def _generate_subqueries(self, query: str) -> list[str]:
        """
        Internal method to generate sub-queries using the llm_chain.
        """
        if not self.llm_chain:
            raise ValueError("LLM chain is not provided for generating sub-queries.")
        
        logger.info(f"Generating {self.num_subqueries} sub-queries for main query: '{query}'")

        result = self.llm_chain.invoke({"main_query": query, "num_subqueries": self.num_subqueries})

        logger.info(f"Generated sub-queries: {result.sub_queries}")
        return result.sub_queries
    
    
    def _reciprocal_rank_fusion(self, retrieved_docs: list[list[Document]]) -> list[Document]:  # [[retrieved_docs_sub_query_1], [retrieved_docs_sub_query_2], [retrieved_docs_sub_query_3]]
        """
        Internal method to perform Reciprocal Rank Fusion (RRF) on the retrieved documents.
        This is a simple implementation that ranks documents based on their relevance scores.
        """
        
        doc_scores: dict[str, tuple[float, Document]] = {}  # {"doc_text": (rrf_score, doc_object)}

        for retrieved_set in retrieved_docs:
            for rank, doc in enumerate(retrieved_set, start=1): # [doc_2, doc_4, doc_1] doc_2 --> 1, doc_4 --> 2, doc_1 --> 3
                
                # RRF contribution from this retriever for this document
                rrf_score = 1.0 / (rank + 60)
                key = doc.page_content
                
                if key in doc_scores:   # if doc is in dict {"key": (rrf_score, doc_object)}
                    # Add contribution to existing score
                    prev_score, prev_doc = doc_scores[key]  # fetching score and doc object
                    doc_scores[key] = (prev_score + rrf_score, prev_doc)  # updating score and creating the tuple again
                else:
                    doc_scores[key] = (rrf_score, doc) # {"key": (rrf_score, doc_object)}
        
        # Sort by accumulated score descending and return the documents
        docs_with_scores = doc_scores.values()
        sorted_docs = sorted(docs_with_scores, key=lambda x: x[0], reverse=True)
        
        # log the sorted results
        logger.info(f"RRF Scores with documents: {docs_with_scores}")
        
        return [doc for _, doc in sorted_docs]
    
    
    def invoke(self, query: str) -> list[Document]:
        """
        Main method to invoke the RAG Fusion process. It generates sub-queries (if llm_chain is provided),
        retrieves documents for each sub-query, and fuses the results.
        """
        if self.llm_chain:
            sub_queries = self._generate_subqueries(query)

            all_retrieved_docs = [self._retrieve_documents(sub_query) for sub_query in sub_queries]
            logger.info(f"Retrieved documents per sub-query: {[[doc.page_content[0:50] for doc in docs] for docs in all_retrieved_docs]}")

            fused_docs = self._reciprocal_rank_fusion(all_retrieved_docs)
            return fused_docs[:self.k]
        else:
            return self._retrieve_documents(query)[:self.k]