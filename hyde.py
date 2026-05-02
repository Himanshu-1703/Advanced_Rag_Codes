import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableSequence, Runnable

log_path = Path(__file__).parent / "logs.log"

handler = logging.FileHandler(log_path)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

logger = logging.getLogger("hyde_logger")
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class CustomHypotheticalDocumentEmbedder:

    def __init__(self, llm_chain: RunnableSequence, retriever: BaseRetriever):
        self.llm_chain = llm_chain
        self.retriever = retriever


    @classmethod
    def from_llm(cls, llm: Runnable, retriever: BaseRetriever):

        prompt = ChatPromptTemplate(
            messages=[
                ("system", (
                    "You are an expert document writer. Given a user query, your task is to generate a hypothetical document "
                    "that would directly and thoroughly answer the query. "
                    "The document should be written as if it were a real, authoritative passage retrieved from a knowledge base — "
                    "not a response to the user, but a self-contained piece of text that contains the answer. "
                    "Write in a factual, informative tone. Do not include phrases like 'Based on your query' or 'Here is a document'. "
                    "Output only the hypothetical document text, nothing else."
                )),
                ("human", "query: {query}")
            ],
            input_variables=["query"]
        )

        llm_chain = prompt | llm
        return cls(llm_chain=llm_chain, retriever=retriever)


    def _get_relevant_documents(self, hypothetical_document: str) -> list[Document]:
        retrieved_docs = self.retriever.invoke(hypothetical_document)
        return retrieved_docs


    def _generate_hypothetical_document(self, query: str) -> str:
        hypothetical_document = self.llm_chain.invoke({"query": query})
        logger.info(f"Hypothetical document generated:\n{hypothetical_document.content}")
        return hypothetical_document.content


    def invoke(self, query: str) -> list[Document]:
        hypothetical_document = self._generate_hypothetical_document(query)
        relevant_documents = self._get_relevant_documents(hypothetical_document)
        return relevant_documents
