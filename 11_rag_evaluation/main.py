from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import build_rag_chain
from evaluate import run_evaluation


def main():
    print("Building RAG pipeline...")
    chain, retriever = build_rag_chain()
    print("Pipeline ready. Running evaluation...\n")
    run_evaluation(chain, retriever)


if __name__ == "__main__":
    main()
