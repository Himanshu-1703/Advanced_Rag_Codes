from openai import AsyncOpenAI
from ragas import EvaluationDataset, evaluate
from typing import Any
from ragas.embeddings.base import embedding_factory


LLM_MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"

# (question, ground_truth_reference)
QA_PAIRS = [
    (
        "What does sustainable development mean?",
        "Development that meets today's needs without preventing future generations from meeting their own needs",
    ),
    (
        "What are the three pillars of sustainable development?",
        "Economic growth, social inclusion, and environmental protection",
    ),
    (
        "How many global goals did countries agree on in the 2030 Agenda?",
        "17 Sustainable Development Goals",
    ),
    (
        "What is the Paris Agreement's temperature target?",
        "Limit global warming to 1.5 degrees Celsius above pre-industrial levels",
    ),
    (
        "What does a circular economy try to reduce?",
        "Waste, by keeping products and materials in use for as long as possible",
    ),
    (
        "What are two examples of circular economy actions?",
        "Recycling and repairing products instead of throwing them away",
    ),
    (
        "What basic needs are highlighted under social equity?",
        "Access to clean water, basic education, and healthcare for all people",
    ),
    (
        "What role do individuals play in sustainable development?",
        "Individuals can reduce waste, save energy, and make eco-friendly choices",
    ),
    (
        "Why is limiting global warming to 1.5 degrees Celsius important?",
        "Beyond 1.5 degrees Celsius, risks of floods, droughts, and extreme weather increase significantly",
    ),
    (
        "What is the target year for the Sustainable Development Goals?",
        "2030",
    ),
]


# dataset --> user_input, retrieved_contexts, response, reference
def run_evaluation(chain, retriever):
    client = AsyncOpenAI()
    
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)
    
    dataset: list[dict[str, Any]] = []
    
    for i, (question, ground_truth) in enumerate(QA_PAIRS):
        context_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in context_docs]


        context_str = "\n\n".join(contexts)
        result = chain.invoke({"context": context_str, "question": question})
        response = result.content

        dataset.append(
            {
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": response,
                "reference": ground_truth,
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    
    results = evaluate(
        dataset=evaluation_dataset,
        embeddings=embeddings,
    )

    df = results.to_pandas()
    
    # save df to a csv file
    df.to_csv("evaluation_results.csv", index=False)
