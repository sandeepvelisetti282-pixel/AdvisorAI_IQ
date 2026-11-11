
import os
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from agent import run_advisor
from rag import get_retriever


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY is missing. Check your .env file."
    )


# --------------------------------------------------
# 2. Configure Gemini evaluator
# --------------------------------------------------

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)

evaluator_llm = LangchainLLMWrapper(gemini_llm)


# --------------------------------------------------
# 3. Configure Gemini embeddings
# --------------------------------------------------

gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    gemini_embeddings
)


# --------------------------------------------------
# 4. Define test question
# --------------------------------------------------

question = "What is compound interest?"


# --------------------------------------------------
# 5. Generate answer using AdvisorIQ
# --------------------------------------------------

print("\nGenerating answer from AdvisorIQ...")

answer = run_advisor(question)

print("\nGenerated Answer:")
print(answer)


# --------------------------------------------------
# 6. Retrieve contexts from ChromaDB
# --------------------------------------------------

print("\nRetrieving documents from ChromaDB...")

retriever = get_retriever(k=4)

documents = retriever.invoke(question)

contexts = [
    doc.page_content
    for doc in documents
]

if not contexts:
    raise ValueError(
        "No documents retrieved from ChromaDB. "
        "Run python rag.py first."
    )


# --------------------------------------------------
# 7. Prepare RAGAS dataset
# --------------------------------------------------

samples = {
    "question": [question],
    "answer": [answer],
    "contexts": [contexts]
}

dataset = Dataset.from_dict(samples)


# --------------------------------------------------
# 8. Configure legacy RAGAS metrics
# --------------------------------------------------

# Gemini does not support multiple candidates
# for this model, so use strictness = 1.

answer_relevancy.strictness = 1


# --------------------------------------------------
# 9. Run evaluation using Gemini
# --------------------------------------------------

print("\nRunning RAGAS evaluation...")

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("\n========== AdvisorIQ RAGAS RESULTS ==========")

print(result)

print("\nDetailed Evaluation Results:")

print(result.to_pandas())

print("\nEvaluation completed successfully!")
