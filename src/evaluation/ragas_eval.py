"""
RAGAS Evaluation module
Measures RAG answer quality using standard metrics:
- Faithfulness: answer grounded in retrieved context?
- Answer relevancy: answer addresses the question?
- Context precision: right chunks retrieved?
"""

import os
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.ingestion.embedder import get_embedding_model
from src.retrieval.retriever import retrieve_chunks
from src.models.llm import generate_answer


# NVH-specific test questions with known correct answers
# These are ground truth Q&A pairs from IS 3028
NVH_TEST_SET = [
    {
        "question": "What is the body weight for vehicles below 3 tonnes in Table 1?",
        "ground_truth": "For vehicles below 3 tonnes gross vehicle weight, body weight is 0.12 tonnes and cab weight is not applicable."
    },
    {
        "question": "What are the load schedule requirements for vehicles between 3 and 4 tonnes?",
        "ground_truth": "For vehicles with gross vehicle weight 3 and above but less than 4 tonnes, the body weight is 0.20 tonnes."
    },
    {
        "question": "What European directives are listed in the standards reference table?",
        "ground_truth": "The standards reference table lists 971241EC for permissible sound level of two or three wheel motor vehicles and 70/157/EEC amended by directive 96/20/EC."
    },
    {
        "question": "What is IS 9779 about?",
        "ground_truth": "IS 9779:1981 is about sound level meters."
    },
    {
        "question": "What does IS 9211 cover?",
        "ground_truth": "IS 9211:1979 covers denominations and definitions of weights of road vehicles."
    },
]


def run_evaluation(test_set=None, top_k=5):
    """
    Run RAGAS evaluation on the NVH RAG system.
    
    Args:
        test_set: List of dicts with 'question' and 'ground_truth'
        top_k: Number of chunks to retrieve per query
        
    Returns:
        RAGAS evaluation results
    """
    if test_set is None:
        test_set = NVH_TEST_SET
    
    print(f"Running RAGAS evaluation on {len(test_set)} questions...")
    print("=" * 50)
    
    # Load embedding model
    embed_model = get_embedding_model()
    
    # Build evaluation dataset
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for i, item in enumerate(test_set):
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"\nQ{i+1}: {q}")
        
        # Retrieve chunks
        retrieved = retrieve_chunks(q, embed_model, top_k=top_k)
        
        if not retrieved:
            print("  No chunks retrieved — skipping")
            continue
        
        # Generate answer
        answer = generate_answer(q, retrieved)
        
        # Extract context texts
        context_texts = [chunk["content"] for chunk in retrieved]
        
        print(f"  Answer: {answer[:100]}...")
        print(f"  Chunks retrieved: {len(retrieved)}")
        
        questions.append(q)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(gt)
    
    # Create HuggingFace dataset format (required by RAGAS)
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    
    print("\n" + "=" * 50)
    print("Running RAGAS scoring...")
    
    # Configure RAGAS to use OpenAI
    llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    ))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    ))
    
    # Set LLM and embeddings for each metric
    metrics = [faithfulness, answer_relevancy, context_precision]
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
    
    # Run evaluation
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )
    
    return results


def print_results(results):
    """Print evaluation results in a readable format"""
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    
    df = results.to_pandas()
    
    # Print available columns
    print(f"\nMetrics available: {list(df.columns)}")
    
    # Print per-question scores
    print("\nPer-question scores:")
    for i, row in df.iterrows():
        print(f"\nQ{i+1}:")
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                print(f"  {col}: {val:.3f}")
    
    # Print aggregate scores for numeric columns
    print("\n" + "-" * 50)
    print("AGGREGATE SCORES (0.0 - 1.0, higher is better):")
    numeric_cols = df.select_dtypes(include='number').columns
    scores = []
    for col in numeric_cols:
        mean_val = df[col].mean()
        scores.append(mean_val)
        print(f"  {col}: {mean_val:.3f}")
    
    # Overall grade
    if scores:
        avg = sum(scores) / len(scores)
        if avg >= 0.8:
            grade = "Excellent"
        elif avg >= 0.6:
            grade = "Good"
        elif avg >= 0.4:
            grade = "Fair"
        else:
            grade = "Needs improvement"
        print(f"\nOverall grade: {grade} (avg: {avg:.3f})")
    
    print("-" * 50)
    return df


if __name__ == "__main__":
    results = run_evaluation()
    print_results(results)