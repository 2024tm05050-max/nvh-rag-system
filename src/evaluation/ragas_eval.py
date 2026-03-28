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
        "ground_truth": "For vehicles with gross vehicle weight below 3 tonnes, the body weight is 0.12 tonnes and cab weight is not applicable."
    },
    {
        "question": "What does IS 3028 specify about noise measurement method?",
        "ground_truth": "IS 3028 specifies the method of measurement of noise emitted by moving automotive vehicles."
    },
    {
        "question": "What is the purpose of IS 3028?",
        "ground_truth": "IS 3028 specifies the method for measurement of noise emitted by moving automotive vehicles to assess compliance with noise regulations."
    },
    {
        "question": "What vehicle categories are covered in IS 3028?",
        "ground_truth": "IS 3028 covers automotive vehicles including passenger cars, trucks, buses and two or three wheeled vehicles."
    },
    {
        "question": "What international standards are referenced in IS 3028?",
        "ground_truth": "IS 3028 references ECE regulations and ISO 362 for measurement of sound emitted by accelerating road vehicles."
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
    
    # Print per-question scores
    print("\nPer-question scores:")
    for i, row in df.iterrows():
        print(f"\nQ{i+1}: {row['question'][:60]}...")
        print(f"  Faithfulness:     {row.get('faithfulness', 'N/A'):.3f}")
        print(f"  Answer relevancy: {row.get('answer_relevancy', 'N/A'):.3f}")
        print(f"  Context precision:{row.get('context_precision', 'N/A'):.3f}")
    
    # Print aggregate scores
    print("\n" + "-" * 50)
    print("AGGREGATE SCORES (0.0 - 1.0, higher is better):")
    print(f"  Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"  Answer relevancy:  {df['answer_relevancy'].mean():.3f}")
    print(f"  Context precision: {df['context_precision'].mean():.3f}")
    print("-" * 50)
    
    # Interpretation
    avg = df[['faithfulness', 
              'answer_relevancy', 
              'context_precision']].mean().mean()
    
    if avg >= 0.8:
        grade = "Excellent"
    elif avg >= 0.6:
        grade = "Good"
    elif avg >= 0.4:
        grade = "Fair"
    else:
        grade = "Needs improvement"
    
    print(f"\nOverall grade: {grade} (avg: {avg:.3f})")
    
    return df


if __name__ == "__main__":
    results = run_evaluation()
    print_results(results)