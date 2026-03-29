from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_core.messages import HumanMessage, SystemMessage
import time

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful AI assistant.
Answer ONLY from the given context.
If answer is not in context, say I don't know.
Context:
{context}
"""

def create_test_dataset():
    test_data = [
        {
            "question"    : "What is the total marks of Mir Ikramullah Raesani?",
            "ground_truth": "Mir Ikramullah Raesani obtained 99 marks out of 150"
        },
        {
            "question"    : "What grade did Laraib Fatima get?",
            "ground_truth": "Laraib Fatima got grade A with 120 marks out of 150"
        },
        {
            "question"    : "Who is the instructor of this course?",
            "ground_truth": "The instructor is Dr. Adil Aslam Mir"
        },
        {
            "question"    : "How many students are in this class?",
            "ground_truth": "There are 48 students in this class"
        },
        {
            "question"    : "Which semester is this result from?",
            "ground_truth": "This is semester 7 Fall 2025"
        },
    ]
    print(f"✅ Test dataset ready — {len(test_data)} questions")
    return test_data


def run_rag_on_testset(test_data, retriever, llm):
    print("\n" + "="*50)
    print("RUNNING RAG ON TEST QUESTIONS")
    print("="*50)

    results = []
    for i, item in enumerate(test_data):
        question      = item["question"]
        ground_truth  = item["ground_truth"]
        print(f"\n[{i+1}/{len(test_data)}] {question}")

        docs          = retriever.invoke(question)
        contexts      = [doc.page_content for doc in docs]
        context_text  = '\n\n'.join(contexts)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_text)
        response      = llm.invoke([
            SystemMessage(system_prompt),
            HumanMessage(question)
        ])
        answer = response.content
        print(f"    → {answer[:80]}...")

        results.append({
            "question"    : question,
            "answer"      : answer,
            "contexts"    : contexts,
            "ground_truth": ground_truth
        })
        time.sleep(20)

    print(f"\n✅ Done — {len(results)} questions processed")
    return results


def run_ragas_evaluation(results, llm, embeddings):
    print("\n" + "="*50)
    print("CALCULATING RAGAS SCORES")
    print("="*50)

    dataset_dict = {
        "question"    : [r["question"]     for r in results],
        "answer"      : [r["answer"]       for r in results],
        "contexts"    : [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }

    dataset          = Dataset.from_dict(dataset_dict)
    ragas_llm        = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    print("Running evaluation — takes 2-3 minutes...")

    scores = evaluate(
        dataset         = dataset,
        metrics         = [faithfulness, answer_relevancy, context_recall],
        llm             = ragas_llm,
        embeddings      = ragas_embeddings,
        raise_exceptions= False
    )
    return scores


def print_evaluation_results(scores):
    df    = scores.to_pandas()
    avg_f = df['faithfulness'].mean()
    avg_r = df['answer_relevancy'].mean()
    avg_c = df['context_recall'].mean()

    print("\n" + "="*50)
    print("RAGAS RESULTS")
    print("="*50)
    print(f"\n  Faithfulness     : {avg_f:.3f}  {'✅' if avg_f > 0.8 else '⚠️'}")
    print(f"  Answer Relevancy : {avg_r:.3f}  {'✅' if avg_r > 0.8 else '⚠️'}")
    print(f"  Context Recall   : {avg_c:.3f}  {'✅' if avg_c > 0.75 else '⚠️'}")

    print(f"\n  {'#':<4}{'Question':<40}{'Faith':<8}{'Relev':<8}{'Recall'}")
    print("  " + "-"*65)
    for i, row in df.iterrows():
        q = row['question'][:37] + "..."
        print(f"  {i+1:<4}{q:<40}{row['faithfulness']:.2f}    {row['answer_relevancy']:.2f}    {row['context_recall']:.2f}")

    df.to_csv("ragas_results.csv", index=False)
    print("\n✅ Saved to ragas_results.csv")
    return df