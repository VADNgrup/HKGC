import argparse
import os
import sys
import csv

from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
load_dotenv(os.path.join(REPO_ROOT, ".env"), override=False)

from tqdm import tqdm
from minirag import MiniRAG, QueryParam
from minirag.llm import (
    ollama_embed,
    ollama_model_complete,
)
from minirag.utils import EmbeddingFunc


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG - QnA LiHua World (one-by-one)")
    parser.add_argument("--llm-provider", type=str, default=os.getenv("LLM_PROVIDER", "ollama"))
    parser.add_argument("--llm-model", type=str, default=os.getenv("LLM_MODEL"))
    parser.add_argument("--llm-host", type=str, default=os.getenv("LLM_HOST", "http://localhost:11434"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("REQUEST_TIMEOUT", "600.0")))
    parser.add_argument("--emb-provider", type=str, default=os.getenv("EMBED_PROVIDER", "ollama"))
    parser.add_argument("--emb-model", type=str, default=os.getenv("EMBED_MODEL"))
    parser.add_argument("--emb-dim", type=int, default=int(os.getenv("EMBED_MODEL_DIM", "1024")))
    parser.add_argument("--graph-storage", type=str, default=os.getenv("GRAPH_STORAGE", "NetworkXStorage"))
    parser.add_argument(
        "--workingdir", type=str, default=os.getenv("WORKING_DIR", "../output/LiW-minirag-9b")
    )
    parser.add_argument(
        "--querypath",
        type=str,
        default=os.getenv("QUERY_PATH", "../dataset/LiHua-World/qa/query_set.csv"),
    )
    parser.add_argument(
        "--outputpath",
        type=str,
        default=os.getenv("OUTPUT_PATH", "../output/LiW-minirag-9b/output_QA_raw.csv"),
    )
    parser.add_argument("--mode", type=str, default="light", help="QueryParam mode: mini | light | global | hybrid")
    parser.add_argument("--force", action="store_true", help="Force re-run even if output file exists")
    return parser.parse_args()


args = get_args()

llm_host = args.llm_host
llm_timeout = args.timeout
LLM_MODEL = args.llm_model
EMB_MODEL = args.emb_model
EMB_DIM = args.emb_dim
LLM_MAX_TOKEN_SIZE = int(os.getenv("LLM_MAX_TOKEN_SIZE", "200"))

WORKING_DIR = args.workingdir
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
QUERY_MODE = args.mode

print("USING LLM MODEL  :", LLM_MODEL)
print("USING EMBED MODEL:", EMB_MODEL)
print("USING WORKING DIR:", WORKING_DIR)
print("QUERY PATH       :", QUERY_PATH)
print("OUTPUT PATH      :", OUTPUT_PATH)
print("QUERY MODE       :", QUERY_MODE)

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Setup LLM + Embedding
# ------------------------------------------------------------------
llm_model_func = ollama_model_complete
llm_model_kwargs = {"host": llm_host, "timeout": llm_timeout}

embedding_func = EmbeddingFunc(
    embedding_dim=EMB_DIM,
    max_token_size=1000,
    func=lambda texts: ollama_embed(
        texts,
        embed_model=EMB_MODEL,
        host=llm_host,
        timeout=llm_timeout,
    ),
)

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    graph_storage=args.graph_storage,
    llm_model_max_token_size=LLM_MAX_TOKEN_SIZE,
    llm_model_name=LLM_MODEL,
    llm_model_kwargs=llm_model_kwargs,
    embedding_func=embedding_func,
)


def run_single_question(question: str) -> str:
    """Query RAG for a single question, return the answer string."""
    try:
        query_param = QueryParam(
            mode=QUERY_MODE,
            max_token_for_text_unit=4000,
            max_token_for_global_context=4000,
            max_token_for_local_context=4000,
        )
        raw_response = rag.query(question, param=query_param)
        answer = (
            raw_response.replace("\n", " ")
            .replace("\r", " ")
            .strip()
        )
    except Exception as exc:
        import traceback
        print(f"  [ERROR] {repr(exc)}")
        traceback.print_exc()
        answer = "Error"
    return answer


def run_experiment():
    # ------------------------------------------------------------------
    # Load full question list
    # ------------------------------------------------------------------
    questions = []
    with open(QUERY_PATH, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "question": row["Question"],
                "gold_answer": row["Gold Answer"],
            })

    print(f"Total questions: {len(questions)}")

    # ------------------------------------------------------------------
    # Resume: find already-answered questions
    # ------------------------------------------------------------------
    answered = set()
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    write_header = True
    if os.path.exists(OUTPUT_PATH) and not args.force:
        with open(OUTPUT_PATH, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                answered.add(row.get("Question", "").strip())
        write_header = len(answered) == 0
        print(f"Already answered: {len(answered)} questions — resuming from where we left off.")

    # ------------------------------------------------------------------
    # Run one-by-one, append immediately
    # ------------------------------------------------------------------
    with open(OUTPUT_PATH, mode="a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(["Question", "GoldAnswer", "RAG"])
            log_file.flush()

        pending = [q for q in questions if q["question"].strip() not in answered]
        print(f"Remaining: {len(pending)} questions\n")

        for i, q in enumerate(tqdm(pending, desc="Querying RAG", unit="q")):
            question = q["question"]
            gold = q["gold_answer"]

            print(f"\n[{i+1}/{len(pending)}] Q: {question}")
            print(f"         Gold: {gold}")

            rag_answer = run_single_question(question)
            print(f"         RAG : {rag_answer}")

            writer.writerow([question, gold, rag_answer])
            log_file.flush()  # save immediately after each question

    print(f"\nDone! Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_experiment()
