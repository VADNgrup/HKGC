# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import argparse
import os
import sys

from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
load_dotenv(os.path.join(REPO_ROOT, ".env"), override=False)

import csv
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import (
    hf_model_complete,
    hf_embed,
    ollama_embed,
    ollama_model_complete,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_SHORTCUTS = {
    "PHI": "microsoft/Phi-3.5-mini-instruct",
    "GLM": "THUDM/glm-edge-1.5b-chat",
    "MiniCPM": "openbmb/MiniCPM3-4B",
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
}


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--llm-provider", type=str, default=os.getenv("LLM_PROVIDER", "ollama"))
    parser.add_argument("--llm-model", type=str, default=os.getenv("LLM_MODEL"))
    parser.add_argument("--llm-host", type=str, default=os.getenv("LLM_HOST", "http://localhost:11434"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("REQUEST_TIMEOUT", "600.0")))
    parser.add_argument("--emb-provider", type=str, default=os.getenv("EMBED_PROVIDER"))
    parser.add_argument("--emb-model", type=str, default=os.getenv("EMBED_MODEL"))
    parser.add_argument("--emb-dim", type=int, default=int(os.getenv("EMBED_MODEL_DIM", "1024")))
    parser.add_argument("--outputpath", type=str, default=os.getenv("OUTPUT_PATH", "./logs/Default_output.csv"))
    parser.add_argument("--workingdir", type=str, default=os.getenv("WORKING_DIR", "./LiHua-World"))
    parser.add_argument("--datapath", type=str, default=os.getenv("DATA_PATH", "./dataset/LiHua-World/data/"))
    parser.add_argument(
        "--querypath",
        type=str,
        default=os.getenv("QUERY_PATH", "./dataset/LiHua-World/qa/query_set.csv"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Legacy HF model shortcut: PHI|GLM|MiniCPM|qwen",
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default=None,
        help="Single question to query from CLI",
    )
    args = parser.parse_args()
    return args


args = get_args()

llm_provider = (args.llm_provider or "ollama").lower()
llm_host = args.llm_host
llm_timeout = args.timeout

LLM_MODEL = args.llm_model
if not LLM_MODEL and args.model:
    LLM_MODEL = MODEL_SHORTCUTS.get(args.model)
    if LLM_MODEL is None:
        raise ValueError("Invalid --model value. Use PHI|GLM|MiniCPM|qwen.")
if not LLM_MODEL:
    raise ValueError("Missing LLM model. Set LLM_MODEL in .env or pass --llm-model/--model.")

emb_provider = args.emb_provider or ("ollama" if llm_provider == "ollama" else "hf")
emb_provider = emb_provider.lower()

EMB_MODEL = args.emb_model
if not EMB_MODEL:
    if emb_provider == "ollama":
        raise ValueError("Missing embedding model. Set EMBED_MODEL in .env or pass --emb-model.")
    EMB_MODEL = EMBEDDING_MODEL

EMB_DIM = args.emb_dim
LLM_MAX_TOKEN_SIZE = int(os.getenv("LLM_MAX_TOKEN_SIZE", "200"))

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("USING LLM PROVIDER:", llm_provider)
print("USING LLM MODEL:", LLM_MODEL)
print("USING EMBED PROVIDER:", emb_provider)
print("USING EMBED MODEL:", EMB_MODEL)
print("USING WORKING DIR:", WORKING_DIR)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

if llm_provider == "ollama":
    llm_model_func = ollama_model_complete
    llm_model_kwargs = {"host": llm_host, "timeout": llm_timeout}
elif llm_provider == "hf":
    llm_model_func = hf_model_complete
    llm_model_kwargs = {}
else:
    raise ValueError(f"Unsupported LLM provider: {llm_provider}")

if emb_provider == "ollama":
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
elif emb_provider == "hf":
    embedding_func = EmbeddingFunc(
        embedding_dim=EMB_DIM,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMB_MODEL),
            embed_model=AutoModel.from_pretrained(EMB_MODEL),
        ),
    )
else:
    raise ValueError(f"Unsupported embedding provider: {emb_provider}")

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    llm_model_max_token_size=LLM_MAX_TOKEN_SIZE,
    llm_model_name=LLM_MODEL,
    llm_model_kwargs=llm_model_kwargs,
    embedding_func=embedding_func,
)

def run_experiment(output_path):
    headers = ["Question", "Gold Answer", "minirag"]

    q_already = []
    if os.path.exists(output_path):
        with open(output_path, mode="r", encoding="utf-8") as question_file:
            reader = csv.DictReader(question_file)
            for row in reader:
                q_already.append(row["Question"])

    row_count = len(q_already)
    print("row_count", row_count)

    with open(output_path, mode="a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if row_count == 0:
            writer.writerow(headers)

        for QUESTIONid in trange(row_count, len(QUESTION_LIST)):  #
            QUESTION = QUESTION_LIST[QUESTIONid]
            Gold_Answer = GA_LIST[QUESTIONid]
            print()
            print("QUESTION", QUESTION)
            print("Gold_Answer", Gold_Answer)

            try:
                minirag_answer = (
                    rag.query(QUESTION, param=QueryParam(mode="mini"))
                    .replace("\n", "")
                    .replace("\r", "")
                )
            except Exception as e:
                import traceback

                print("Error in minirag_answer", repr(e))
                traceback.print_exc()
                minirag_answer = "Error"

            writer.writerow([QUESTION, Gold_Answer, minirag_answer])

    print(f"Experiment data has been recorded in the file: {output_path}")


def run_single_question(question: str):
    print()
    print("QUESTION", question)
    try:
        answer = (
            rag.query(question, param=QueryParam(mode="mini"))
            .replace("\n", "")
            .replace("\r", "")
        )
    except Exception as exc:
        import traceback

        print("Error in minirag_answer", repr(exc))
        traceback.print_exc()
        answer = "Error"
    print("ANSWER", answer)


# if __name__ == "__main__":

if args.question:
    run_single_question(args.question)
else:
    # Now QA
    QUESTION_LIST = []
    GA_LIST = []
    with open(QUERY_PATH, mode="r", encoding="utf-8") as question_file:
        reader = csv.DictReader(question_file)
        for row in reader:
            QUESTION_LIST.append(row["Question"])
            GA_LIST.append(row["Gold Answer"])

    run_experiment(OUTPUT_PATH)
