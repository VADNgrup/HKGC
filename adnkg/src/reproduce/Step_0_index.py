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


from minirag import MiniRAG
from minirag.llm import gpt_4o_mini_complete, hf_embed, ollama_embed, ollama_model_complete
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_args():
    parser = argparse.ArgumentParser(description="HisRAG")
    parser.add_argument("--llm-provider", type=str, default=os.getenv("LLM_PROVIDER", "ollama"))
    parser.add_argument("--llm-model", type=str, default=os.getenv("LLM_MODEL"))
    parser.add_argument("--llm-host", type=str, default=os.getenv("LLM_HOST", "http://localhost:11434"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("REQUEST_TIMEOUT", "600.0")))
    parser.add_argument("--emb-provider", type=str, default=os.getenv("EMBED_PROVIDER"))
    parser.add_argument("--emb-model", type=str, default=os.getenv("EMBED_MODEL"))
    parser.add_argument("--emb-dim", type=int, default=int(os.getenv("EMBED_MODEL_DIM", "1024")))
    parser.add_argument("--graph-storage", type=str, default=os.getenv("GRAPH_STORAGE", "NetworkXStorage"))
    parser.add_argument("--outputpath", type=str, default=os.getenv("OUTPUT_PATH", "./logs/Default_output.csv"))
    parser.add_argument("--workingdir", type=str, default=os.getenv("WORKING_DIR", "../output/LiHua-World"))
    parser.add_argument("--datapath", type=str, default=os.getenv("DATA_PATH", "./dataset/LiHua-World/data/"))
    parser.add_argument(
        "--querypath",
        type=str,
        default=os.getenv("QUERY_PATH", "./dataset/LiHua-World/qa/query_set.csv"),
    )
    parser.add_argument(
        "--fl",
        action="store_true",
        default=os.getenv("ENABLE_FUZZY_LOGIC", "False").lower() in {"1", "true", "yes", "on"},
        help="Enable fuzzy logic alignment",
    )
    parser.add_argument("--fl-mode", type=str, default=os.getenv("FL_MODE", "simple"), choices=["simple", "full"], help="Fuzzy logic alignment mode (simple or full)")
    parser.add_argument(
        "-mp",
        "--minirag-prompt",
        action="store_true",
        default=False,
        help="Use prompt_minirag instead of schema prompt",
    )

    args = parser.parse_args()
    return args


args = get_args()
print("USING FL: ", args.fl, " with mode: ", args.fl_mode)

llm_provider = (args.llm_provider or "ollama").lower()
llm_host = args.llm_host
llm_timeout = args.timeout

LLM_MODEL = args.llm_model
if not LLM_MODEL:
    raise ValueError("Missing LLM model. Set LLM_MODEL in .env or pass --llm-model.")

emb_provider = args.emb_provider or ("ollama" if llm_provider == "ollama" else "hf")
emb_provider = emb_provider.lower()

EMB_MODEL = args.emb_model
if not EMB_MODEL:
    if emb_provider == "ollama":
        raise ValueError("Missing embedding model. Set EMBED_MODEL in .env or pass --emb-model.")
    EMB_MODEL = EMBEDDING_MODEL

EMB_DIM = int(os.getenv("EMBED_MODEL_DIM", "1024"))
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
else:
    llm_model_func = gpt_4o_mini_complete
    llm_model_kwargs = {}

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
else:
    embedding_func = EmbeddingFunc(
        embedding_dim=args.emb_dim,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    )

rag = MiniRAG(
    working_dir=WORKING_DIR,
    graph_storage=args.graph_storage,
    llm_model_func=llm_model_func,
    llm_model_max_token_size=2000,
    llm_model_name=LLM_MODEL,
    llm_model_kwargs=llm_model_kwargs,
    embedding_func=embedding_func,
    enable_fuzzy_logic=args.fl,
    fl_mode=args.fl_mode,
    use_minirag_prompt=args.minirag_prompt,
)

print("Starting indexing...")
# Now indexing
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files

import time
from tqdm import tqdm

stime = time.time()
WEEK_LIST = find_txt_files(DATA_PATH)
for idx, WEEK in enumerate(tqdm(WEEK_LIST, desc="Indexing files"), 1):
    tqdm.write(f"\n[{idx}/{len(WEEK_LIST)}] Indexing: {os.path.basename(WEEK)}")
    with open(WEEK, encoding="utf-8") as f:
        content = f.read()
        
    # Check if content starts with HTML DOCTYPE or contains typical HTML tags
    if "<!DOCTYPE html>" in content or "<html" in content:
        tqdm.write(f"Detected HTML format in {os.path.basename(WEEK)}. Converting to clean markdown text...")
        from minirag.preprocess.html2md import clean_html_to_markdown
        content = clean_html_to_markdown(content)
        
    rag.insert(content)

print("Total time: ", time.time() - stime)
print("======================================================\n\n")