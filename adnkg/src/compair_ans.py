import os
import pandas as pd
import requests
import json
from tqdm import tqdm



LLM_ENDPOINT = "http://172.16.125.200:33040/v1/chat/completions"
LLM_API_KEY = "qwen35b@"
MODEL_NAME = "qwen36"

def evaluate_response(question, gold, rag):
    prompt = f"""You are an expert judge evaluating a Question Answering system.
Given a Question, an Expected Answer (Gold), and a System Response (RAG), evaluate the System Response based on two metrics:

1. Accuracy (acc): True if the System Response is semantically equivalent to or correctly answers the Expected Answer. For instance, expected "bottle", response "water bottle" is True.
2. Error (err): True if the System Response provides incorrect information without recognizing its mistake. For example, if it confidently gives a wrong answer, this is True.

Return your evaluation as a valid JSON object with the keys "acc" (boolean) and "err" (boolean). Do not include any other text.

Question: {question}
Expected Answer: {gold}
System Response: {rag}"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(LLM_ENDPOINT, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        # Extract JSON if surrounded by markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        result = json.loads(content)
        return result.get("acc", False), result.get("err", False)
    except Exception as e:
        print(f"Error evaluating: {e}")
        return False, False

def evaluate_csv(filepath):
    print(f"Evaluating {filepath}...")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    df = pd.read_csv(filepath)
    total = len(df)
    acc_count = 0
    err_count = 0
    
    # Optional: Limit for quick testing, e.g., df = df.head(10)
    
    for idx, row in tqdm(df.iterrows(), total=total):
        q = row.get("Question", "")
        gold = row.get("GoldAnswer", "")
        rag = row.get("RAG", "")
        
        acc, err = evaluate_response(q, gold, rag)
        if acc:
            acc_count += 1
        if err:
            err_count += 1
            
    print(f"Results for {filepath}:")
    print(f"Total: {total}")
    print(f"Accuracy (acc): {acc_count} / {total} ({(acc_count/total)*100:.2f}%)")
    print(f"Error Rate (err): {err_count} / {total} ({(err_count/total)*100:.2f}%)")
    print("-" * 30)

if __name__ == "__main__":
    # ---- CONFIG ----
    # FILE_1 = "LiW-fl-9b/output_QA_raw.csv"
    # FILE_2 = "LiW-minirag-9b/output_QA_raw.csv"
    # FILES = ["../output/Vhis-fl-9b/output_QA_raw.csv",
    #         "../output/Vhis-minirag-9b/output_QA_raw.csv",
    #         "../output/ablation1/Vhis-fl-grid_llm0_map0.75/output_QA_raw.csv",
    #         "../output/ablation1/Vhis-fl-grid_llm0_map0.80/output_QA_raw.csv",
    #         "../output/ablation1/Vhis-fl-grid_llm0_map0.85/output_QA_raw.csv",
    #         "../output/ablation1/Vhis-fl-grid_llm1_map0.75/output_QA_raw.csv",
    #         "../output/ablation1/Vhis-fl-grid_llm1_map0.80/output_QA_raw.csv",
    #         "../output/ablation1/Vhis-fl-grid_llm1_map0.85/output_QA_raw.csv",
    #         ]
    FILES  = [
        # "../output/VNH3045-minirag-35b/output_QA_raw.csv",
        # "../output/VNH3045-fl-35b-r2/output_QA_raw.csv",
        # "../output/VNH3045-onlyschema-9b/output_QA_raw.csv",
        "../output/VNH3045-fl-noschema-9b/output_QA_raw.csv"
        
    ]
    for FILE in FILES:
        evaluate_csv(FILE)
