import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
import pandas as pd
import requests

load_dotenv()  # Read variables from .env file

# Config Environment
FINED_FOLDER = os.getenv("FINED_FOLDER", ".")
RAW_FOLDER = os.getenv("RAW_FOLDER", ".")
WIKI_SOURCE = os.getenv("WIKI_SOURCE", "")
QNA_SOURCE = os.getenv("QNA_SOURCE", "")

INPUT_FILE = os.getenv("QNA_SOURCE")
DEFAULT_OUTPUT_JSON = os.path.join(FINED_FOLDER, "questions_merged.json")
UNKNONW_JSON = os.path.join(FINED_FOLDER, "questions_unknown.json")
STATS_JSON = os.path.join(FINED_FOLDER, "period_stats.json")

# LLM Configuration
LLM_ENDPOINT = os.getenv(
    "LLM_ENDPOINT", "http://172.16.125.200:33040/v1/chat/completions"
)
LLM_API_KEY = os.getenv("LLM_API_KEY", "qwen35b@")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen36")


# ============================
# 1. Load ALL sheets
# ============================
def load_all_sheets(path):
    xls = pd.ExcelFile(path)
    merged = []
    print("Sheets found:", xls.sheet_names)

    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        df.columns = [str(c).strip().lower() for c in df.columns]

        for _, row in df.iterrows():
            q = str(row.get("câu hỏi", "")).strip()
            opts = row.get("lựa chọn", "")
            src = row.get("nguồn đề", "")
            qtype = row.get("loại câu hỏi", "")
            ans = row.get("đáp án", "")

            merged.append({
                "question": q,
                "answer": ans,
                "metadata": {
                    "org_option": opts,
                    "org_code": src,
                    "org_answer": ans,
                    "type_question": qtype,
                },
            })

    return merged


# ============================
# 2. Remove duplicates
# ============================
def remove_duplicates(rows):
    seen = set()
    unique = []
    for r in rows:
        q = r["question"]
        if q not in seen and q:
            seen.add(q)
            unique.append(r)
    return unique


# ============================
# 3. Extract year & Classify Period
# ============================
def extract_year(text):
    years = re.findall(r"(1[0-9]{3}|20[0-9]{2})", text)
    if years:
        return int(years[0])
    return None


PERIODS = [
    ("VN_TCN", 0, 0),
    ("BAC_THUOC", 0, 938),
    ("PHONG_KIEN", 938, 1858),
    ("PHAP_THUOC", 1858, 1918),
    ("VN_1919_1930", 1919, 1930),
    ("VN_1930_1945", 1930, 1945),
    ("VN_1945_1954", 1945, 1954),
    ("VN_1954_1975", 1954, 1975),
    ("VN_1975_NAY", 1975, 2100),
]


def classify_period(question):
    q = question.lower()

    if any(
        k in q
        for k in [
            "lý",
            "trần",
            "lê sơ",
            "lê trung hưng",
            "hồ",
            "tây sơn",
            "phong kiến",
            "chúa nguyễn",
            "chúa trịnh",
        ]
    ):
        return "PHONG_KIEN"

    if any(
        k in q
        for k in [
            "cần vương",
            "đấu tranh chống pháp",
            "pháp xâm lược",
            "đề thám",
            "phan đình phùng",
            "yên thế",
            "đông du",
            "duy tân",
            "phong trào yêu nước cuối thế kỉ xix",
        ]
    ):
        return "PHAP_THUOC"

    if any(
        k in q
        for k in [
            "nguyễn ái quốc",
            "1919",
            "1920",
            "1921",
            "1922",
            "1923",
            "1924",
            "1925",
            "1926",
            "1927",
            "1928",
            "1929",
            "hội việt nam cách mạng thanh niên",
            "tâm tâm xã",
        ]
    ):
        return "VN_1919_1930"

    if any(
        k in q
        for k in [
            "1930",
            "1931",
            "xô viết nghệ tĩnh",
            "đảng cộng sản",
            "mặt trận dân chủ đông dương",
            "cao trào kháng nhật",
            "cách mạng tháng tám",
        ]
    ):
        return "VN_1930_1945"

    if any(
        k in q
        for k in [
            "1946",
            "toàn quốc kháng chiến",
            "biên giới 1950",
            "điện biên phủ",
            "hiệp định giơnevơ",
        ]
    ):
        return "VN_1945_1954"

    if any(
        k in q
        for k in [
            "miền bắc",
            "miền nam",
            "mĩ",
            "đế quốc mĩ",
            "đường mòn hồ chí minh",
            "đồng khởi",
            "mậu thân",
            "điện biên phủ trên không",
            "hiệp định paris",
        ]
    ):
        return "VN_1954_1975"

    if any(
        k in q
        for k in [
            "đổi mới",
            "1986",
            "xu thế toàn cầu hóa",
            "hội nhập",
            "asean",
            "wto",
            "kinh tế thị trường",
        ]
    ):
        return "VN_1975_NAY"

    year = extract_year(question)
    if year:
        for label, start, end in PERIODS:
            if start <= year <= end:
                return label

    return "UNKNOWN"


# ============================
# 4. Smart Answer Extraction (Rule-based Fallback)
# ============================
def convert_answer_to_text(answer_letter, org_option):
    """Trích xuất text đáp án dựa trên ký tự chọn (A, B, C, D) từ danh sách

    options.
    """
    options = []
    try:
        if isinstance(org_option, str):
            options = json.loads(org_option)
        elif isinstance(org_option, list):
            options = org_option
    except Exception:
        return None

    if not options:
        return None

    # Clean Prefix A., B., C., D.
    clean_options = [
        re.sub(r"^[A-Da-d][\.\:\s\)\-]+", "", str(opt)).strip()
        for opt in options
    ]

    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    idx = mapping.get(str(answer_letter).strip().upper())

    if idx is None or idx >= len(clean_options):
        return None

    selected_text = clean_options[idx]

    # Xử lý trường hợp đáp án tổng hợp: "Cả A, B, C", "Tất cả các ý trên"
    selected_lower = selected_text.lower()
    if any(
        k in selected_lower
        for k in ["cả 3", "cả a và b", "cả a, b", "tất cả các", "cả hai"]
    ):
        valid_opts = [
            opt
            for i, opt in enumerate(clean_options)
            if i != idx and "tất cả" not in opt.lower()
        ]
        return " ; ".join(valid_opts)

    return selected_text


# ============================
# 5. Helper Chuẩn hóa Options Trắc nghiệm
# ============================
def format_mcq_options(org_options):
    """Format lại danh sách options thành chuỗi có dạng:

    - Nội dung A
    - Nội dung B
    - Nội dung C
    - Nội dung D
    """
    opts_list = []
    try:
        opts_list = (
            json.loads(org_options)
            if isinstance(org_options, str)
            else org_options
        )
    except Exception:
        pass

    if not isinstance(opts_list, list):
        return ""

    options_formatted = []
    letters = ["A", "B", "C", "D"]
    for i, opt in enumerate(opts_list):
        prefix = f"{letters[i]}." if i < len(letters) else f"{i + 1}."
        clean_opt = re.sub(r"^[A-Da-d][\.\:\s\)\-]+", "", str(opt)).strip()
        options_formatted.append(f"{prefix} {clean_opt}")

    return "\n".join(options_formatted)


# ============================
# 6. LLM Rewrite Module (100% Trắc nghiệm)
# ============================
def rewrite_qna_with_llm(question, raw_answer, org_options):
    """Xử lý câu hỏi trắc nghiệm:

    1. Gắn danh sách 'Các lựa chọn: A, B, C, D' vào question_text.
    2. Gọi LLM suy luận và chuyển answer từ dạng trắc nghiệm/ký hiệu thành câu tự luận hoàn chỉnh.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    # Format options A, B, C, D
    options_text = format_mcq_options(org_options)
    

    # Đính kèm Options vào Question Text để phục vụ Context Search cho RAG
    if options_text:
        full_context_question = (
            f"{question.strip()}\nCác lựa chọn:\n{options_text}"
        )
    else:
        full_context_question = question.strip()

    system_prompt = (
        "Bạn là chuyên gia tinh chỉnh dữ liệu câu hỏi trắc nghiệm cho hệ thống RAG (Retrieval-Augmented Generation).\n"
        "Nhiệm vụ của bạn là phân tích các câu hỏi trắc nghiệm, các lựa chọn (A, B, C, D) và đáp án chọn, "
        "sau đó tinh chỉnh lại câu hỏi và đáp án thành một câu query - answer hoàn chỉnh, rõ nghĩa cho RAG .\n\n"
        "YÊU CẦU :\n"
        "1. KHÔNG ĐƯỢC THAY ĐỔI HAY THÊM THÔNG TIN .\n"
        "2. NẾU câu hỏi theo lựa chọn kiểu `đáp án nào, ý nào, chỗ nào, sự kiện nào ,... trong các phương án sau` thì câu hỏi phải thêm lựa trọn gạch ý thay vì A./B./C .\n"
        "3. NẾU câu hỏi theo kiểu hỏi thông tin thì giữ nguyên mọi thứ. .\n"
        "4. NẾU ĐÁP ÁN LÀ 'CẢ A, B, C' HOẶC 'TẤT CẢ CÁC Ý TRÊN': Hãy tổng hợp nội dung của tất cả các lựa chọn đúng đó thành một đoạn văn trả lời hoàn chỉnh.\n"
        "ĐỊNH DẠNG ĐẦU RA (JSON chuẩn):\n"
        "{\n"
        '  "reasoning": "Phân tích câu hỏi, chọn phương án đúng và lý do...",\n'
        '  "answer": "Nội dung đáp án diễn giải tự luận hoàn chỉnh"\n'
        "}"
    )

    user_prompt = f"""Input:
- Câu hỏi trắc nghiệm: {question}
- Các lựa chọn:
{options_text}
- Đáp án tham chiếu từ hệ thống: {raw_answer}

Hãy phân tích và trả về JSON chứa 'reasoning' và 'answer' theo yêu cầu.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "response_format": (
            {"type": "json_object"} if "qwen" in MODEL_NAME.lower() else None
        ),
    }

    try:
        response = requests.post(
            LLM_ENDPOINT, headers=headers, json=payload, timeout=30
        )
        if response.status_code == 200:
            res_data = response.json()
            content = res_data["choices"][0]["message"]["content"].strip()

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                new_a = parsed.get("answer", raw_answer)
                return full_context_question, new_a
    except Exception as e:
        print(
            f"[LLM Warning] Lỗi khi gọi LLM cho câu hỏi '{question[:30]}...': {e}"
        )

    # Fallback nếu LLM gặp sự cố
    return full_context_question, raw_answer


# ============================
# 7. Helper & Export Functions
# ============================
def stats_by_period(rows):
    counter = Counter([r["period"] for r in rows])
    print("\n=== Thống kê theo giai đoạn ===")
    for period, count in counter.items():
        print(f"{period}: {count} câu hỏi")
    return counter


def save_unknown(rows, filename="questions_unknown.json"):
    unknown = [r for r in rows if r["period"] == "UNKNOWN"]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(unknown, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu các câu hỏi UNKNOWN → {filename}")


def export_to_csv(rows, output_file):
    csv_data = [
        {
            "Question": r.get("question", ""),
            "Gold Answer": r.get("answer_text", ""),
            "Evidence": "",
            "Type": r.get("metadata", {}).get("type_question", ""),
            "Period": r.get("period", ""),
        }
        for r in rows
    ]
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Đã xuất file CSV → {output_file}")


# ============================
# 8. Main Function
# ============================
def main():
    parser = argparse.ArgumentParser(
        description="MCQ Data Processing & LLM RAG Rewriter Tool"
    )

    parser.add_argument(
        "-p",
        "--period",
        type=str,
        default=None,
        help="Lọc theo period (VD: VN_1919_1930, PHONG_KIEN...)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Đường dẫn file đầu ra"
    )
    parser.add_argument(
        "-c",
        "--format-csv",
        action="store_true",
        help="Xuất file dạng CSV thay vì JSON",
    )


    args = parser.parse_args()

    # 1. Load & Deduplicate
    rows = load_all_sheets(INPUT_FILE)
    rows = remove_duplicates(rows)

    print(f"\n[Processing] Đang xử lý {len(rows)} câu hỏi trắc nghiệm...")

    # 2. Process Questions
    new_rows = []
    for idx, r in enumerate(rows):
        r["period"] = classify_period(r["question"])

        # Lấy text đáp án sơ bộ qua Rule-based
        raw_ans_text = convert_answer_to_text(
            r["answer"], r["metadata"]["org_option"]
        )
        r["answer_text"] = raw_ans_text
        if args.period and r["period"] != args.period:
            continue
        if "hình ảnh" in r["question"] or "bức tranh" in r["question"] or not r["answer_text"] or r["answer_text"] == "":
            continue    
        if r["question"].strip()[-1]==',': r["question"] = r["question"].strip()[:-1]
        print(
            f"\n[{idx + 1}/{len(rows)}] [Processing] {r['question']} "
        )

        # 3. LLM Rewrite Integration (Khi bật cờ --use-llm)
        # if args.use_llm and ("nào sau đây" in r["question"].lower() ):

            # rewritten_q, rewritten_a = rewrite_qna_with_llm(
            #     question=r["question"],
            #     raw_answer=raw_ans_text or r["answer"],
            #     org_options=r["metadata"]["org_option"],
            # )
            # r["question"] = rewritten_q
            # r["answer_text"] = rewritten_a
            # r["metadata"]["is_llm_rewritten"] = True
            # r["metadata"]["org_text"] = r["question"]
            # r["metadata"]["org_answer"] = r["answer_text"]
            # print("\t=====>New question is: ", rewritten_q, "=====>New answer is: ", rewritten_a)
        # else:
        if  ("nào sau đây" in r["question"].lower() ):
            # Nếu không bật LLM, vẫn tự động đính kèm options vào question
            opts_formatted = format_mcq_options(r["metadata"]["org_option"])
            if opts_formatted:
                r["question"] = (
                    f"{r['question'].strip()}\n{opts_formatted}"
                )
        elif r['question'].split()[-1].strip() in ["đã", "đã,"] : r['question'] = r['question'] + " làm gì?"
        elif r['question'][-1] != '?': r['question'] = r['question'] + "?"

        print("\t=====>New question is: ", r["question"], "\n\t=====>New answer is: ", r["answer_text"])
        
        new_rows.append(r)
    # Filter question having image or answer is None:
    rows = new_rows
  

    # Determine Output Path
    output_path = args.output
    if not output_path:
        output_path = os.path.join(
            FINED_FOLDER,
            (
                "questions_merged.csv"
                if args.format_csv
                else DEFAULT_OUTPUT_JSON
            ),
        )

    # Save Output
    if args.format_csv:
        export_to_csv(rows, output_path)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Đã xuất file JSON → {output_path}")

    print("Tổng số câu hỏi đã lưu:", len(rows))

    # Save Stats & Unknowns if not filtering
    if not args.period:
        save_unknown(rows, UNKNONW_JSON)
        stats = stats_by_period(rows)
        with open(STATS_JSON, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()