import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()  # reads variables from a .env file and sets them in os.environ

FINED_FOLDER = os.getenv("FINED_FOLDER")
RAW_FOLDER = os.getenv("RAW_FOLDER")
WIKI_SOURCE = os.getenv("WIKI_SOURCE")
QNA_SOURCE = os.getenv("QNA_SOURCE")
OUTPUT_DIR = os.getenv("FINED_FOLDER") + "/wiki-text"


def extract_wiki_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Nội dung chính của Wikipedia nằm ở đây
    content = soup.select_one("div#mw-content-text div.mw-parser-output")
    if not content:
        return ""

    # Loại bỏ các phần không phải nội dung chính
    for tag in content.find_all(["table", "style", "script", "figure", "span"]):
        tag.decompose()

    # Loại bỏ các mục không cần thiết (Tham khảo, Liên kết ngoài...)
    blacklist = ["Tham khảo", "Liên kết ngoài", "Xem thêm", "Chú thích"]
    for h2 in content.find_all("h2"):
        if any(b in h2.get_text() for b in blacklist):
            # Xóa toàn bộ phần sau heading này
            for sib in list(h2.next_siblings):
                sib.decompose()
            h2.decompose()

    # Thêm newline sau các thẻ block để ngắt dòng đúng chỗ
    for block in content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "div"]):
        block.insert_after("\n")
        
    for br in content.find_all("br"):
        br.replace_with("\n")

    # Lấy text không dùng separator để giữ liền mạch các thẻ inline (như <a>, <b>)
    text = content.get_text(separator="")
    
    # Dọn dẹp khoảng trắng
    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            # Xoá khoảng trắng thừa
            line = re.sub(r"\s{2,}", " ", line)
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines)



def parse_markdown_table(md_path):
    rows = []
    current_group = None  # FIX: khởi tạo trước

    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Chỉ xử lý dòng bắt đầu bằng "|"
            if not line.startswith("|"):
                continue

            parts = [p.strip() for p in line.split("|")[1:-1]]

            # Bỏ qua header (dòng tên cột hoặc separator ---)
            if parts[0] in ["Giai đoạn", "Period", "---"] or parts[0].startswith("---"):
                continue

            # Nhóm lớn (bold text như **VN_1930_1945**)
            if parts[0].startswith("**"):
                current_group = re.sub(r"\*\*", "", parts[0]).strip()
                continue

            # Nếu chưa có group → bỏ qua
            if current_group is None:
                continue

            # Dòng dữ liệu phải có 5 hoặc 6 cột
            if len(parts) < 5:
                continue

            # Tách URL từ Markdown: [text](url) — cột index 3
            url_match = re.search(r"\]\((.*)\)", parts[3])
            url = url_match.group(1) if url_match else None

            # Cột 5 là description, cột 6 (nếu có) là ContentLanguage
            desc = parts[4] if len(parts) > 4 else ""
            lang = parts[5] if len(parts) > 5 else ""

            rows.append({
                "group": current_group,
                "period": parts[0],
                "topic": parts[1],
                "type": parts[2],
                "url": url,
                "desc": desc,
                "lang": lang,
            })

    return rows

def download_html(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.text


def process_rows(rows):
    for item in rows:
        group_dir = os.path.join(OUTPUT_DIR, item["group"])
        os.makedirs(group_dir, exist_ok=True)

        safe_topic = re.sub(r"[^\w\- ]", "", item["topic"]).replace(" ", "_")
        out_path = os.path.join(group_dir, f"{safe_topic}.txt")

        if not item["url"]:
            print(f"❌ Không có URL cho: {item['topic']}")
            continue

        print(f"📥 Download: {item['topic']} → {item['url']}")

        try:
            html = download_html(item["url"])
            text = extract_wiki_text(html)


            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ Saved: {out_path}")

        except Exception as e:
            print(f"⚠️ Error for {item['topic']}: {e}")


if __name__ == "__main__":
    rows = parse_markdown_table(WIKI_SOURCE)
    print(f"Parsed {len(rows)} rows.")
    process_rows(rows)
