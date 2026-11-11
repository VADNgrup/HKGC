import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "dataset" / ".raw" / "entities.csv"
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "entities_gold.json"

MANUAL_ALIASES = {
    "chủ tịch hồ chí minh": (
        "hồ chí minh",
        "Honorific title and personal name identify the same person.",
    ),
    "chủ tịch nước": (
        "tôn đức thắng",
        "In the 1976 source paragraph, the title appears directly before Tôn Đức Thắng's name and refers to him.",
    ),
    "phong trào bình dân học vụ": (
        "bình dân học vụ",
        "The movement label and its short form refer to the same movement.",
    ),
    "chiến dịch chống nạn mù chữ": (
        "chiến dịch xóa nạn mù chữ",
        "Both labels refer to the 1945 anti-illiteracy campaign in these paragraphs.",
    ),
    "bắc cạn": (
        "bắc kạn",
        "Historical spelling variant of the same province name.",
    ),
    "đế quốc nhật bản": (
        "nhật bản",
        "The empire/state label and short country name identify the same actor here.",
    ),
    "phát xít nhật": (
        "nhật bản",
        "In the Cao trào kháng Nhật article, this names the same Japanese wartime actor as Đế quốc Nhật Bản.",
    ),
    "phe việt minh": (
        "việt minh",
        "The faction label refers to the Viet Minh movement.",
    ),
    "cao trào kháng nhật": (
        "cao trào kháng nhật cứu nước-khởi nghĩa từng phần",
        "Short and extended names of the same historical movement.",
    ),
    "khoá huấn luyện giáo viên bình dân học vụ đầu tiên": (
        "khóa huấn luyện giáo viên bình dân học vụ đầu tiên",
        "Orthographic variation in the same training-course name.",
    ),
    "kho thóc, gạo của nhật": (
        "kho thóc gạo của nhật",
        "Punctuation-only variation for the same generic object mention.",
    ),
    "thực dân pháp ở đông dương": (
        "thực dân pháp tại đông dương",
        "The source paragraphs use two locative phrasings for the same colonial actor.",
    ),
    "các lớp bình dân học vụ": (
        "các lớp học bình dân học vụ",
        "The source paragraphs use shortened and expanded labels for the literacy classes.",
    ),
}

PREFERRED_NAMES = {
    "hồ chí minh": "Hồ Chí Minh",
    "bình dân học vụ": "Bình dân học vụ",
    "chiến dịch xóa nạn mù chữ": "Chiến dịch xóa nạn mù chữ",
    "bắc kạn": "Bắc Kạn",
    "nhật bản": "Nhật Bản",
    "tôn đức thắng": "Tôn Đức Thắng",
    "việt minh": "Việt Minh",
    "cao trào kháng nhật cứu nước-khởi nghĩa từng phần": (
        "Cao trào kháng Nhật cứu nước-Khởi nghĩa từng phần"
    ),
    "khóa huấn luyện giáo viên bình dân học vụ đầu tiên": (
        "Khóa huấn luyện giáo viên Bình dân học vụ đầu tiên"
    ),
    "kho thóc gạo của nhật": "Kho thóc gạo của Nhật",
    "thực dân pháp tại đông dương": "Thực dân Pháp tại Đông Dương",
    "các lớp học bình dân học vụ": "Các lớp học Bình dân học vụ",
}

TIME_VALUE = re.compile(r"^(?:\?|(?:\d{1,2}/){0,2}\d{4})$")
PARENTHESIZED_TUPLE = re.compile(r"\(([^()]*)\)")


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.strip().strip("\"'“”‘’")
    return " ".join(value.split())


def article_id_from_url(value: str, fallback: str) -> str:
    slug = unquote(urlsplit(value).path.rsplit("/", 1)[-1], errors="replace")
    slug = unicodedata.normalize("NFKD", slug)
    slug = "".join(char for char in slug if not unicodedata.combining(char))
    slug = re.sub(r"[^a-zA-Z0-9]", "", slug).casefold()
    if slug.startswith("binhdanhocv"):
        return "binh_dan_hoc_vu"
    if slug.startswith("caotraokhangnhatcuunu"):
        return "cao_trao_khang_nhat_cuu_nuoc"
    return slug or fallback


def clean_name(value: str) -> str:
    value = value.strip().strip("\"'“”‘’")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+,", ",", value)
    value = re.sub(r",\s*", ", ", value)
    value = value.strip(" ,")
    if value == "súng của Nhật ở, Ngọc Hà":
        return "súng của Nhật ở Ngọc Hà"
    return value


def extract_tuple_mentions(consensus: str) -> list[tuple[str, str]]:
    mentions: list[tuple[str, str]] = []
    for raw_tuple in PARENTHESIZED_TUPLE.findall(consensus):
        fields = next(csv.reader([raw_tuple], skipinitialspace=True))
        fields = [field.strip() for field in fields]
        if not fields:
            continue

        if len(fields) == 1:
            continue

        end = len(fields)
        removed_time_fields = 0
        while (
            end > 3
            and removed_time_fields < 3
            and TIME_VALUE.fullmatch(fields[end - 1].strip())
        ):
            end -= 1
            removed_time_fields += 1
        content = fields[:end]
        if len(content) < 3:
            continue

        relation_index = 1
        if len(content) >= 6 and normalize_name(content[4]) == "phá kho thóc":
            relation_index = 4
        subject = clean_name(", ".join(content[:relation_index]))
        object_name = clean_name(", ".join(content[relation_index + 1 :]))
        if subject:
            mentions.append((subject, "subject"))
        if object_name:
            mentions.append((object_name, "object"))
    return mentions


def build_inventory(source_path: Path) -> dict:
    with source_path.open(encoding="utf-8-sig", newline="") as source_file:
        rows = list(csv.DictReader(source_file))

    entity_mentions: dict[str, list[dict]] = defaultdict(list)
    entity_names: dict[str, set[str]] = defaultdict(set)
    tuple_count = 0
    incomplete_annotation_count = 0
    paragraph_count = 0
    article_ids = set()

    alias_rules = {
        normalize_name(alias): (normalize_name(canonical), rationale)
        for alias, (canonical, rationale) in MANUAL_ALIASES.items()
    }

    for row_number, row in enumerate(rows, start=2):
        paragraph = (row.get("Paragraph") or "").strip()
        if not paragraph:
            continue
        paragraph_count += 1
        article_ids.add(
            article_id_from_url(row.get("ref_url") or "", f"row_{row_number}")
        )
        consensus = row.get("Consensus") or ""
        tuples = PARENTHESIZED_TUPLE.findall(consensus)
        for raw_tuple in tuples:
            fields = next(csv.reader([raw_tuple], skipinitialspace=True))
            if len(fields) >= 3:
                tuple_count += 1
            else:
                incomplete_annotation_count += 1

        for mention, role in extract_tuple_mentions(consensus):
            mention_key = normalize_name(mention)
            canonical_key = alias_rules.get(mention_key, (mention_key, ""))[0]
            canonical_name = PREFERRED_NAMES.get(
                canonical_key, mention
            )
            entity_names[canonical_key].add(canonical_name)
            entity_names[canonical_key].add(mention)
            entity_mentions[canonical_key].append(
                {
                    "surface": mention,
                    "source_row": row_number,
                    "article_id": article_id_from_url(
                        row.get("ref_url") or "", f"row_{row_number}"
                    ),
                    "source_url": (row.get("ref_url") or "").strip(),
                    "role": role,
                }
            )

    entities = []
    mention_index = 1
    for index, canonical_key in enumerate(sorted(entity_names), start=1):
        observed_names = entity_names[canonical_key]
        canonical_name = PREFERRED_NAMES.get(
            canonical_key, min(observed_names, key=lambda value: (value.casefold(), value))
        )
        aliases = sorted(
            (
                name
                for name in observed_names
                if normalize_name(name) != normalize_name(canonical_name)
            ),
            key=lambda value: (value.casefold(), value),
        )
        mentions = []
        for item in entity_mentions[canonical_key]:
            mentions.append(
                {
                    "id": f"MENTION-{mention_index:04d}",
                    **item,
                }
            )
            mention_index += 1
        entities.append(
            {
                "id": f"ENT-{index:04d}",
                "name": canonical_name,
                "alias": aliases,
                "mention_count": len(mentions),
                "source_rows": sorted(
                    {item["source_row"] for item in entity_mentions[canonical_key]}
                ),
                "article_ids": sorted(
                    {
                        item["article_id"]
                        for item in entity_mentions[canonical_key]
                    }
                ),
                "mentions": mentions,
            }
        )

    manual_merges = [
        {
            "alias": alias,
            "canonical": canonical,
            "rationale": rationale,
        }
        for alias, (canonical, rationale) in MANUAL_ALIASES.items()
    ]
    return {
        "schema_version": 1,
        "source_file": source_path.name,
        "article_count": len(article_ids),
        "article_ids": sorted(article_ids),
        "paragraph_count": paragraph_count,
        "gold_tuple_count": tuple_count,
        "incomplete_annotation_entry_count": incomplete_annotation_count,
        "entity_count": len(entities),
        "manual_merges": manual_merges,
        "entities": entities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the unique gold-entity inventory from consensus tuples."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    inventory = build_inventory(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {inventory['entity_count']} unique entities from "
        f"{inventory['gold_tuple_count']} gold tuples and "
        f"{inventory['paragraph_count']} paragraphs to {args.output}"
    )


if __name__ == "__main__":
    main()
