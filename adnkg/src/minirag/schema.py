import re
import html
import unicodedata
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class BaseEntity(BaseModel):
    entity_name: str
    entity_type: str
    description: str
    source_id: str
    alternate_names: List[str] = Field(default_factory=list)
    time_range: Optional[str] = None
    geo_location: Optional[str] = None


class Person(BaseEntity):
    entity_type: str = "person"
    birth_date: Optional[str] = None
    death_date: Optional[str] = None


class Place(BaseEntity):
    entity_type: str = "place"
    location_type: Optional[str] = None


class TimeEntity(BaseEntity):
    entity_type: str = "time"
    temporal_value: Optional[str] = None


class Event(BaseEntity):
    entity_type: str = "event"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    location: Optional[str] = None


class Artifact(BaseEntity):
    entity_type: str = "artifact"
    creator: Optional[str] = None
    time_period: Optional[str] = None


class ObjectEntity(BaseEntity):
    entity_type: str = "object"


class Concept(BaseEntity):
    entity_type: str = "concept"


# ---------------------------------------------------------------------------
# Time range parsing utility
# ---------------------------------------------------------------------------
ROMAN_NUMERALS = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18,
    "XIX": 19, "XX": 20, "XXI": 21
}

def parse_time_range(time_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse chronological year ranges from strings.
    E.g.:
      "1225-1258" -> (1225, 1258)
      "năm 1370" -> (1370, 1370)
      "thế kỷ XIII" -> (1201, 1300)
      "thế kỷ 14" -> (1301, 1400)
    Returns:
      (start_year, end_year) or None
    """
    if not time_str:
        return None
    
    clean_str = time_str.strip().lower()
    
    # 1. Check for century pattern, e.g. "thế kỷ xiii", "thế kỷ 13", "century 13"
    century_match = re.search(
        r"(?:thế kỷ|the ky|century)\s+([0-9ivxlcdm]+)", clean_str, re.IGNORECASE
    )
    if century_match:
        cent_val = century_match.group(1).upper()
        cent_num = None
        if cent_val.isdigit():
            cent_num = int(cent_val)
        elif cent_val in ROMAN_NUMERALS:
            cent_num = ROMAN_NUMERALS[cent_val]
        
        if cent_num is not None:
            end_year = cent_num * 100
            start_year = end_year - 99
            return (start_year, end_year)
            
    # 2. Check for ranges like "1225-1258" or "1225 - 1258" or "1225 to 1258"
    range_match = re.search(r"(\d+)\s*(?:-|to|đến)\s*(\d+)", clean_str)
    if range_match:
        return (int(range_match.group(1)), int(range_match.group(2)))
        
    # 3. Check for single years like "năm 1370" or just digits "1370"
    single_match = re.search(r"\b(\d{3,4})\b", clean_str)
    if single_match:
        year = int(single_match.group(1))
        return (year, year)
        
    return None


# ---------------------------------------------------------------------------
# Strict Cleaning and Noise Filtering Utilities
# ---------------------------------------------------------------------------

class EntityNormalizer(ABC):
    @abstractmethod
    def normalize_name(self, name: str) -> str:
        pass


class MultiLingualHistoryNormalizer(EntityNormalizer):
    def __init__(self):
        # Maps specific variants to canonical English forms or standardized uppercase
        self.canonical_map = {
            "ĐẠI VIỆT SỬ KÍ TOÀN THƯ": "DAI VIET SU KY TOAN THU",
            "ĐẠI VIỆT SỬ KÝ TOÀN THƯ": "DAI VIET SU KY TOAN THU",
            "ĐẠI VIỆT SỬ KÝ TOÀN THƯ (TOAN THU": "DAI VIET SU KY TOAN THU",
            "ĐVSKTT": "DAI VIET SU KY TOAN THU",
            "ĐẠI VIỆT": "DAI VIET",
            "ĐẠI TƯỚNG QUÂN": "GENERAL",
            "ĐẠI ĐẾ MÔNG CỔ": "MONGOL EMPIRE",
            "ĐẾ QUỐC MÔNG CỔ": "MONGOL EMPIRE",
            "XÂM LƯỢC, CAI TRỊ": "INVASION",
            "XÂM LƯỢC, CHIẾN TRANH": "INVASION",
            "CHIẾN TRANH": "WAR",
            "ĐỘNG VŨ LÂM": "VU LAM",
            "THĂNG LONG": "THANG LONG",
            "ĐÔNG KINH THĂNG LONG": "THANG LONG",
            "ĐÔNG ĐÔ THĂNG LONG": "THANG LONG",
            "XÂY DỰNG, LẬP ĐÔ": "CONSTRUCTION",
            "ĐỔI MỚI": "DOI MOI",
            "ĐƠN VỊ TIỀN TỆ": "CURRENCY",
            "ĐẾ QUỐC VIỆT NAM": "EMPIRE OF VIETNAM"
        }

        # Substrings to just strip out
        self.noise_substrings = [
            '" ENTITY"<', 'MW-COLLAPSED BORDER CSS SELECTION', '"<',
            '".', ' (TOAN THU', ' (THÁI BÌNH'
        ]

    def is_pure_ascii(self, text: str) -> bool:
        """Returns True if the text contains only ASCII characters."""
        return all(ord(char) < 128 for char in text)

    def detect_script(self, text: str) -> str:
        """Classify script type of name: latin, cjk, or other."""
        for ch in text:
            try:
                name_ch = unicodedata.name(ch, "")
            except ValueError:
                continue
            if "CJK" in name_ch or "HANGUL" in name_ch or "KATAKANA" in name_ch or "HIRAGANA" in name_ch:
                return "cjk"
        return "latin"

    def remove_diacritics(self, text: str) -> str:
        """Removes Vietnamese diacritics/accents to enforce standard ASCII fallback."""
        text = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', text)
        text = re.sub(r'[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]', 'A', text)
        text = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', text)
        text = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', text)
        text = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', text)
        text = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', text)
        text = re.sub(r'[ìíịỉĩ]', 'i', text)
        text = re.sub(r'[ÌÍỊỈĨ]', 'I', text)
        text = re.sub(r'[ùúụủũưừứựửữ]', 'u', text)
        text = re.sub(r'[ÙÚỤỦŨƯỪỨỰỬỮ]', 'U', text)
        text = re.sub(r'[ỳýỵỷỹ]', 'y', text)
        text = re.sub(r'[ỲÝỴỶỸ]', 'Y', text)
        text = re.sub(r'[đ]', 'd', text)
        text = re.sub(r'[Đ]', 'D', text)
        return text

    def normalize_name(self, name: str) -> str:
        if not name:
            return name
        
        # Strip exact match known noise strings
        for n in self.noise_substrings:
            name = name.replace(n, "")
            
        name = name.strip()
        
        # Check canonical map first
        if name.upper() in self.canonical_map:
            return self.canonical_map[name.upper()]
            
        script = self.detect_script(name)
        if script == "latin":
            # Normalize Unicode NFC to preserve Vietnamese accents cleanly
            norm = unicodedata.normalize("NFC", name)
            return norm.upper()
        else:
            # For CJK or other non-latin, uppercase and NFC normalize
            norm = unicodedata.normalize("NFC", name)
            return norm.upper()


# Default instance
_normalizer = MultiLingualHistoryNormalizer()


def strip_html_tags(text: str) -> str:
    """Strips HTML tags like <a> or <span> and returns text content."""
    # E.g. <a href="http...">Hàm Tử</a> -> Hàm Tử
    text = html.unescape(text)
    # Remove HTML comments <!-- ... -->
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Remove HTML tags, keeping inner text
    cleaned = re.sub(r"<[^>]*>", "", text)
    return cleaned.strip()


def clean_quotes_and_brackets(text: str) -> str:
    """Strips leading/trailing outer single/double quotes, formatting markers, and brackets."""
    if not text:
        return text
    
    # Keep stripping outer quotes, brackets, and markup until stable
    prev = ""
    curr = text.strip()
    while prev != curr:
        prev = curr
        # Strip outer quotes and markdown/bracket characters
        curr = re.sub(r'^[\'"`\s\(\)\[\]\{\}\-\+\=\>\/\*\<]+|[\'"`\s\(\)\[\]\{\}\-\+\=\>\/\*\<]+$', '', curr)
        # Handle cases like "NAME"> or /NAME or NAME/>
        curr = curr.strip().strip('"').strip("'").strip('>').strip('<').strip('*')
        curr = curr.strip()
        
    return curr


def clean_entity_name(name: str) -> str:
    """
    Deep-clean an entity name extracted from LLM output which may contain
    HTML artifacts, leaked CSS/wiki markup, or alternative name suggestions.

    Examples of inputs that become clean:
      '"ĐẠI VIỆT"/>                  -> "ĐẠI VIỆT"
      '"ĐẠI VIỆT"_SƯ_ĐOÀN_         -> "ĐẠI VIỆT"
      '"ĐẠI ĐẾ MÔNG CỔ"<!--MONGOL EMPIRE-->  -> "ĐẠI ĐẾ MÔNG CỔ"
      '"ĐỀN TRẦN (THÁI BÌNH)" CLASS="MW-REDIRECT">ĐỀN TRẦN Ở LONG HƯNG'
                                      -> "ĐỀN TRẦN (THÁI BÌNH)"
      'ĐỘNG VŨ LINH" (NOTE: TEXT SAYS VŨ LÂM) -> "ĐỘNG VŨ LÂM"'
                                      -> "ĐỘNG VŨ LÂM"
      '"ĐỒI MỒI" <OR> "GIẾNG CHIM THÚ"...'  -> "ĐỒI MỒI"
    """
    if not name:
        return name

    text = name

    # 0. Remove HTML comments  <!-- ... -->
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 0.5. Strip markdown formatting artifacts from LLM output
    # Remove bold (**text**), italic (*text*), heading markers (#), and wiki links ([[text]])
    text = re.sub(r'\*{1,3}', '', text)        # *** / ** / * markers
    text = re.sub(r'#{1,6}\s*', '', text)       # # heading markers
    text = re.sub(r'\[\[|\]\]', '', text)       # [[ ]] wiki link brackets
    text = text.strip('<>').strip()              # trailing < or >

    # 1. Remove full HTML tags (keep inner text first pass)
    text = re.sub(r"<[^>]*>", " ", text)
    text = html.unescape(text)

    # 2. Remove leaked HTML attributes:  class="..." style="..." href="..."
    text = re.sub(r'\b(?:class|style|href|rel|id|data-[a-z\-]*)\s*=\s*"[^"]*"', '', text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:class|style|href|rel|id|data-[a-z\-]*)\s*=\s*'[^']*'", '', text, flags=re.IGNORECASE)

    # 3. Handle alternative name patterns: split and keep first segment
    #    e.g.  NAME <OR> NAME2   or   NAME -> NAME2   or   NAME (LIKELY NAME2)
    for splitter in [r'\s*<OR>\s*', r'\s*->\s*', r'\s*\(LIKELY\b[^)]*\)\s*', r'\s*\(NOTE:[^)]*\)\s*']:
        parts = re.split(splitter, text, maxsplit=1, flags=re.IGNORECASE)
        text = parts[-1] if parts[-1].strip() else parts[0]  # prefer the corrected name (after ->)

    # 4. Remove trailing tag fragments:  /> |> "> '>
    text = re.sub(r'[/|]\s*>\s*$', '', text)
    text = re.sub(r'["\']\s*>\s*$', '', text)

    # 5. Remove trailing underscores and similar connectors
    text = re.sub(r'[_]+$', '', text)
    text = re.sub(r'^[_]+', '', text)

    # 6. Remove content after first pipe |  (wiki table leak)
    if '|>' in text or '|' in text:
        text = text.split('|')[0]

    # 7. Final pass: strip quotes, brackets, whitespace
    text = clean_quotes_and_brackets(text)

    text = text.strip()
    
    # 8. Standardize through normalizer
    text = _normalizer.normalize_name(text)
    
    return text


def _has_foreign_script(text: str) -> bool:
    """
    Returns True if the text contains characters from scripts irrelevant to
    a Vietnamese-history KG (Arabic, CJK ideographs, Cyrillic, etc.).
    Vietnamese Latin (with diacritics) is allowed.
    """
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):  # Letter category
            try:
                script_name = unicodedata.name(ch, "")
            except ValueError:
                continue
            # Allow Latin letters (includes Vietnamese diacritics)
            if "LATIN" in script_name:
                continue
            # Allow some common symbols that appear in Vietnamese text
            if "COMBINING" in script_name:
                continue
            # Block Arabic, CJK, Cyrillic, etc.
            if any(s in script_name for s in ["ARABIC", "CJK", "CYRILLIC", "HANGUL", "KATAKANA", "HIRAGANA", "DEVANAGARI", "THAI"]):
                return True
    return False


def is_noise_entity(name: str) -> bool:
    """
    Checks if the name is garbage (CSS properties, Hex color codes, SVGs, Base64 strings, URLs,
    foreign scripts, wiki/mediawiki remnants, CSS class selectors, pure-punctuation strings).
    Returns True if it is noise and should be discarded.
    """
    if not name:
        return True
    
    # 1. Less than 2 characters
    if len(name) < 2:
        return True

    # 2. Pure punctuation / numbers / symbols  (no letters at all)
    if not re.search(r"[a-zA-ZÀ-ỹ]", name):
        return True
        
    # 3. Check for hexadecimal color codes, e.g. #FFEF8F
    if re.match(r"^#?[a-f0-9]{6}$", name, re.IGNORECASE) or re.match(r"^#?[a-f0-9]{3}$", name, re.IGNORECASE):
        return True
        
    # 4. Check for typical CSS properties/values
    css_noise_keywords = [
        "display:", "color:", "background:", "font-", "border-", "margin-", "padding-", 
        "box-shadow", "clip-path", "rotate", "translate", "url(", "class=", "id=", "webkit-",
        "position:", "float:", "width:", "height:", "progressive", "opacity:", "border:",
        "overflow", "flex", "z-index", "text-align", "white-space", "line-height",
        "transform", "transition", "animation", "cursor:", "visibility:",
    ]
    name_lower = name.lower()
    for kw in css_noise_keywords:
        if kw in name_lower:
            return True
            
    # 5. CSS variable / selector patterns
    if name.startswith("--") or name.endswith(":") or name.startswith("."):
        return True
    # CSS class selectors like .cdx-button, .mw-mmv-...
    if re.match(r"^\.[a-zA-Z][a-zA-Z0-9_-]*", name):
        return True
        
    # 6. Check for CSS units/dimensions, e.g. .5em, 8.5714286em, 10px, 100vh
    if re.match(r"^\.?\d+(\.\d+)?(em|px|vh|vw|rem|%)\b", name_lower):
        return True
        
    # 7. Check for SVG coordinate paths, e.g. M20 14.5V-2.9L...
    if re.match(r"^[mldcqzshva]\d+", name_lower) and len(name) > 10:
        return True
        
    # 8. Check for long strings without spaces (base64/hex/token metadata)
    if len(name) > 25 and " " not in name:
        return True
        
    # 9. MediaWiki / Wikipedia system labels
    wiki_noise = [
        "mediawiki", "mediawiki engine", "mediawiki css framework",
        "vector interface", "user interface", "cdx-tabs", "cdx-button",
        "cdx-tab", "mw-mmv", "mw-redirect", "wikidata", "wikimedia",
        "wikimedia foundation", "wikimedia foundation, inc.", "wikipedia",
        "widget-repo-buzzard", "wayback machine", "w3d",
        "đặc biệt: editpage/bản mẫu:lịch sử việt nam", "đăng nhập",
        "đây là chủ đề trung tâm của trang web liên quan đến lịch sử việt nam."
    ]
    if name_lower in wiki_noise or any(name_lower.startswith(w) for w in ["cdx-", "mw-", ".mw-", ".cdx-"]):
        return True

    # 10. Contains HTML tag remnants
    if re.search(r"<[a-zA-Z/]", name) or re.search(r"[a-zA-Z]\s*=\s*[\"']", name):
        return True

    # 11. Foreign scripts (Arabic, CJK, Cyrillic, etc.)
    if _has_foreign_script(name):
        return True

    # 12. Check for "VAR(" pattern (CSS variable function)
    if "var(" in name_lower:
        return True
        
    return False


def clean_entity_info(
    name: Optional[str],
    entity_type: Optional[str],
    description: Optional[str],
    alternate_names: Optional[Union[List[str], str]] = None,
    time_range: Optional[str] = None,
    geo_location: Optional[str] = None
) -> Optional[Tuple[str, str, str, List[str], Optional[str], Optional[str]]]:
    """
    Cleans raw entity information extracted from LLM.
    Returns cleaned tuple (name, type, description, alternate_names, time_range, geo_location)
    or None if the entity is invalid/noise.
    """
    if not name or not entity_type:
        return None
        
    # 1. Deep-clean entity name (handles HTML artifacts, alternatives, wiki markup)
    name_clean = clean_entity_name(name)
    
    # 2. Clean HTML tags from other fields
    type_clean = strip_html_tags(entity_type).upper()
    desc_clean = strip_html_tags(description or "")
    time_clean = strip_html_tags(time_range or "")
    geo_clean = strip_html_tags(geo_location or "")
    
    # 3. Clean outer quotes, brackets, and spaces
    type_clean = clean_quotes_and_brackets(type_clean)
    desc_clean = clean_quotes_and_brackets(desc_clean)
    time_clean = clean_quotes_and_brackets(time_clean)
    geo_clean = clean_quotes_and_brackets(geo_clean)
    
    # 4. Clean alternate names
    clean_aliases: List[str] = []
    if alternate_names:
        if isinstance(alternate_names, str):
            raw_aliases = re.split(r"[;,\n|]+", alternate_names)
        else:
            raw_aliases = alternate_names
        for alt in raw_aliases:
            c_alt = clean_entity_name(alt)
            if c_alt and c_alt.upper() != name_clean and not is_noise_entity(c_alt):
                c_alt = unicodedata.normalize("NFC", c_alt).upper()
                if c_alt not in clean_aliases:
                    clean_aliases.append(c_alt)

    # 5. Guard against noise
    if is_noise_entity(name_clean) or is_noise_entity(type_clean):
        return None
    
    # Post-clean length check: entities cleaned to fewer than 2 real letters are garbage
    letters_only = re.sub(r'[^a-zA-ZÀ-ỹ]', '', name_clean)
    if len(letters_only) < 2:
        return None

    # Enforce uppercase & NFC normalization for entity names
    name_clean = unicodedata.normalize("NFC", name_clean).upper()
    
    return name_clean, type_clean, desc_clean, clean_aliases, time_clean or None, geo_clean or None
