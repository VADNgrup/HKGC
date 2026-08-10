import re
from bs4 import BeautifulSoup

def clean_html_to_markdown(html_content: str) -> str:
    """
    Parses a raw HTML string and returns clean, readable Markdown text.
    Filters out scripts, styles, header/footer elements, and nav areas,
    focusing on body content, headings, paragraphs, and list items.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 1. Strip out non-content elements
    for element in soup(["script", "style", "header", "footer", "nav", "aside", "noscript", "iframe"]):
        element.decompose()
        
    # 2. Extract article or main contents if present to ignore sidebar noise
    content_area = soup.find("article") or soup.find("main") or soup.find("div", {"id": "content"}) or soup.find("div", {"class": "mw-parser-output"}) or soup
    
    # 3. Recursively convert tags to markdown structure
    markdown_lines = []
    
    def process_node(node):
        if node.name is None:
            # Text node
            text = node.strip()
            if text:
                # Normalise spacing
                text = re.sub(r"\s+", " ", text)
                markdown_lines.append(text)
            return
            
        if node.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(node.name[1])
            text = node.get_text().strip()
            if text:
                markdown_lines.append(f"\n\n{'#' * level} {text}\n")
        elif node.name == "p":
            text = "".join(child.get_text() if child.name else str(child) for child in node.children).strip()
            if text:
                # Clean nested whitespace
                text = re.sub(r"\s+", " ", text)
                markdown_lines.append(f"\n{text}\n")
        elif node.name in ["ul", "ol"]:
            markdown_lines.append("\n")
            for li in node.find_all("li", recursive=False):
                li_text = li.get_text().strip()
                if li_text:
                    markdown_lines.append(f"- {li_text}\n")
            markdown_lines.append("\n")
        elif node.name == "br":
            markdown_lines.append("\n")
        else:
            for child in node.children:
                process_node(child)
                
    process_node(content_area)
    
    # Combine lines and normalise paragraph breaks
    full_text = " ".join(markdown_lines)
    # Fix spacing around paragraph breaks and headings
    full_text = re.sub(r"\s*\n\s*", "\n", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    
    return full_text.strip()
