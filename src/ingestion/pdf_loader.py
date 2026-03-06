# src/ingestion/pdf_loader.py

from collections import Counter
from pathlib import Path
import re
import PyPDF2


class PDFLoader:

    def __init__(self, folder_path: str, max_pages: int = 30):
        self.folder_path = Path(folder_path)
        self.max_pages = max_pages

    @staticmethod
    def _is_heading(line: str) -> bool:
        line = line.strip()
        if not line:
            return False
        words = line.split()
        if len(words) > 14:
            return False
        if re.match(r"^\d+(\.\d+)*[\)\.]?\s+\S+", line):
            return True
        if line.isupper() and len(line) <= 90:
            return True
        if line.endswith(":") and len(words) <= 10:
            return True
        return False

    def _clean_page_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Fix OCR/PDF hard line-break hyphenation.
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

        # Normalize odd bullets and dashes to ASCII.
        text = text.replace("•", "- ").replace("–", "-").replace("—", "-")

        # Normalize whitespace while preserving line boundaries.
        text = re.sub(r"[ \t]+", " ", text)

        lines = [ln.strip() for ln in text.split("\n")]
        cleaned = []

        for line in lines:
            if not line:
                if cleaned and cleaned[-1] != "":
                    cleaned.append("")
                continue

            lowered = line.lower()
            if re.fullmatch(r"\d{1,4}", line):
                continue
            if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", lowered):
                continue
            if re.fullmatch(r"\[\d+\]", line):
                continue
            if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}", line):
                continue
            if lowered.startswith(("department of", "school of", "college of")) and len(line) <= 120:
                continue

            if self._is_heading(line):
                if cleaned and cleaned[-1] != "":
                    cleaned.append("")
                cleaned.append(line)
                cleaned.append("")
            else:
                cleaned.append(line)

        # Collapse multiple blank lines.
        compact = []
        for line in cleaned:
            if line == "" and compact and compact[-1] == "":
                continue
            compact.append(line)

        return "\n".join(compact).strip()

    def _remove_repeating_headers_footers(self, pages_text):
        first_lines = []
        last_lines = []
        split_pages = []

        for text in pages_text:
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            split_pages.append(lines)
            if lines:
                first_lines.append(lines[0])
                last_lines.append(lines[-1])

        threshold = max(3, len(split_pages) // 3)
        repeated_headers = {ln for ln, c in Counter(first_lines).items() if c >= threshold and len(ln) > 4}
        repeated_footers = {ln for ln, c in Counter(last_lines).items() if c >= threshold and len(ln) > 4}

        filtered_pages = []
        for lines in split_pages:
            filtered = []
            for idx, line in enumerate(lines):
                if idx == 0 and line in repeated_headers:
                    continue
                if idx == len(lines) - 1 and line in repeated_footers:
                    continue
                filtered.append(line)
            filtered_pages.append("\n".join(filtered))

        return filtered_pages

    def load_papers(self):
        """
        Returns:
            List of tuples -> [(paper_name, text), ...]
        """
        papers = []

        for pdf_file in self.folder_path.glob("*.pdf"):
            try:
                raw_pages = []
                with open(pdf_file, "rb") as f:
                    reader = PyPDF2.PdfReader(f)

                    for page in reader.pages[: self.max_pages]:
                        page_text = page.extract_text() or ""
                        raw_pages.append(page_text)

                filtered_pages = self._remove_repeating_headers_footers(raw_pages)
                cleaned_pages = []
                for page_text in filtered_pages:
                    cleaned = self._clean_page_text(page_text)
                    if cleaned:
                        cleaned_pages.append(cleaned)

                text = "\n\n".join(cleaned_pages).strip()
                papers.append((pdf_file.name, text))

            except Exception as e:
                print(f"Error reading {pdf_file.name}: {e}")

        return papers
