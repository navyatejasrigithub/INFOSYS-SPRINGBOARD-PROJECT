# src/ingestion/chunking.py

import re


class TextChunker:
    """
    Semantic chunker:
    - splits on headings/paragraphs/sentences
    - keeps sentence boundaries
    - applies overlap via trailing sentence carry-over
    """

    def __init__(self, chunk_size=1500, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text or ""
        # Preserve newlines for heading/paragraph cues, collapse repeated spaces.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _is_heading(line: str) -> bool:
        line = line.strip()
        if not line:
            return False

        words = line.split()
        if len(words) > 14:
            return False

        # Common heading shapes: numbered headings, all caps, title-like short lines.
        if re.match(r"^\d+(\.\d+)*[\)\.]?\s+\S+", line):
            return True
        if line.isupper() and len(line) <= 90:
            return True
        if line.endswith(":") and len(words) <= 10:
            return True

        alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
        if alpha_words and len(alpha_words) <= 10:
            titled = sum(1 for w in alpha_words if w[:1].isupper())
            if titled / len(alpha_words) >= 0.8:
                return True

        return False

    @staticmethod
    def _split_sentences(text: str):
        if not text:
            return []
        # Sentence split with punctuation boundary.
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _semantic_segments(self, text: str):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []

        segments = []
        para_buffer = []

        def flush_para():
            if not para_buffer:
                return
            paragraph = " ".join(para_buffer).strip()
            if paragraph:
                segments.extend(self._split_sentences(paragraph))
            para_buffer.clear()

        for line in lines:
            if self._is_heading(line):
                flush_para()
                segments.append(line)
            else:
                para_buffer.append(line)

        flush_para()
        return segments

    def chunk_text(self, text: str):
        text = self._normalize_text(text)
        if not text:
            return []

        segments = self._semantic_segments(text)
        if not segments:
            return []

        chunks = []
        current = []
        current_len = 0

        def flush_current():
            nonlocal current, current_len
            if not current:
                return
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)

            if self.overlap <= 0:
                current = []
                current_len = 0
                return

            # Keep trailing segments as semantic overlap.
            carry = []
            carry_len = 0
            for seg in reversed(current):
                carry.insert(0, seg)
                carry_len += len(seg) + 1
                if carry_len >= self.overlap:
                    break
            current = carry
            current_len = sum(len(seg) + 1 for seg in current)

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            seg_len = len(seg) + 1

            # Very long segment fallback: split hard.
            if seg_len > self.chunk_size:
                flush_current()
                for i in range(0, len(seg), self.chunk_size):
                    piece = seg[i:i + self.chunk_size].strip()
                    if piece:
                        chunks.append(piece)
                current = []
                current_len = 0
                continue

            if current and current_len + seg_len > self.chunk_size:
                flush_current()

            current.append(seg)
            current_len += seg_len

        if current:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)

        return chunks
