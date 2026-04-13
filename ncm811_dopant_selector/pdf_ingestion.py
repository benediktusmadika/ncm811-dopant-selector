from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .constants import LOGGER_NAME
from .models import Chunk, Document
from .optional_deps import PdfReader, fitz, httpx, pdfplumber

LOG = logging.getLogger(LOGGER_NAME)


class GrobidClient:
    def __init__(self, base_url: str, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def is_alive(self) -> bool:
        if httpx is None:
            return False
        try:
            r = httpx.get(f"{self.base_url}/api/isalive", timeout=self.timeout_s)
            return r.status_code == 200 and "true" in (r.text or "").lower()
        except Exception:
            return False

    def process_fulltext(self, pdf_path: Path) -> Optional[str]:
        if httpx is None:
            return None
        url = f"{self.base_url}/api/processFulltextDocument"
        try:
            with open(pdf_path, "rb") as f:
                files = {"input": (pdf_path.name, f, "application/pdf")}
                r = httpx.post(url, files=files, timeout=self.timeout_s)
            if r.status_code != 200:
                return None
            if not r.text or "<TEI" not in r.text:
                return None
            return r.text
        except Exception:
            return None

def _alpha_score(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    if n == 0:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    return alpha / n

def tei_to_text_rich(tei_xml: str, max_chars: Optional[int] = None) -> str:
    """
    Rich TEI -> text:
    - title + abstract + section headings + paragraphs
    - figure/table captions
    - table cell text (important for numeric capacity values)
    """
    if not tei_xml:
        return ""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(tei_xml)
    except Exception:
        txt = re.sub(r"<[^>]+>", " ", tei_xml)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt[:max_chars] if max_chars else txt

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    out: List[str] = []

    # Title
    title_nodes = root.findall(".//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:title", ns)
    if title_nodes:
        title = " ".join("".join(t.itertext()).strip() for t in title_nodes if "".join(t.itertext()).strip())
        if title:
            out.append("TITLE: " + title)

    # Abstract
    abs_nodes = root.findall(".//tei:teiHeader//tei:profileDesc//tei:abstract//tei:p", ns)
    if abs_nodes:
        abs_txt = "\n".join("".join(p.itertext()).strip() for p in abs_nodes if "".join(p.itertext()).strip())
        if abs_txt:
            out.append("ABSTRACT:\n" + abs_txt)

    # Body with headings
    for div in root.findall(".//tei:text//tei:body//tei:div", ns):
        head = div.find("tei:head", ns)
        if head is not None:
            hs = "".join(head.itertext()).strip()
            if hs:
                out.append("SECTION: " + hs)
        for p in div.findall(".//tei:p", ns):
            s = "".join(p.itertext()).strip()
            if s:
                out.append(s)

    # Figure/table captions
    for cap in root.findall(".//tei:figure//tei:figDesc", ns):
        s = "".join(cap.itertext()).strip()
        if s:
            out.append("FIGURE: " + s)
    for cap in root.findall(".//tei:table//tei:head", ns):
        s = "".join(cap.itertext()).strip()
        if s:
            out.append("TABLE: " + s)

    # Table cell text
    # Grobid often stores tables as <table><row><cell>...
    for table in root.findall(".//tei:table", ns):
        rows_txt: List[str] = []
        for row in table.findall(".//tei:row", ns):
            cells = [" ".join("".join(c.itertext()).split()) for c in row.findall(".//tei:cell", ns)]
            cells = [c for c in cells if c]
            if cells:
                rows_txt.append(" | ".join(cells))
        if rows_txt:
            out.append("TABLE_ROWS:\n" + "\n".join(rows_txt))

    txt = "\n\n".join(out)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    if max_chars:
        txt = txt[:max_chars]
    return txt

class DocumentLoader:
    def __init__(self, use_grobid: bool, grobid_url: str, grobid_cache_dir: Optional[Path], max_tei_chars: int = 2_000_000):
        self.use_grobid = use_grobid
        self.grobid = GrobidClient(grobid_url) if use_grobid else None
        self.grobid_cache_dir = grobid_cache_dir
        self.max_tei_chars = max_tei_chars
        self._grobid_is_alive: Optional[bool] = None
        if self.grobid_cache_dir:
            self.grobid_cache_dir.mkdir(parents=True, exist_ok=True)

    def _grobid_available(self) -> bool:
        if not self.use_grobid or self.grobid is None:
            return False
        if self._grobid_is_alive is None:
            self._grobid_is_alive = self.grobid.is_alive()
            if not self._grobid_is_alive:
                LOG.info("GROBID unavailable at %s; using local PDF extractors.", self.grobid.base_url)
        return bool(self._grobid_is_alive)

    def load_pdf(self, pdf_path: Path) -> Document:
        doc_id = pdf_path.stem

        # Grobid-first
        if self._grobid_available():
            tei = None
            cache_path = None
            if self.grobid_cache_dir:
                cache_path = self.grobid_cache_dir / f"{doc_id}.tei.xml"
                if cache_path.exists() and cache_path.stat().st_mtime >= pdf_path.stat().st_mtime:
                    tei = cache_path.read_text(encoding="utf-8", errors="replace")
            if tei is None:
                tei = self.grobid.process_fulltext(pdf_path)
                if tei and cache_path:
                    cache_path.write_text(tei, encoding="utf-8")
            if tei:
                txt = tei_to_text_rich(tei, max_chars=self.max_tei_chars)
                if _alpha_score(txt) >= 0.12:
                    return Document(doc_id=doc_id, path=str(pdf_path), text=txt)
                LOG.warning("Grobid output low-quality for %s; falling back to local PDF extractors.", pdf_path)

        # Local fallback
        txt = self._fallback_pdf_text(pdf_path)
        return Document(doc_id=doc_id, path=str(pdf_path), text=txt)

    def _fallback_pdf_text(self, pdf_path: Path) -> str:
        candidates: List[Tuple[str, str]] = []  # (text, name)

        if fitz is not None:
            try:
                doc = fitz.open(str(pdf_path))
                pages = []
                for i in range(doc.page_count):
                    pages.append(doc.load_page(i).get_text("text") or "")
                full = "\n\n".join(f"--- Page {i+1} ---\n{t}" for i, t in enumerate(pages))
                candidates.append((full, "pymupdf"))
            except Exception as e:
                LOG.debug("PyMuPDF failed: %s", e)

        if pdfplumber is not None:
            try:
                pages = []
                with pdfplumber.open(str(pdf_path)) as pdf:
                    for p in pdf.pages:
                        pages.append(p.extract_text() or "")
                full = "\n\n".join(f"--- Page {i+1} ---\n{t}" for i, t in enumerate(pages))
                candidates.append((full, "pdfplumber"))
            except Exception as e:
                LOG.debug("pdfplumber failed: %s", e)

        if PdfReader is not None:
            try:
                reader = PdfReader(str(pdf_path))
                pages = [(p.extract_text() or "") for p in reader.pages]
                full = "\n\n".join(f"--- Page {i+1} ---\n{t}" for i, t in enumerate(pages))
                candidates.append((full, "pypdf"))
            except Exception as e:
                LOG.debug("pypdf failed: %s", e)

        if not candidates:
            raise RuntimeError("No PDF backend available. Install pymupdf or pdfplumber or pypdf.")

        best = max(candidates, key=lambda x: _alpha_score(x[0]))
        LOG.info("Fallback PDF extraction used %s for %s (alpha=%.3f).", best[1], pdf_path.name, _alpha_score(best[0]))
        return best[0]

def discover_pdfs(pdf_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths = [p for p in pdf_dir.rglob("*.pdf") if p.is_file()]
    else:
        paths = [p for p in pdf_dir.glob("*.pdf") if p.is_file()]
    return sorted(paths)

def chunk_text(doc: Document, chunk_chars: int = 4000, overlap_chars: int = 400) -> List[Chunk]:
    """
    Simple character-based chunker.
    We avoid "traditional NLP" sentence splitting; embeddings handle semantics.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars to avoid infinite loops.")

    t = doc.text or ""
    t = t.replace("\x00", " ")
    if len(t) <= chunk_chars:
        return [Chunk(chunk_id=f"{doc.doc_id}::0", doc_id=doc.doc_id, text=t)]

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    while start < len(t):
        end = min(len(t), start + chunk_chars)
        chunk = t[start:end]
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}::{idx}", doc_id=doc.doc_id, text=chunk))
        idx += 1
        start = end - overlap_chars
        if start < 0:
            start = 0
        if end == len(t):
            break
    return chunks
