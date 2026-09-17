"""
Your notes: PDFs and photos uploaded in the page, read once, then opened by the AI with tools.

On upload a note is saved under NOTES_DIR/<id>/ and read in the background, one note at a time:
  * PDF pages with real text are used as they are. Scanned pages and photos go to an image model once (NVIDIA NIM's
    NVIDIA_VISION_MODEL, Groq's NOTES_VISION_MODEL as backup) and come back as markdown.
  * The text is split into numbered sections (headings, else pages) and indexed in the local search store, and a
    model writes a title, subject, topics, summary and suggested questions.
Files and everything read from them stay on this computer; only page images and text go to the AI service, at upload.

The AI sees a short "notes shelf" in its prompt and uses NotesTools to open a note, read a section, or search all
notes. The section being taught stays in its prompt (NotesTools.focus), so follow-ups need no new lookup.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import queue
import random
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

import config
from tutor import providers, web
from tutor.rag import split

log = logging.getLogger("notes")

ROOT = Path(__file__).resolve().parent.parent
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
PDF_TYPES = {".pdf"}
MIN_PAGE_TEXT = 40          # a PDF page with less real text than this is treated as a scan
VISION_WIDTH = 1400         # page images sent to the image model are at most this wide
THUMB_WIDTH = 320
MAX_SECTION_CHARS = 3500    # longer sections are split into parts, so one read fits in the prompt
MAX_SEARCH_DISTANCE = 0.8
_PAGE_MARK = re.compile(r"^<!-- page (\d+) -->$", re.M)
_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```(?:markdown|md)?\s*|\s*```$")
_TITLE_LINE = re.compile(r"^[A-Z0-9][^\n.:;!?]{1,59}$")   # short, capitalised, no sentence punctuation
_BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+")

VISION_PROMPT = (
    "These are a student's study notes (a photo or a scanned page). Transcribe them into clean markdown: start every "
    "title or topic heading with ## (even if it's only underlined, boxed or written larger), keep bullet points and "
    "the original wording; write math as LaTeX between single dollar "
    "signs; describe any diagram or figure in one line starting with 'Diagram:'. Output only the transcription. If "
    "the page is blank or unreadable, output exactly: Unreadable page")

FILLERS = {
    "open_note": ["Let me pull up your notes.", "Okay, opening your notes now.", "One sec, let me look at your notes.",
                  "Let me check your notes section."],
    "read_note_section": ["Let me read that part of your notes.", "Okay, looking at that section.",
                          "One sec, reading that bit."],
    "search_notes": ["Let me search your notes.", "Let me look through your notes for that.",
                     "One sec, checking your notes."],
    "list_note_links": ["Let me look at the links in your notes.", "Let me see what links are in there."],
    "open_link": ["Let me open {site}.", "Okay, checking {site} now.", "One sec, looking at {site}."],
}
# How each step shows up in the page's thinking timeline: (while running, when done).
STEP_WORDS = {
    "open_note": ("Opening", "Opened"),
    "read_note_section": ("Reading", "Read"),
    "search_notes": ("Searching your notes for", "Searched your notes for"),
    "list_note_links": ("Looking for links in", "Checked the links in"),
    "open_link": ("Visiting", "Visited"),
}

TOOLS = [
    {"type": "function", "function": {
        "name": "open_note",
        "description": ("Open one of the user's uploaded notes to see its summary, topics and numbered sections. Use it "
                        "whenever they mention notes they uploaded or ask you to look at their notes."),
        "parameters": {"type": "object", "properties": {
            "note_id": {"type": "string", "description": "the note's id from the notes shelf"}},
            "required": ["note_id"]}}},
    {"type": "function", "function": {
        "name": "read_note_section",
        "description": "Read the full text of one numbered section of a note, so you can teach it.",
        "parameters": {"type": "object", "properties": {
            "note_id": {"type": "string"},
            "section": {"type": "integer", "description": "section number from open_note, starting at 1"}},
            "required": ["note_id", "section"]}}},
    {"type": "function", "function": {
        "name": "search_notes",
        "description": "Search all of the user's notes for a topic, term or phrase.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_note_links",
        "description": "List the web links in one of the user's notes, like the GitHub, portfolio or project links on a resume.",
        "parameters": {"type": "object", "properties": {"note_id": {"type": "string"}}, "required": ["note_id"]}}},
    {"type": "function", "function": {
        "name": "open_link",
        "description": ("Open and read a public web page. Allowed: links in the user's notes, links the user gave you, "
                        "and links found on pages you already opened (for example a repository on a GitHub profile)."),
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
]


class NoteError(Exception):
    """A problem worth showing to the user as is."""


class _Deleted(Exception):
    pass


def _clean(value, limit: int) -> str:
    """Plain one-line text; square brackets would break the app's [SYSTEM: ...] notes."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.replace("[", "(").replace("]", ")")).strip()[:limit]


def _ago(t: float) -> str:
    s = time.time() - t
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)} minute{'s' if s >= 120 else ''} ago"
    if s < 86400:
        return f"{int(s // 3600)} hour{'s' if s >= 7200 else ''} ago"
    if s < 172800:
        return "yesterday"
    return time.strftime("on %b %d", time.localtime(t))


def _plain(text: str) -> str:
    return _PAGE_MARK.sub(lambda m: f"(page {m.group(1)})", text).strip()


def _page_at(body: str, offset: int) -> int:
    pages = [int(m.group(1)) for m in _PAGE_MARK.finditer(body, 0, offset + 1)]
    return pages[-1] if pages else 1


def _title_lines(body: str) -> list[tuple[int, str]]:
    """Unmarked headings: a short title-like line directly followed by bullet points."""
    heads, offset = [], 0
    lines = body.split("\n")
    for i, line in enumerate(lines):
        text = line.strip()
        following = next((l for l in lines[i + 1:] if l.strip()), "")
        if _TITLE_LINE.match(text) and not _BULLET.match(line) and _BULLET.match(following):
            heads.append((offset, text[:80]))
        offset += len(line) + 1
    return heads


def build_sections(body: str) -> list[dict]:
    """Numbered sections for walking through a note: its headings (marked or not), else one per page.
    Long sections become parts."""
    marked = [(m.start(), m.group(1).strip()[:80]) for m in re.finditer(r"^#{1,3}\s+(\S.*)$", body, re.M)]
    single_page = len(_PAGE_MARK.findall(body)) <= 1
    heads = marked if len(marked) >= 2 else _title_lines(body)
    if len(heads) < 2:
        heads = marked if marked and single_page else []
    if not heads:
        heads = [(m.start(), f"Page {m.group(1)}") for m in _PAGE_MARK.finditer(body)]
    elif len(_plain(body[:heads[0][0]])) > 80:
        heads.insert(0, (0, "Introduction"))
    sections = []
    for i, (start, title) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(body)
        parts = max(1, math.ceil((end - start) / MAX_SECTION_CHARS))
        cursor = start
        for p in range(parts):
            stop = end if p == parts - 1 else start + (end - start) * (p + 1) // parts
            if p < parts - 1:
                newline = body.rfind("\n", cursor + MAX_SECTION_CHARS // 2, stop)
                stop = newline if newline > cursor else stop
            sections.append({"title": title if parts == 1 else f"{title} (part {p + 1})",
                             "page": _page_at(body, cursor), "start": cursor, "end": stop})
            cursor = stop
    # A heading with (almost) nothing under it, like a lecture title, joins the next section, so the AI never has
    # to "teach" a section that holds no content.
    merged: list[dict] = []
    carry = None
    for section in sections:
        if carry is not None:
            section = {**section, "start": carry["start"], "page": carry["page"]}
            carry = None
        content = body[section["start"]:section["end"]]
        own = content.split("\n", 1)[1] if "\n" in content else ""
        if len(_plain(own)) < 3 and section is not sections[-1]:
            carry = section
            continue
        merged.append(section)
    if carry is not None:
        merged.append(carry)
    return [s for s in merged if len(_plain(body[s["start"]:s["end"]])) > 15]


@dataclass
class Note:
    id: str
    filename: str
    kind: str                       # "pdf" | "image"
    created_at: float
    status: str = "queued"          # queued | reading | ready | failed
    progress: str = ""              # e.g. "Reading page 2 of 6"
    pages: int = 0
    title: str = ""
    subject: str = ""
    topics: list[str] = field(default_factory=list)
    summary: str = ""
    questions: list[str] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)   # {"title", "page", "start", "end"} into text.md
    links: list[dict] = field(default_factory=list)      # {"url", "label", "page"}
    error: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sections"] = [{"title": s["title"], "page": s["page"]} for s in self.sections]
        data["added"] = _ago(self.created_at)
        return data


class NotesLibrary:
    def __init__(self, retriever=None, on_change: Callable[[], None] | None = None):
        load_dotenv(ROOT / ".env")
        self.dir = ROOT / config.NOTES_DIR
        self.dir.mkdir(exist_ok=True)
        self.on_change = on_change
        self._retriever = retriever
        self._col = None
        if retriever is not None:
            try:
                self._col = retriever.client.get_or_create_collection(
                    "notes", metadata={"hnsw:space": "cosine"}, embedding_function=None)
            except Exception:
                log.exception("notes search index unavailable; searching note text directly")
        self._lock = threading.Lock()
        self._notes: dict[str, Note] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._load()
        threading.Thread(target=self._worker, name="notes", daemon=True).start()

    # ------------------------------------------------------------------ public

    def add(self, filename: str, data: bytes) -> Note:
        name = _clean(Path(filename or "").name, 120) or "notes"
        ext = Path(name).suffix.lower()
        if ext not in IMAGE_TYPES | PDF_TYPES:
            raise NoteError("Only PDFs and images (PNG, JPG, WebP) can be added as notes.")
        if not data:
            raise NoteError("That file is empty.")
        if len(data) > config.NOTES_MAX_MB * 1024 * 1024:
            raise NoteError(f"That file is bigger than {config.NOTES_MAX_MB} MB.")
        note = Note(id=uuid.uuid4().hex[:6], filename=name, kind="pdf" if ext in PDF_TYPES else "image",
                    created_at=time.time(), progress="Waiting to be read")
        folder = self.dir / note.id
        folder.mkdir(parents=True)
        (folder / f"original{ext}").write_bytes(data)
        with self._lock:
            self._notes[note.id] = note
        self._save(note)
        self._queue.put(note.id)
        log.info("note %s added: %s (%d KB)", note.id, name, len(data) // 1024)
        self._changed()
        return note

    def list(self) -> list[Note]:
        with self._lock:
            return sorted(self._notes.values(), key=lambda n: n.created_at, reverse=True)

    def get(self, note_id: str | None) -> Note | None:
        with self._lock:
            return self._notes.get((note_id or "").strip())

    def find(self, ref: str | None) -> Note | None:
        """A note by id, or by a word from its title or file name (the model sometimes passes those)."""
        note = self.get(ref)
        if note or not ref:
            return note
        needle = ref.lower().strip()
        return next((n for n in self.list() if needle in n.title.lower() or needle in n.filename.lower()), None)

    def delete(self, note_id: str) -> bool:
        with self._lock:
            note = self._notes.pop(note_id, None)
        if note is None:
            return False
        shutil.rmtree(self.dir / note_id, ignore_errors=True)
        if self._col is not None:
            try:
                self._col.delete(where={"note_id": note_id})
            except Exception:
                log.exception("could not remove note %s from the search index", note_id)
        log.info("note %s deleted (%s)", note_id, note.filename)
        self._changed()
        return True

    def public(self, note: Note) -> dict:
        """What the page shows for a note, including whether its thumbnail exists yet."""
        return {**note.to_dict(), "thumb": (self.dir / note.id / "pages" / "1.jpg").is_file()}

    def page_image(self, note_id: str, page: int) -> Path | None:
        path = self.dir / note_id / "pages" / f"{int(page)}.jpg"
        return path if self.get(note_id) and path.is_file() else None

    def link_keys(self) -> set[str]:
        return {web.key(link["url"]) for n in self.list() for link in n.links}

    def section_text(self, note: Note, index: int) -> str:
        section = note.sections[index]
        body = (self.dir / note.id / "text.md").read_text()
        return _plain(body[section["start"]:section["end"]])

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Best-matching passages across all ready notes: note, section number, page and text."""
        ready = {n.id: n for n in self.list() if n.status == "ready"}
        if not ready or not query.strip():
            return []
        if self._col is not None and self._col.count():
            res = self._col.query(query_embeddings=self._retriever.embed([query]), n_results=min(k, self._col.count()),
                                  include=["documents", "metadatas", "distances"])
            hits = []
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                note = ready.get(meta["note_id"])
                if note and dist <= MAX_SEARCH_DISTANCE:
                    hits.append({"note": note, "section": meta["section"], "page": meta["page"], "text": _plain(doc)})
            return hits
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        hits = []
        for note in ready.values():
            for i in range(len(note.sections)):
                text = self.section_text(note, i)
                score = sum(text.lower().count(w) for w in words)
                if score:
                    hits.append((score, {"note": note, "section": i, "page": note.sections[i]["page"], "text": text}))
        return [h for _, h in sorted(hits, key=lambda x: -x[0])[:k]]

    def shelf(self) -> str:
        """The AI's view of the notes section, newest first."""
        lines = []
        for n in self.list()[:config.NOTES_ON_SHELF]:
            if n.status == "ready":
                lines.append(f'- {n.id}: "{n.title}" ({n.subject or "notes"}, {n.pages} page{"s" if n.pages != 1 else ""}, '
                             f'added {_ago(n.created_at)}; {len(n.sections)} sections; topics: '
                             f'{", ".join(n.topics) or "not listed"})')
            elif n.status == "failed":
                lines.append(f'- {n.id}: "{n.filename}" (added {_ago(n.created_at)}, could not be read)')
            else:
                lines.append(f'- {n.id}: "{n.filename}" (added {_ago(n.created_at)}, still being read: '
                             f'{n.progress or "waiting"})')
        return "\n".join(lines)

    # ------------------------------------------------------------------ internal

    def _changed(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception:
                log.exception("notes change handler failed (continuing)")

    def _save(self, note: Note) -> None:
        folder = self.dir / note.id
        if folder.is_dir():
            (folder / "note.json").write_text(json.dumps(asdict(note), indent=1))

    def _update(self, note: Note, **changes) -> None:
        if self.get(note.id) is None:
            raise _Deleted()
        for key, value in changes.items():
            setattr(note, key, value)
        self._save(note)
        self._changed()

    def _load(self) -> None:
        for path in sorted(self.dir.glob("*/note.json")):
            try:
                raw = json.loads(path.read_text())
                note = Note(**raw)
            except Exception:
                log.exception("skipping unreadable note %s", path.parent.name)
                continue
            self._notes[note.id] = note
            if note.status == "ready" and "links" not in raw:   # read before notes had links
                try:
                    note.links = self._find_links(note)
                    self._save(note)
                    log.info("note %s: found %d links", note.id, len(note.links))
                except Exception:
                    log.exception("could not collect links for note %s", note.id)
            if note.status in ("queued", "reading"):
                note.status, note.progress = "queued", "Waiting to be read"
                self._queue.put(note.id)   # the app closed while it was being read
            elif note.status == "ready" and self._col is not None:
                try:
                    if not self._col.get(where={"note_id": note.id}, limit=1)["ids"]:
                        body = (path.parent / "text.md").read_text()
                        self._index(note, body)   # the search index was cleared
                except Exception:
                    log.exception("could not re-index note %s", note.id)
        if self._notes:
            log.info("notes: %d on the shelf", len(self._notes))

    def _worker(self) -> None:
        while True:
            note = self.get(self._queue.get())
            if note is None:
                continue
            t0 = time.perf_counter()
            try:
                self._read(note)
                log.info("note %s ready in %.1fs: %r, %d pages, %d sections", note.id, time.perf_counter() - t0,
                         note.title, note.pages, len(note.sections))
            except _Deleted:
                log.info("note %s was deleted while being read", note.id)
            except Exception as e:
                log.exception("READING NOTE %s FAILED", note.id)
                message = str(e) if isinstance(e, NoteError) else "Something went wrong while reading this file."
                try:
                    self._update(note, status="failed", progress="", error=message)
                except _Deleted:
                    pass

    def _read(self, note: Note) -> None:
        folder = self.dir / note.id
        original = next(folder.glob("original.*"))
        pages_dir = folder / "pages"
        pages_dir.mkdir(exist_ok=True)
        self._update(note, status="reading", progress="Opening the file", error="")
        texts: list[str] = []
        if note.kind == "pdf":
            import pypdfium2 as pdfium
            try:
                pdf = pdfium.PdfDocument(str(original))
            except Exception as e:
                raise NoteError("This PDF couldn't be opened (it may be damaged or password protected).") from e
            try:
                count = min(len(pdf), config.NOTES_MAX_PAGES)
                self._update(note, pages=count)
                for i in range(count):
                    self._update(note, progress=f"Reading page {i + 1} of {count}")
                    page = pdf[i]
                    text = page.get_textpage().get_text_range().replace("\r\n", "\n").strip()
                    image = page.render(scale=min(3.0, VISION_WIDTH / max(1.0, page.get_width()))).to_pil()
                    image = image.convert("RGB")
                    self._thumb(image, pages_dir / f"{i + 1}.jpg")
                    if len(text) < MIN_PAGE_TEXT:
                        text = self._transcribe(note, image)
                    texts.append(text)
            finally:
                pdf.close()
        else:
            from PIL import Image, ImageOps
            try:
                with Image.open(original) as im:
                    image = ImageOps.exif_transpose(im).convert("RGB")
            except Exception as e:
                raise NoteError("This image couldn't be opened.") from e
            self._update(note, pages=1, progress="Reading your notes")
            self._thumb(image, pages_dir / "1.jpg")
            texts.append(self._transcribe(note, image))

        if not any(len(t) > 10 and t != "Unreadable page" for t in texts):
            raise NoteError("I couldn't find any readable text in this file.")
        body = "\n\n".join(f"<!-- page {i} -->\n{t}" for i, t in enumerate(texts, 1))
        (folder / "text.md").write_text(body)
        sections = build_sections(body)
        self._update(note, sections=sections, links=self._find_links(note, texts), progress="Picking out the topics")
        self._index(note, body)
        self._update(note, status="ready", progress="", **self._describe(note, body))

    def _find_links(self, note: Note, texts: list[str] | None = None) -> list[dict]:
        """Web links in a note: clickable PDF links plus web addresses written in the text."""
        folder = self.dir / note.id
        found: list[tuple[str, int]] = []
        if note.kind == "pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(next(folder.glob("original.*"))))
                for number, page in enumerate(reader.pages[:config.NOTES_MAX_PAGES], 1):
                    for annot in page.get("/Annots") or []:
                        action = annot.get_object().get("/A")
                        uri = action.get_object().get("/URI") if action is not None else None
                        if uri:
                            found.append((uri.decode(errors="ignore") if isinstance(uri, bytes) else str(uri), number))
            except Exception:
                log.exception("note %s: could not read PDF links", note.id)
        if texts is None:
            body = (folder / "text.md").read_text()
            marks = list(_PAGE_MARK.finditer(body))
            texts = [body[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(body)] for i, m in enumerate(marks)]
        for number, text in enumerate(texts, 1):
            found += [(url, number) for url in web.find_urls(text)]
        links, seen = [], set()
        for url, number in found:
            url = url.strip()
            if web._SCHEME.match(url) and not url.lower().startswith(("http://", "https://")):
                continue   # mailto:, tel: and friends
            url = web.normalize(url)
            if "." not in web.host(url) or web.key(url) in seen:
                continue
            seen.add(web.key(url))
            links.append({"url": url, "label": web.label(url), "page": number})
        return links[:40]

    @staticmethod
    def _thumb(image, path: Path) -> None:
        thumb = image.copy()
        thumb.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 2))
        thumb.save(path, "JPEG", quality=80)

    def _ai_call(self, note: Note, kind: str, build: Callable[[providers.Route], dict]):
        """One request on the first provider that works for `kind` (NVIDIA NIM, then Groq). The last provider
        waits out rate limits, since a long scanned PDF can hit them. Returns (route, response)."""
        routes = providers.routes(kind)
        if not routes:
            raise NoteError("No AI service is set up to read notes (add NVIDIA_API_KEY or GROQ_API_KEY to .env).")
        for i, route in enumerate(routes):
            last = i == len(routes) - 1
            for attempt in range(6 if last else 1):
                try:
                    api = route.client.with_options(timeout=90)
                    return route, api.chat.completions.create(model=route.model, **build(route))
                except Exception as e:
                    if last and providers.status(e) == 429 and attempt < 5:
                        progress = note.progress
                        self._update(note, progress=f"Waiting for {route.provider.upper()} (rate limit)…")
                        time.sleep(10 * (attempt + 1))
                        self._update(note, progress=progress)
                        continue
                    if last:
                        raise
                    log.warning("note %s: %s failed (%s: %s); trying %s", note.id, route.name,
                                providers.status(e) or type(e).__name__, str(e)[:120], routes[i + 1].name)
                    break
        raise NoteError("No AI service could read this note.")

    def _transcribe(self, note: Note, image) -> str:
        img = image.copy()
        img.thumbnail((VISION_WIDTH, VISION_WIDTH * 2))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        t0 = time.perf_counter()
        route, r = self._ai_call(note, "vision", lambda route: {"temperature": 0, "max_tokens": 2500, "messages": [
            {"role": "user", "content": [{"type": "text", "text": VISION_PROMPT},
                                         {"type": "image_url", "image_url": {"url": url}}]}]})
        text = _FENCE.sub("", _THINK.sub("", r.choices[0].message.content or "")).strip()
        log.info("note %s: page read by %s in %.1fs (%d chars)", note.id, route.name, time.perf_counter() - t0,
                 len(text))
        return text

    def _index(self, note: Note, body: str) -> None:
        if self._col is None:
            return
        self._col.delete(where={"note_id": note.id})
        docs, metas = [], []
        for i, section in enumerate(note.sections or build_sections(body)):
            for _, chunk in split(body[section["start"]:section["end"]]):
                docs.append(chunk)
                metas.append({"note_id": note.id, "section": i, "page": section["page"], "title": section["title"]})
        if docs:
            self._col.add(ids=[f"{note.id}::{k}" for k in range(len(docs))], documents=docs, metadatas=metas,
                          embeddings=self._retriever.embed([f"{m['title']}\n{d}" for m, d in zip(metas, docs)]))

    def _describe(self, note: Note, body: str) -> dict:
        """Title, subject, topics, summary and suggested questions, written by a small model."""
        fallback_title = next((s["title"] for s in note.sections if not s["title"].startswith("Page ")),
                              Path(note.filename).stem.replace("_", " "))
        outline = "\n".join(f"{i + 1}. {s['title']}" for i, s in enumerate(note.sections[:30]))
        prompt = (f"File name: {note.filename}\nSections:\n{outline}\n\nNotes:\n{_plain(body)[:12000]}\n\n"
                  'Return a JSON object with: "title" (a short, human title naming the subject and the main topic, '
                  "based only on these notes; for example a page about cell division in biology would be "
                  '"Biology: Cell Division"), '
                  '"subject" (one to three words), "topics" (3 to 6 short topics), "summary" (two plain sentences), '
                  '"questions" (3 short questions a student might ask about these notes).')
        def build(route: providers.Route) -> dict:
            request = {"temperature": 0.2, "max_tokens": 1500, "messages": [
                {"role": "system", "content": "You organise a student's study notes. Reply with one JSON object only."},
                {"role": "user", "content": prompt}]}
            request.update(providers.reasoning_options(route, "low"))
            if route.provider == "groq":
                request["response_format"] = {"type": "json_object"}
            return request

        try:
            route, r = self._ai_call(note, "summary", build)
            content = _THINK.sub("", r.choices[0].message.content or "")
            match = re.search(r"\{.*\}", content, re.S)   # some models wrap the JSON in prose or a code block
            data = json.loads(match.group(0)) if match else {}
            log.info("note %s: summary written by %s", note.id, route.name)
        except Exception:
            log.exception("note %s: could not write a summary (using the file name)", note.id)
            data = {}
        title = re.sub(r"^(subject|title)\s*:\s*", "", _clean(data.get("title"), 90), flags=re.I)
        return {"title": title or _clean(fallback_title, 90),
                "subject": _clean(data.get("subject"), 40),
                "topics": [t for t in (_clean(x, 40) for x in data.get("topics") or []) if t][:6],
                "summary": _clean(data.get("summary"), 400),
                "questions": [q for q in (_clean(x, 120) for x in data.get("questions") or []) if q][:3]}


class NotesTools:
    """Runs the AI's notes tools, remembers what it's teaching from, and tells the page."""

    def __init__(self, library: NotesLibrary, on_focus: Callable[[dict | None, bool], None] | None = None):
        self.library = library
        self.on_focus = on_focus
        self.focus: dict | None = None   # note being taught: id, title, section, sections, section title, page, text
        self.page: dict | None = None    # web page opened most recently: url, site, title, text
        self._user_hosts: set[str] = set()   # sites the user mentioned or linked this session
        self._page_keys: set[str] = set()    # links seen on pages the AI opened
        self._last_filler = ""
        self._fillers = 0
        self._outcome: tuple[bool, str] = (True, "")   # how the last tool went, for its timeline label

    @property
    def available(self) -> bool:
        return bool(self.library.list()) or bool(self._user_hosts)

    def begin_turn(self) -> None:
        self._fillers = 0

    def clear(self) -> None:
        self.focus = None
        self.page = None
        self._user_hosts.clear()
        self._page_keys.clear()
        self._notify(False)

    def remember_user_text(self, text: str) -> None:
        """Links and site names the user gives may be opened."""
        self._user_hosts |= {web.host(u) for u in web.find_urls(text)} | web.find_hosts(text)

    def announce(self, name: str, args: dict) -> str | None:
        """A short line to say while a tool runs, so there's no silent pause (at most two per turn)."""
        if name not in FILLERS or self._fillers >= 2 or (self._fillers and name != "open_link"):
            return None
        site = web.host(str(args.get("url") or "")) if name == "open_link" else ""
        options = [f for f in FILLERS[name] if f != self._last_filler] or FILLERS[name]
        self._last_filler = random.choice(options)
        self._fillers += 1
        return f"[thoughtful] {self._last_filler.replace('{site}', site or 'that page')} "

    def describe(self, name: str, args: dict, done: bool = False) -> tuple[str, bool]:
        """(label, ok) for the page's thinking timeline."""
        ok, detail = self._outcome if done else (True, "")
        running, finished = STEP_WORDS.get(name, (name, name))
        verb = finished if done else running
        if name == "open_link":
            target = web.label(str(args.get("url") or ""))
            if done and not ok:
                return f"Couldn't open {target}: {detail}", False
            return f"{verb} {target}" + (f" · {detail}" if done and detail else ""), ok
        if name == "search_notes":
            return f"{verb} “{args.get('query', '')}”" + (f" · {detail}" if done and detail else ""), ok
        note = self.library.find(str(args.get("note_id") or ""))
        title = note.title if note else str(args.get("note_id") or "a note")
        if name == "read_note_section":
            section = args.get("section")
            name_part = ""
            if note and str(section).isdigit() and 1 <= int(section) <= len(note.sections):
                name_part = f": {note.sections[int(section) - 1]['title']}"
            return f"{verb} section {section} of “{title}”{name_part}", ok
        if name == "list_note_links" and done and ok:
            return f"Found {detail} in “{title}”", ok
        return f"{verb} “{title}”", ok

    def run(self, name: str, args: dict) -> str:
        log.info("notes tool %s(%s)", name, json.dumps(args)[:120])
        self._outcome = (True, "")
        try:
            if name == "open_note":
                return self._open(args.get("note_id"))
            if name == "read_note_section":
                return self._read(args.get("note_id"), args.get("section"))
            if name == "search_notes":
                return self._search(str(args.get("query") or ""))
            if name == "list_note_links":
                return self._links(args.get("note_id"))
            if name == "open_link":
                return self._open_link(str(args.get("url") or ""))
            return self._fail(f"Unknown tool {name}.", "unknown tool")
        except Exception:
            log.exception("notes tool %s failed", name)
            return self._fail("That couldn't be read just now. Tell the user briefly and offer to try again.",
                              "something went wrong")

    def prompt(self) -> str:
        """The note being taught and the page opened last, for the system prompt."""
        parts = []
        f = self.focus
        if f:
            head = f'Note you\'re working from: "{f["title"]}" (id {f["note_id"]}), {f["sections"]} sections.'
            if f.get("text") is None:
                parts.append(head + " You've found it; read a section before teaching details.")
            else:
                more = (f" After this, continue with section {f['section'] + 1} when they're ready."
                        if f["section"] < f["sections"] else " This is its last section.")
                parts.append(f'{head} Current section {f["section"]}: {f["section_title"]} (page {f["page"]}).{more} '
                             f'Its text:\n<<<\n{f["text"]}\n>>>')
        if self.page:
            p = self.page
            parts.append(f'Web page you opened most recently: "{p["title"]}" ({p["url"]}). Its text:\n<<<\n'
                         f'{p["text"]}\n>>>')
        return "\n".join(parts)

    def _fail(self, message: str, detail: str) -> str:
        self._outcome = (False, detail)
        return message

    def _notify(self, reading: bool, focus: dict | None = None) -> None:
        if self.on_focus is not None:
            try:
                self.on_focus(focus or self.public_focus(), reading)
            except Exception:
                log.exception("notes focus handler failed (continuing)")

    def public_focus(self) -> dict | None:
        f = self.focus
        if not f:
            return None
        return {k: f[k] for k in ("note_id", "title", "section", "sections", "section_title", "page")}

    def _note(self, ref) -> Note | str:
        note = self.library.find(str(ref or ""))
        if note is None:
            shelf = self.library.shelf() or "(empty)"
            return self._fail(f"There is no note {ref!r}. The notes shelf is:\n{shelf}", "no such note")
        if note.status == "failed":
            return self._fail(f'The note "{note.filename}" could not be read: {note.error} Tell the user.',
                              "it couldn't be read")
        if note.status != "ready":
            return self._fail(f'The note "{note.filename}" is still being read ({note.progress or "waiting"}). Tell '
                              "the user it will be ready in a moment and offer to start as soon as it is.",
                              "still being read")
        return note

    def _links(self, ref) -> str:
        note = self._note(ref)
        if isinstance(note, str):
            return note
        self._outcome = (True, f"{len(note.links)} link{'s' if len(note.links) != 1 else ''}")
        if not note.links:
            return f'"{note.title}" has no web links.'
        return f'Links in "{note.title}":\n' + "\n".join(f"- {l['url']} (page {l['page']})" for l in note.links)

    def _open_link(self, url: str) -> str:
        if not url.strip():
            return self._fail("Give the link to open.", "no link given")
        url = web.normalize(url)
        k, site = web.key(url), web.host(url)
        note_keys = self.library.link_keys()
        allowed = (k in note_keys or any(k.startswith(n + "/") for n in note_keys)
                   or site in self._user_hosts or k in self._page_keys)
        if not allowed:
            return self._fail("You may only open links from the user's notes, links they gave you, or links on pages "
                              "you already opened. Ask them to share this link first.", "not a link you were given")
        self._notify(True, {"kind": "link", "url": url, "site": site, "title": web.label(url)})
        try:
            page = web.open_page(url)
        except web.WebError as e:
            self._notify(False, {"kind": "link", "url": url, "site": site, "title": web.label(url), "error": str(e)})
            return self._fail(f"Couldn't open {url}: {e} Tell the user briefly.", str(e).rstrip("."))
        self._page_keys |= {web.key(link) for _, link in page.links[:80]}
        self.page = {"url": page.url, "site": page.site, "title": page.title, "text": page.text[:config.NOTES_FOCUS_CHARS]}
        self._notify(False, {"kind": "link", "url": page.url, "site": page.site, "title": page.title})
        self._outcome = (True, page.title if page.title != page.site else "")
        links = "\n".join(f"- {text}: {link}" for text, link in page.links[:25])
        return f"Page: {page.title}\nURL: {page.url}\n\n{page.text}" + (f"\n\nLinks on this page:\n{links}" if links else "")

    def _open(self, ref) -> str:
        note = self._note(ref)
        if isinstance(note, str):
            return note
        # The overview comes with section 1, so explaining a note needs one lookup instead of two.
        first = note.sections[0]
        text = self.library.section_text(note, 0)
        self.focus = {"note_id": note.id, "title": note.title, "section": 1, "sections": len(note.sections),
                      "section_title": first["title"], "page": first["page"], "text": text[:config.NOTES_FOCUS_CHARS]}
        self._notify(True)
        outline = "\n".join(f"{i}. {s['title']} (page {s['page']})" for i, s in enumerate(note.sections[:40], 1))
        return (f'Note {note.id}: "{note.title}" ({note.subject or "notes"}, {note.pages} pages, added '
                f"{_ago(note.created_at)})\nSummary: {note.summary or 'not available'}\n"
                f"Topics: {', '.join(note.topics) or 'not listed'}\nSections:\n{outline}\n\n"
                f"Section 1 of {len(note.sections)}: {first['title']} (page {first['page']})\n{text}\n\n"
                "To explain the note, teach section 1 now. If they asked about a specific part, read that section "
                "instead.")

    def _read(self, ref, section) -> str:
        note = self._note(ref)
        if isinstance(note, str):
            return note
        try:
            number = int(section)
        except (TypeError, ValueError):
            number = 1
        if not 1 <= number <= len(note.sections):
            return f'"{note.title}" has sections 1 to {len(note.sections)}; there is no section {section}.'
        info = note.sections[number - 1]
        text = self.library.section_text(note, number - 1)
        self.focus = {"note_id": note.id, "title": note.title, "section": number, "sections": len(note.sections),
                      "section_title": info["title"], "page": info["page"], "text": text[:config.NOTES_FOCUS_CHARS]}
        self._notify(True)
        return (f'Note "{note.title}", section {number} of {len(note.sections)}: {info["title"]} (page '
                f'{info["page"]})\n\n{text}')

    def _search(self, query: str) -> str:
        hits = self.library.search(query)
        self._outcome = (True, f"{len(hits)} match{'es' if len(hits) != 1 else ''}")
        if not hits:
            return f"Nothing in the user's notes matches {query!r}.\nNotes shelf:\n{self.library.shelf() or '(empty)'}"
        top = hits[0]
        self.focus = {"note_id": top["note"].id, "title": top["note"].title, "section": top["section"] + 1,
                      "sections": len(top["note"].sections), "page": top["page"], "text": None,
                      "section_title": top["note"].sections[top["section"]]["title"]}
        self._notify(True)
        return "\n\n".join(f'From note {h["note"].id} "{h["note"].title}", section {h["section"] + 1} '
                           f'(page {h["page"]}):\n{h["text"][:900]}' for h in hits)
