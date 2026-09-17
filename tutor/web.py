"""
Opening web pages for the AI: links in your notes, links you give it, and links on pages it already opened.

Public http(s) pages only. Every redirect hop is checked, and addresses on this computer or the local network are
refused. Pages are read as text (HTML, plain text, JSON or PDF) and capped in size. GitHub profiles and repositories
are read through GitHub's public API and README, which gives far better text than the page itself.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

log = logging.getLogger("web")

TIMEOUT_S = 8.0
MAX_BYTES = 2_000_000
MAX_TEXT = 6000
MAX_REDIRECTS = 4
USER_AGENT = "Mozilla/5.0 (compatible; AI-Tutor/1.0; local study assistant)"
_URL = re.compile(r"(?:https?://|www\.)[^\s<>()\"'\]]+"
                  r"|\b(?:github\.com|gitlab\.com|linkedin\.com|behance\.net|medium\.com|kaggle\.com)/[^\s<>()\"'\]]+",
                  re.I)
_HOST = re.compile(r"\b((?:[a-z0-9-]+\.)+(?:com|org|net|io|dev|ai|app|me|in|co|edu|gov|xyz|tech|site|page|blog))\b", re.I)
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.I)
_GITHUB_RESERVED = {"about", "features", "pricing", "login", "join", "explore", "topics", "orgs", "settings",
                    "marketplace", "sponsors", "search", "trending", "collections", "notifications"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCKS = {"p", "div", "section", "article", "li", "br", "tr", "pre", "blockquote", "header", "footer", "ul", "ol",
           "table", "main", "nav", "aside", "dd", "dt", "figcaption"}
_SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "canvas"}


class WebError(Exception):
    """A reason a page couldn't be read, worded for the user."""


@dataclass
class Page:
    url: str
    title: str
    text: str
    links: list[tuple[str, str]] = field(default_factory=list)   # (link text, url)

    @property
    def site(self) -> str:
        return host(self.url)


def normalize(url: str) -> str:
    url = url.strip().rstrip(".,;:!?)]}'\"")
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


def host(url: str) -> str:
    return (urlparse(normalize(url)).hostname or "").lower().removeprefix("www.")


def key(url: str) -> str:
    """Comparable form of a link: host and path, without scheme, www or trailing slash."""
    p = urlparse(normalize(url))
    return f"{(p.hostname or '').lower().removeprefix('www.')}{p.path.rstrip('/')}".lower()


def label(url: str) -> str:
    k = key(url)
    return k if len(k) <= 48 else k[:45] + "…"


def find_urls(text: str) -> list[str]:
    urls, seen = [], set()
    for m in _URL.finditer(text or ""):
        raw = m.group(0)
        if _SCHEME.match(raw) and not raw.lower().startswith(("http://", "https://")):
            continue
        url = normalize(raw)
        if "." not in host(url) or key(url) in seen:
            continue
        seen.add(key(url))
        urls.append(url)
    return urls


def find_hosts(text: str) -> set[str]:
    """Website names mentioned in plain words, like "sneha.dev" or "GitHub.com"."""
    return {m.group(1).lower().removeprefix("www.") for m in _HOST.finditer(text or "")}


def _check(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise WebError("Only http and https links can be opened.")
    if p.port not in (None, 80, 443):
        raise WebError("That link uses an unusual port, so I won't open it.")
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror as e:
        raise WebError(f"{p.hostname} couldn't be found.") from e
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if not address.is_global:
            raise WebError("That link points to this computer or a private network, so I won't open it.")


def _get(client: httpx.Client, url: str, accept: str | None = None) -> tuple[str, str, bytes]:
    """GET with every redirect hop checked. Returns (final url, content type, body)."""
    for _ in range(MAX_REDIRECTS + 1):
        _check(url)
        headers = {"Accept": accept} if accept else {}
        with client.stream("GET", url, headers=headers) as r:
            if r.is_redirect:
                url = urljoin(url, r.headers.get("location", ""))
                continue
            site = host(url)
            if r.status_code in (401, 403, 429, 451, 999):
                raise WebError(f"{site} didn't let me read that page (it blocks automated visits or needs a login).")
            if r.status_code == 404:
                raise WebError("That page doesn't exist (error 404).")
            if r.status_code >= 400:
                raise WebError(f"{site} answered with error {r.status_code}.")
            body = bytearray()
            for chunk in r.iter_bytes():
                body += chunk
                if len(body) > MAX_BYTES:
                    break
            return str(r.url), r.headers.get("content-type", "").lower(), bytes(body)
    raise WebError("That link redirects too many times.")


class _Text(HTMLParser):
    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.title = ""
        self.description = ""
        self._skip = 0
        self._in_title = False
        self._link: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        a = {k: v or "" for k, v in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "meta" and (a.get("name", "").lower() == "description" or a.get("property") == "og:description"):
            self.description = self.description or a.get("content", "")
        if tag in _SKIP:
            self._skip += 1
        elif tag in _HEADINGS:
            self.parts.append("\n\n## ")
        elif tag in _BLOCKS:
            self.parts.append("\n")
        if tag == "a" and a.get("href"):
            self._link = [urljoin(self.base, a["href"]), ""]

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in _HEADINGS or tag in _BLOCKS:
            self.parts.append("\n")
        if tag == "a" and self._link:
            url, text = self._link
            if url.startswith(("http://", "https://")) and text.strip():
                self.links.append((re.sub(r"\s+", " ", text).strip()[:80], url))
            self._link = None

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)
            if self._link is not None:
                self._link[1] += data

    def text(self) -> str:
        text = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.parts))
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def _decode(body: bytes, ctype: str) -> str:
    m = re.search(r"charset=([\w-]+)", ctype)
    try:
        return body.decode(m.group(1) if m else "utf-8", errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _pdf_text(body: bytes) -> str:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(body)
    try:
        return "\n\n".join(pdf[i].get_textpage().get_text_range() for i in range(min(len(pdf), 10)))
    finally:
        pdf.close()


def _github_api(client: httpx.Client, path: str, raw: bool = False):
    _, _, body = _get(client, f"https://api.github.com{path}",
                      accept="application/vnd.github.raw" if raw else "application/vnd.github+json")
    return body.decode("utf-8", errors="replace") if raw else json.loads(body)


def _github(client: httpx.Client, url: str) -> Page | None:
    """Richer text for GitHub profiles and repositories. None means: read it as a normal page."""
    p = urlparse(url)
    parts = [x for x in p.path.split("/") if x]
    if host(url) != "github.com" or not parts or parts[0].lower() in _GITHUB_RESERVED:
        return None
    try:
        if len(parts) == 1:
            user = _github_api(client, f"/users/{parts[0]}")
            repos = _github_api(client, f"/users/{parts[0]}/repos?sort=updated&per_page=12")
            lines = [f"GitHub profile: {user.get('name') or user['login']} (@{user['login']})"]
            for field_name in ("bio", "company", "location", "blog"):
                if user.get(field_name):
                    lines.append(f"{field_name.capitalize()}: {user[field_name]}")
            lines.append(f"Public repositories: {user.get('public_repos', 0)}, followers: {user.get('followers', 0)}")
            lines.append("\nRecently updated repositories:")
            links = []
            for r in repos:
                if r.get("fork"):
                    continue
                lines.append(f"- {r['name']}: {r.get('description') or 'no description'} "
                             f"({r.get('language') or 'no language listed'}, {r.get('stargazers_count', 0)} stars, "
                             f"updated {str(r.get('pushed_at', ''))[:10]}) {r['html_url']}")
                links.append((r["name"], r["html_url"]))
            return Page(url, f"{user['login']} on GitHub", "\n".join(lines), links)
        owner, name = parts[0], parts[1]
        repo = _github_api(client, f"/repos/{owner}/{name}")
        try:
            readme = _github_api(client, f"/repos/{owner}/{name}/readme", raw=True)
        except WebError:
            readme = "(no README)"
        lines = [f"GitHub repository {repo['full_name']}: {repo.get('description') or 'no description'}",
                 f"Language: {repo.get('language') or 'not listed'}, stars: {repo.get('stargazers_count', 0)}, "
                 f"forks: {repo.get('forks_count', 0)}, last push: {str(repo.get('pushed_at', ''))[:10]}"]
        if repo.get("topics"):
            lines.append("Topics: " + ", ".join(repo["topics"]))
        if repo.get("homepage"):
            lines.append(f"Homepage: {repo['homepage']}")
        lines.append("\nREADME:\n" + readme[:MAX_TEXT - 600])
        return Page(url, repo["full_name"], "\n".join(lines), [])
    except WebError as e:
        if "404" in str(e):
            raise
        log.info("GitHub API unavailable for %s (%s); reading the page instead", url, e)
        return None


def open_page(url: str) -> Page:
    """Read a public web page as text. Raises WebError with a user-friendly reason."""
    url = normalize(url)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.5"}
    with httpx.Client(follow_redirects=False, timeout=TIMEOUT_S, headers=headers) as client:
        try:
            page = _github(client, url)
            if page is not None:
                return page
            final, ctype, body = _get(client, url)
        except httpx.TimeoutException as e:
            raise WebError(f"{host(url)} took too long to answer.") from e
        except httpx.HTTPError as e:
            raise WebError(f"{host(url)} couldn't be reached.") from e
    title, links, text = "", [], ""
    if "pdf" in ctype or body[:5] == b"%PDF-":
        text = _pdf_text(body)
        title = final.rsplit("/", 1)[-1] or host(final)
    elif "json" in ctype:
        text = json.dumps(json.loads(_decode(body, ctype)), indent=1)
    elif "html" in ctype or body[:300].lstrip().lower().startswith((b"<!doctype", b"<html")):
        parser = _Text(final)
        parser.feed(_decode(body, ctype))
        title = re.sub(r"\s+", " ", parser.title).strip()
        text = parser.text()
        if parser.description:
            text = f"{parser.description.strip()}\n\n{text}"
        links = parser.links
    elif ctype.startswith("text/") or not ctype:
        text = _decode(body, ctype)
    else:
        raise WebError("That link isn't a web page or document I can read.")
    if len(text.strip()) < 40:
        raise WebError("That page had almost no readable text (it may need JavaScript or a login).")
    return Page(final, title or host(final), text[:MAX_TEXT], links)
