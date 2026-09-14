"""
Turning streamed tutor text into speakable chunks.

  SentenceChunker  - releases chunks at sentence boundaries as LLM deltas arrive (never inside $...$),
                     cutting the first chunk early at a clause so TTS can start sooner.
  to_speech        - converts one chunk for TTS: LaTeX math -> words, markdown stripped.

The raw chunk (with LaTeX) is what the UI displays; only the TTS copy goes through to_speech.
"""
from __future__ import annotations

import re

FIRST_CHUNK_MIN_CHARS = 40   # first chunk may end at , ; : once it's at least this long
FIRST_CHUNK_MAX_CHARS = 50   # ...and is cut at a word boundary past this, since TTS time grows with chunk length
MAX_CHUNK_CHARS = 250        # later chunks: cut at a word boundary if a run-on sentence gets this long

_ABBREVIATIONS = {"e.g", "i.e", "etc", "vs", "approx", "dr", "mr", "mrs", "ms", "fig", "eq"}
# Don't end a forced cut on these; "the chain rule lets you differentiate a | composition" sounds broken.
_WEAK_ENDINGS = {"a", "an", "the", "of", "to", "and", "or", "but", "by", "in", "on", "at", "for", "with",
                 "is", "are", "as", "that", "your", "you", "its", "this", "if", "then"}


class SentenceChunker:
    def __init__(self):
        self._buf = ""
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out = []
        while (cut := self._find_cut()) is not None:
            chunk, self._buf = self._buf[:cut].strip(), self._buf[cut:]
            if chunk:
                out.append(chunk)
                self._emitted += 1
        return out

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []

    def _find_cut(self) -> int | None:
        b = self._buf
        limit = FIRST_CHUNK_MAX_CHARS if self._emitted == 0 else MAX_CHUNK_CHARS
        in_math = False
        word_break = -1  # last space outside math and before `limit` that doesn't follow a weak word
        i = 0
        while i < len(b):
            ch = b[i]
            if ch == "$":
                in_math = not in_math
                i += 2 if b.startswith("$$", i) else 1
                continue
            if not in_math:
                nxt = b[i + 1] if i + 1 < len(b) else ""
                if ch in ".!?" and nxt in (" ", "\n"):
                    word = re.search(r"([A-Za-z.]+)$", b[:i])
                    if not (ch == "." and word and word.group(1).lower() in _ABBREVIATIONS):
                        return i + 1
                elif ch == "\n" and b[:i].strip():
                    return i + 1
                elif (self._emitted == 0 and ch in ",;:" and nxt == " "
                      and i + 1 >= FIRST_CHUNK_MIN_CHARS):
                    return i + 1
                elif ch == " " and i <= limit:
                    prev = b[:i].rsplit(" ", 1)[-1].lower()
                    if prev and prev not in _WEAK_ENDINGS:
                        word_break = i
            i += 1
        if len(b) > limit and word_break > 0:
            return word_break + 1
        return None


# --------------------------------------------------------------------------- LaTeX -> words

_SPACING = re.compile(
    r"\\(?:[!,;:> ]|q?quad|left|right|[bB]igg?[lr]?|displaystyle|textstyle|scriptstyle|limits|nolimits)(?![a-zA-Z])")
_NAMED_FUNCS = "sin|cos|tan|sec|csc|cot|arcsin|arccos|arctan|sinh|cosh|tanh|ln|log|exp"
_FUNC_LETTERS = "fghpqrsuvyFGHPA"

_COMMANDS = {
    "sin": "sine", "cos": "cosine", "tan": "tangent", "sec": "secant", "csc": "cosecant",
    "cot": "cotangent", "arcsin": "arc sine", "arccos": "arc cosine", "arctan": "arc tangent",
    "sinh": "hyperbolic sine", "cosh": "hyperbolic cosine", "tanh": "hyperbolic tangent",
    "ln": "natural log", "log": "log", "exp": "exp",
    "cdot": " times ", "times": " times ", "div": " divided by ", "pm": " plus or minus ",
    "mp": " minus or plus ", "le": " is less than or equal to ", "leq": " is less than or equal to ",
    "ge": " is greater than or equal to ", "geq": " is greater than or equal to ",
    "ne": " is not equal to ", "neq": " is not equal to ", "approx": " is approximately ",
    "to": " approaches ", "rightarrow": ", so ", "Rightarrow": ", so ", "implies": ", which means ",
    "longrightarrow": ", so ", "infty": "infinity", "int": " the integral of ", "sum": " the sum of ",
    "partial": " partial ", "ldots": " and so on ", "dots": " and so on ", "cdots": " and so on ",
    "circ": " degrees ", "prime": " prime ",
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta", "Delta": "delta",
    "epsilon": "epsilon", "theta": "theta", "lambda": "lambda", "mu": "mu", "pi": "pi",
    "sigma": "sigma", "phi": "phi", "omega": "omega",
}

_OF, _TIMES = "\x01", "\x02"  # placeholders so later rules don't re-match inserted words


def _read_group(s: str, i: int) -> tuple[str, int]:
    """Read a {...} group (or a single token) starting at s[i]."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
        return s[i + 1:], len(s)
    if s[i] == "\\":
        m = re.match(r"\\[a-zA-Z]+", s[i:])
        if m:
            return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def _frac(num: str, den: str) -> str:
    n, d = num.strip(), den.strip()
    top = re.fullmatch(r"d\s*(?:\^\s*\{?\s*(\d)\s*\}?)?\s*([A-Za-z]?)", n)
    bottom = re.fullmatch(r"d\s*([A-Za-z])\s*(?:\^\s*\{?\s*\d\s*\}?)?", d)
    if top and bottom:
        order, var, wrt = top.group(1), top.group(2), bottom.group(1)
        # Bare operator (d/dx with nothing on top) applies to what follows: "d by d x of ..."
        suffix = "" if var else f" {_OF}"
        if order == "2":
            return f" d squared {var} by d {wrt} squared{suffix} "
        return f" d {var} by d {wrt}{suffix} "
    return f" {n} over {d} "


def _expand(s: str) -> str:
    """Expand commands that take arguments: fractions, roots, text, limits."""
    out, i = [], 0
    while i < len(s):
        m = re.match(r"\\([dt]?frac|sqrt|text|mathrm|operatorname|mathbf|mathit|lim)(?![a-zA-Z])", s[i:])
        if not m:
            out.append(s[i])
            i += 1
            continue
        name = m.group(1)
        i += len(m.group(0))
        if name.endswith("frac"):
            num, i = _read_group(s, i)
            den, i = _read_group(s, i)
            out.append(_frac(_expand(num), _expand(den)))
        elif name == "sqrt":
            index = None
            if i < len(s) and s[i] == "[" and "]" in s[i:]:
                j = s.index("]", i)
                index, i = s[i + 1:j].strip(), j + 1
            arg, i = _read_group(s, i)
            root = "square" if index in (None, "2") else ("cube" if index == "3" else f"{index}th")
            out.append(f" the {root} root of {_expand(arg)} ")
        elif name == "lim":
            below = ""
            if i < len(s) and s[i] == "_":
                below, i = _read_group(s, i + 1)
            out.append(f" the limit as {_expand(below)} of " if below else " the limit of ")
        else:
            arg, i = _read_group(s, i)
            out.append(f" {_expand(arg)} ")
    return "".join(out)


def _power(exp: str) -> str:
    e = exp.strip()
    if e == "2":
        return " squared "
    if e == "3":
        return " cubed "
    if e in ("\\circ", "circ"):
        return " degrees "
    return f" to the power of {e} "


def latex_to_speech(expr: str) -> str:
    s = _SPACING.sub(" ", expr)
    # Function application and implicit multiplication, decided on the raw LaTeX before words are inserted.
    s = re.sub(rf"\\({_NAMED_FUNCS})(\s*\^\s*(?:\{{[^{{}}]*\}}|\S))?\s*\(", rf"\\\1\2 {_OF}(", s)
    s = re.sub(rf"(?<![A-Za-z\\])([{_FUNC_LETTERS}])('*)\s*\(", rf"\1\2 {_OF}(", s)
    s = re.sub(r"([0-9A-Za-z)}])\s*\(", rf"\1 {_TIMES}(", s)
    s = re.sub(rf"\)\s*(?=[A-Za-z]|\\(?:{_NAMED_FUNCS}|sqrt|frac|pi)(?![a-zA-Z]))", rf") {_TIMES} ", s)
    s = _expand(s)
    s = re.sub(rf"{_OF}\s*{_TIMES}", _OF, s)
    s = re.sub(r"\|([^|]+)\|", r" the absolute value of \1 ", s)
    s = re.sub(r"\^\s*\{([^{}]*)\}", lambda m: _power(m.group(1)), s)
    s = re.sub(r"\^\s*(\\[a-zA-Z]+|-?[0-9A-Za-z])", lambda m: _power(m.group(1)), s)
    s = re.sub(r"_\s*\{([^{}]*)\}", r" sub \1 ", s)
    s = re.sub(r"_\s*([0-9A-Za-z])", r" sub \1 ", s)
    s = s.replace("''", " double prime ").replace("'", " prime ")
    s = re.sub(r"\\([a-zA-Z]+)", lambda m: f" {_COMMANDS.get(m.group(1), m.group(1))} ", s)
    s = re.sub(r"(^|[=(,{" + _OF + _TIMES + r"])\s*-", r"\1 negative ", s)
    for sym, word in (("<=", " is less than or equal to "), (">=", " is greater than or equal to "),
                      ("=", " equals "), ("+", " plus "), ("-", " minus "), ("<", " is less than "),
                      (">", " is greater than "), ("/", " over "), ("%", " percent ")):
        s = s.replace(sym, word)
    s = s.replace(_OF, " of ").replace(_TIMES, " times ")
    s = re.sub(r"[{}()\[\]\\]", " ", s)
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    return re.sub(r"\s+", " ", s).strip()


_MATH_SPANS = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)
# Math symbols the LLM sometimes types directly in prose instead of LaTeX.
_UNICODE_MATH = {"π": " pi ", "θ": " theta ", "Δ": " delta ", "∞": " infinity ", "≈": " is approximately ",
                 "≤": " is less than or equal to ", "≥": " is greater than or equal to ", "≠": " is not equal to ",
                 "×": " times ", "÷": " divided by ", "−": " minus ", "→": " approaches ", "√": " the square root of ",
                 "²": " squared ", "³": " cubed "}
# Anything outside Latin letters, Latin-1 and general punctuation (keeps curly quotes and dashes).
_NON_LATIN = re.compile(r"[^\x00-ɏ -⁯]")


def to_speech(text: str) -> str:
    """Speakable version of a chunk: math spoken as words, markdown and stray symbols removed."""
    s = text.replace("\u202f", " ").replace("\u00a0", " ").replace("\u2011", "-")
    s = s.replace("[interrupted by the student]", "")
    s = _MATH_SPANS.sub(lambda m: " " + latex_to_speech(next(g for g in m.groups() if g is not None)) + " ", s)
    s = re.sub(r"^\s*(#{1,6}|[-*\u2022]|\d+[.)])\s+", "", s, flags=re.M)
    s = s.replace("**", "").replace("__", "").replace("`", "").replace("$", "")
    for sym, word in _UNICODE_MATH.items():
        s = s.replace(sym, word)
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isascii() for c in letters) / len(letters) < 0.5:
        return ""  # mostly non-Latin script: the English voice can't say it, and synthesis crawls
    s = _NON_LATIN.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", s)
