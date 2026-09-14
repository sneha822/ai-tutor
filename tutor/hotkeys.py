"""
Global hotkeys via a pynput keyboard listener, matched by physical key code.

On macOS, Option turns letters into dead keys (Option+I types "ˆ"), so pynput's character-based
GlobalHotKeys never matches combos like Ctrl+Option+I. Matching the virtual key code avoids that.
Each physical press fires once: key auto-repeat while held is ignored. Failure to start is logged loudly
and never crashes the app.
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Callable

log = logging.getLogger("hotkey")

# macOS virtual key codes (kVK_ANSI_*) for letters and digits.
_MAC_VK = {"a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12,
           "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23,
           "9": 25, "7": 26, "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40,
           "n": 45, "m": 46}
_VK_TO_CHAR = {vk: ch for ch, vk in _MAC_VK.items()}
_MODIFIERS = {"ctrl", "alt", "cmd", "shift"}


def parse_combo(combo: str) -> tuple[frozenset[str], str]:
    """'<ctrl>+<alt>+i' -> ({'ctrl', 'alt'}, 'i')."""
    mods, key = set(), None
    for part in combo.lower().split("+"):
        name = part.strip().strip("<>")
        if part.strip().startswith("<") and name in _MODIFIERS:
            mods.add(name)
        elif len(name) == 1 and name.isalnum():
            key = name
        else:
            raise ValueError(f"unsupported hotkey part {part!r} in {combo!r}: use <ctrl>/<alt>/<cmd>/<shift> "
                             f"plus one letter or digit")
    if key is None:
        raise ValueError(f"hotkey {combo!r} needs a letter or digit")
    return frozenset(mods), key


class Hotkeys:
    def __init__(self, bindings: dict[str, Callable[[], None]], on_any_key: Callable[[], None] | None = None):
        self._bindings = {parse_combo(combo): (combo, fn) for combo, fn in bindings.items()}
        self._on_any_key = on_any_key   # diagnostics only: called with no key information
        self._held: set[str] = set()    # modifiers currently down
        self._down: set[str] = set()    # other keys currently down, so auto-repeat doesn't re-fire
        self._listener = None

    def start(self) -> None:
        names = ", ".join(combo for combo, _ in self._bindings.values())
        try:
            from pynput import keyboard
            self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.start()
            self._listener.wait()
            if getattr(self._listener, "IS_TRUSTED", True) is False:
                log.warning("macOS reports this terminal as not Accessibility-trusted. Hotkeys (%s) still work when it "
                            "has Input Monitoring; confirm with scripts/hotkey_check.py.", names)
            else:
                log.info("Hotkeys active: %s", names)
        except Exception as e:
            log.error("HOTKEYS UNAVAILABLE (%s: %s); use terminal commands instead", type(e).__name__, e)

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass

    @staticmethod
    def _modifier(key) -> str | None:
        name = getattr(key, "name", None) or ""   # pynput Key enum members: ctrl_l, alt_r, cmd, shift, ...
        base = name.split("_")[0]
        return base if base in _MODIFIERS else None

    @staticmethod
    def _char(key) -> str | None:
        vk = getattr(key, "vk", None)
        if sys.platform == "darwin" and vk in _VK_TO_CHAR:
            return _VK_TO_CHAR[vk]
        char = getattr(key, "char", None)
        return char.lower() if char and len(char) == 1 else None

    def _on_press(self, key) -> None:
        try:
            if self._on_any_key is not None:
                self._on_any_key()
            mod = self._modifier(key)
            if mod:
                self._held.add(mod)
                return
            char = self._char(key)
            if char is None or char in self._down:
                return
            self._down.add(char)
            hit = self._bindings.get((frozenset(self._held), char))
            if hit:
                combo, fn = hit
                log.info("hotkey %s", combo)
                fn()
        except Exception:
            log.exception("hotkey handler failed (continuing)")

    def _on_release(self, key) -> None:
        mod = self._modifier(key)
        if mod:
            self._held.discard(mod)
            if not self._held:
                self._down.clear()   # safety net if a key-up was ever missed
            return
        char = self._char(key)
        if char:
            self._down.discard(char)
