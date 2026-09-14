// Meet-style self-view: your own camera in a floating tile you can drag (it snaps to the nearest corner of the
// stage), resize from its inner corner, and minimize to a pill. The video is opened and drawn only inside this
// browser tab: it never goes to Python, the network or the AI.

const STORE_KEY = "tutor.selfview.v1";
const MIN_W = 180;
const MAX_W = 520;
const ASPECT = 16 / 9;
const DEFAULTS = { corner: "tr", width: 260, minimized: false };

export function createSelfView({ send }) {
  const tile = document.getElementById("selfview");
  const video = tile.querySelector("video");
  const statusEl = tile.querySelector(".sv-status");
  const grip = tile.querySelector(".sv-resize");
  const prefs = loadPrefs();

  let stream = null;
  let wanted = false;       // what the app says: camera on or off
  let starting = false;
  let blockedMessage = "";  // set when the browser refuses the camera

  // ---------------------------------------------------------------- layout

  function loadPrefs() {
    try {
      return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORE_KEY) || "{}") };
    } catch {
      return { ...DEFAULTS };
    }
  }

  function savePrefs() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(prefs));
    } catch {
      // private window or blocked storage: the tile still works, it just won't remember its place
    }
  }

  // The area the tile may occupy: left of the conversation panel (above it on narrow screens), between the
  // top bar and the control bar, and never over the focus card.
  function bounds() {
    const narrow = window.innerWidth <= 900;
    const gap = narrow ? 14 : 24;
    const panel = document.querySelector(".panel").getBoundingClientRect();
    const bar = document.getElementById("controls").getBoundingClientRect();
    const card = document.getElementById("focus-card").getBoundingClientRect();
    return {
      left: gap,
      right: narrow ? window.innerWidth - gap : panel.left - gap,
      top: narrow ? 64 : 76,
      bottom: bar.top - 12,
      belowCard: card.bottom + 12,
    };
  }

  function clampWidth(width, b) {
    const maxByArea = Math.min((b.right - b.left) * 0.55, (b.bottom - b.top) * 0.55 * ASPECT);
    return Math.round(Math.max(MIN_W, Math.min(MAX_W, maxByArea, width)));
  }

  function place(animate) {
    const b = bounds();
    tile.classList.toggle("minimized", prefs.minimized);
    if (prefs.minimized) {
      tile.style.width = "";
      tile.style.height = "";
    } else {
      const width = clampWidth(prefs.width, b);
      tile.style.width = `${width}px`;
      tile.style.height = `${Math.round(width / ASPECT)}px`;
    }
    const w = tile.offsetWidth;
    const h = tile.offsetHeight;
    const vertical = prefs.corner[0];
    const horizontal = prefs.corner[1];
    const x = horizontal === "l" ? b.left : b.right - w;
    let y = vertical === "t" ? b.top : b.bottom - h;
    if (prefs.corner === "tl") y = b.belowCard;
    tile.dataset.corner = prefs.corner;
    tile.classList.toggle("animate", animate);
    tile.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  }

  // ---------------------------------------------------------------- drag to move (snaps to a corner)

  let drag = null;

  tile.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || e.target.closest("button") || e.target === grip) return;
    const rect = tile.getBoundingClientRect();
    drag = { dx: e.clientX - rect.left, dy: e.clientY - rect.top, moved: false };
    try {
      tile.setPointerCapture(e.pointerId);
    } catch {
      // capture refused (e.g. synthetic events): moves still arrive while the pointer is over the tile
    }
  });

  tile.addEventListener("pointermove", (e) => {
    if (!drag) return;
    if (!drag.moved) {
      drag.moved = true;
      tile.classList.remove("animate");
      tile.classList.add("dragging");
    }
    tile.style.transform = `translate(${e.clientX - drag.dx}px, ${e.clientY - drag.dy}px)`;
  });

  const endDrag = (e) => {
    if (!drag) return;
    const moved = drag.moved;
    drag = null;
    tile.classList.remove("dragging");
    if (tile.hasPointerCapture(e.pointerId)) tile.releasePointerCapture(e.pointerId);
    if (!moved) return;
    const b = bounds();
    const rect = tile.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    prefs.corner = (cy < (b.top + b.bottom) / 2 ? "t" : "b") + (cx < (b.left + b.right) / 2 ? "l" : "r");
    savePrefs();
    place(true);
  };
  tile.addEventListener("pointerup", endDrag);
  tile.addEventListener("pointercancel", endDrag);

  // ---------------------------------------------------------------- resize from the inner corner (16:9)

  let resize = null;

  grip.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    resize = { x: e.clientX, y: e.clientY, width: tile.offsetWidth };
    try {
      grip.setPointerCapture(e.pointerId);
    } catch {
      // capture refused: resizing still follows moves over the grip
    }
    tile.classList.remove("animate");
  });

  grip.addEventListener("pointermove", (e) => {
    if (!resize) return;
    // Dragging the grip away from the tile's anchored corner makes it bigger.
    const sx = prefs.corner[1] === "r" ? -1 : 1;
    const sy = prefs.corner[0] === "t" ? 1 : -1;
    const grow = ((e.clientX - resize.x) * sx + (e.clientY - resize.y) * sy * ASPECT) / 2;
    prefs.width = clampWidth(resize.width + grow, bounds());
    place(false);
  });

  const endResize = (e) => {
    if (!resize) return;
    resize = null;
    if (grip.hasPointerCapture(e.pointerId)) grip.releasePointerCapture(e.pointerId);
    savePrefs();
  };
  grip.addEventListener("pointerup", endResize);
  grip.addEventListener("pointercancel", endResize);

  // ---------------------------------------------------------------- buttons

  tile.addEventListener("click", (e) => {
    const button = e.target.closest("[data-sv]");
    if (button) {
      const what = button.dataset.sv;
      if (what === "camera") send("toggle_camera");
      if (what === "minimize" || what === "expand") {
        prefs.minimized = what === "minimize";
        savePrefs();
        place(true);
      }
      return;
    }
    // Clicking the placeholder retries a camera the browser blocked (browsers need a user gesture for this).
    if (wanted && blockedMessage && e.target.closest(".sv-placeholder")) {
      blockedMessage = "";
      start();
    }
  });

  window.addEventListener("resize", () => place(false));

  // ---------------------------------------------------------------- camera

  function render() {
    tile.classList.toggle("cam-off", !wanted);
    tile.classList.toggle("no-video", wanted && !stream);
    statusEl.textContent = !wanted ? "Your camera is off" : blockedMessage || (stream ? "" : "Starting camera…");
  }

  async function start() {
    if (stream || starting) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      blockedMessage = "Camera preview isn't available in this browser";
      render();
      return;
    }
    starting = true;
    render();
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      if (!wanted) {
        media.getTracks().forEach((t) => t.stop()); // turned off while the permission prompt was open
      } else {
        stream = media;
        video.srcObject = stream;
        await video.play().catch(() => {});
      }
    } catch (err) {
      console.warn("self-view camera unavailable", err);
      blockedMessage = err && err.name === "NotAllowedError"
        ? "Allow camera access for this page, then click here"
        : "Camera preview unavailable. Click to retry";
    } finally {
      starting = false;
      render();
    }
  }

  function stop() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    video.srcObject = null;
  }

  place(false);
  render();

  return {
    setCameraEnabled(on) {
      if (on === wanted) return;
      wanted = on;
      blockedMessage = "";
      if (on) start();
      else stop();
      render();
    },
    setMicEnabled(on) {
      tile.classList.toggle("mic-off", !on);
    },
    setLevel(level) {
      const speaking = tile.classList.contains("mic-off") ? 0 : Math.max(0, Math.min(1, (level - 0.08) * 2.2));
      tile.style.setProperty("--speak", speaking.toFixed(2));
    },
    relayout() {
      place(false);
    },
  };
}
