// AI Tutor page: conversation, focus, Meet-style call controls (with the microphone/speaker menu), self-view and the
// privacy panel, all driven over one WebSocket. The 3D avatar and the self-view load as separate modules; if either
// fails, the rest still works.

const $ = (id) => document.getElementById(id);
const convo = $("conversation");
const empty = $("empty");
const focusCard = $("focus-card");
const root = document.documentElement;
const actionButtons = document.querySelectorAll("[data-action]");
const drawer = $("privacy");
const scrim = $("scrim");
const privacyBtn = $("privacy-btn");
const audioMenu = $("audio-menu");
const audioMenuBtn = $("audio-menu-btn");

const MATH_DELIMITERS = [
  { left: "$$", right: "$$", display: true },
  { left: "\\[", right: "\\]", display: true },
  { left: "$", right: "$", display: false },
  { left: "\\(", right: "\\)", display: false },
];
const STATE_TEXT = { FOCUSED: "Focused", UNFOCUSED: "Drifting", DISTRACTED: "Distracted" };
const MODE_TEXT = { listening: "Listening", user: "Hearing you", thinking: "Thinking", speaking: "Speaking", muted: "Mic off" };
const CHECK_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>';

let socket = null;
let retries = 0;
let greeted = false;
let tutorBubble = null;
let avatar = null;
let selfView = null;
let toastTimer = null;
let lastStatus = null;
let devices = null;   // latest microphone/speaker list from the app

import("./avatar.js")
  .then(({ createAvatar }) => {
    avatar = createAvatar($("stage"));
    document.body.classList.add("has-avatar");
  })
  .catch((err) => {
    console.warn("3D avatar unavailable, using the fallback orb", err);
    document.body.classList.add("no-avatar");
  });

import("./selfview.js")
  .then(({ createSelfView }) => {
    selfView = createSelfView({ send: sendAction });
    if (lastStatus) syncSelfView(lastStatus);
  })
  .catch((err) => console.warn("self-view unavailable", err));

function sendMessage(message) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify(message));
  return true;
}

function sendAction(action) {
  return sendMessage({ action });
}

// ------------------------------------------------------------------ conversation

const nearBottom = () => convo.scrollHeight - convo.scrollTop - convo.clientHeight < 160;

function append(el) {
  const stick = nearBottom();
  empty.hidden = true;
  convo.appendChild(el);
  if (stick) convo.scrollTop = convo.scrollHeight;
}

function message(kind, label) {
  const el = document.createElement("div");
  el.className = `msg ${kind}`;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = label;
  el.appendChild(who);
  return el;
}

function renderMath(el) {
  if (typeof window.renderMathInElement !== "function") return; // KaTeX unavailable: raw LaTeX stays readable
  try {
    window.renderMathInElement(el, { delimiters: MATH_DELIMITERS, throwOnError: false });
  } catch (err) {
    console.warn("KaTeX render failed", err);
  }
}

function endTutor() {
  if (tutorBubble) tutorBubble.classList.remove("live");
  tutorBubble = null;
}

function showToast(text, kind = "nudge") {
  const toast = $("toast");
  toast.textContent = text;
  toast.classList.toggle("info", kind === "info");
  toast.hidden = false;
  toast.classList.remove("show");
  void toast.offsetWidth; // restart the transition
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => { toast.hidden = true; }, 450);
  }, kind === "info" ? 2800 : 4500);
}

function onEvent(ev, live) {
  switch (ev.type) {
    case "user": {
      endTutor();
      const el = message("user", "You");
      const text = document.createElement("span");
      text.textContent = ev.text;
      el.appendChild(text);
      append(el);
      break;
    }
    case "tutor_chunk": {
      if (!tutorBubble) {
        tutorBubble = message("tutor", "Tutor");
        if (live) tutorBubble.classList.add("live");
        append(tutorBubble);
      }
      const stick = nearBottom();
      const span = document.createElement("span");
      span.textContent = `${ev.text} `;
      tutorBubble.appendChild(span);
      renderMath(span); // each chunk holds only complete $...$ spans
      if (stick) convo.scrollTop = convo.scrollHeight;
      break;
    }
    case "tutor_done":
      endTutor();
      break;
    case "interrupted":
      if (tutorBubble) {
        tutorBubble.classList.add("interrupted");
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = "interrupted";
        tutorBubble.appendChild(tag);
      }
      endTutor();
      if (live) avatar?.react("interrupt");
      break;
    case "nudge": {
      endTutor();
      const chip = document.createElement("div");
      chip.className = "nudge";
      chip.textContent = `Focus nudge · ${ev.reason}`;
      append(chip);
      if (live) {
        showToast(`Focus nudge · ${ev.reason}`);
        avatar?.react("nudge");
      }
      break;
    }
  }
}

// ------------------------------------------------------------------ status, toggles, voice levels

function setToggle(button, on, device, keys) {
  const tip = `${on ? "Turn off" : "Turn on"} ${device} (${keys})`;
  button.classList.toggle("off", !on);
  button.setAttribute("aria-pressed", String(!on));
  button.setAttribute("aria-label", tip);
  button.dataset.tip = tip;
}

function syncSelfView(s) {
  selfView?.setCameraEnabled(s.camera_enabled !== false);
  selfView?.setMicEnabled(s.mic_enabled !== false);
}

function applyStatus(s) {
  lastStatus = s;
  const cameraOn = s.camera_enabled !== false;
  const micOn = s.mic_enabled !== false;

  $("meter-threshold").style.left = `${s.threshold}%`;
  const f = s.focus;
  let kind = "nocam";
  if (f && f.camera_ok) {
    const score = Math.round(f.score);
    kind = f.state === "DISTRACTED" ? "bad" : f.state === "UNFOCUSED" ? "warn" : "good";
    $("focus-score").textContent = score;
    $("meter-fill").style.width = `${score}%`;
    $("focus-state").textContent = (STATE_TEXT[f.state] || f.state) + (f.calibrated ? "" : " · calibrating");
    $("focus-detail").textContent = f.description || (f.face ? "Eyes on the lesson" : "No face in view");
  } else {
    $("focus-score").textContent = "--";
    $("meter-fill").style.width = "0%";
    $("focus-state").textContent = "Camera off";
    $("focus-detail").textContent = cameraOn ? "Waiting for the camera" : "Tracking paused while your camera is off";
  }
  focusCard.className = `focus-card glass ${kind}`;
  avatar?.setFocus(kind);

  const mode = s.user_speaking ? "user" : s.tutor_speaking ? "speaking" : s.thinking ? "thinking" : micOn ? "listening" : "muted";
  $("mode").dataset.mode = mode;
  $("mode-text").textContent = MODE_TEXT[mode];
  avatar?.setMode(mode === "muted" ? "listening" : mode);

  setToggle($("mic-btn"), micOn, "microphone", "⌃⌥M");
  setToggle($("cam-btn"), cameraOn, "camera", "⌃⌥V");
  audioMenu.classList.toggle("mic-off", !micOn);
  syncSelfView(s);

  $("paused").hidden = !s.suppressed;
  const suppress = $("suppress-btn");
  suppress.classList.toggle("active", s.suppressed);
  suppress.dataset.tip = s.suppressed ? "Resume automatic nudges (⌃⌥S)" : "Pause automatic nudges (⌃⌥S)";
  suppress.setAttribute("aria-label", suppress.dataset.tip);

  updatePrivacyLive(s, cameraOn, micOn);
}

function applyLevel(m) {
  avatar?.setLevels(m.tutor, m.user);
  selfView?.setLevel(m.user);
  if (!audioMenu.hidden) $("dm-level").style.width = `${Math.round(Math.min(1, m.user) * 100)}%`;
  root.style.setProperty("--tutor-level", m.tutor.toFixed(3));
}

// ------------------------------------------------------------------ microphone + speaker menu

function renderDeviceList(list, { names, chosen, fallback, active, action, key, label }) {
  const items = [{ value: null, name: "System default", sub: fallback || "" }];
  names.forEach((name) => items.push({ value: name, name, sub: "" }));
  if (chosen && !names.includes(chosen)) {
    // Saved choice that isn't plugged in: the app is using the default until it comes back.
    items.push({ value: chosen, name: chosen, sub: `Not connected · using ${active || "the default"}`, missing: true });
  }
  list.replaceChildren();
  for (const item of items) {
    const checked = item.value === (chosen || null);
    const button = document.createElement("button");
    button.type = "button";
    button.className = item.missing ? "dm-item missing" : "dm-item";
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(checked));
    button.innerHTML = CHECK_ICON;
    const name = document.createElement("span");
    name.className = "dm-name";
    name.textContent = item.name;
    const sub = document.createElement("span");
    sub.className = "dm-sub";
    sub.textContent = item.sub;
    button.append(name, sub);
    button.addEventListener("click", () => {
      if (checked || !sendMessage({ action, device: item.value })) return;
      devices[key] = item.value; // show the pick right away; the app's reply confirms it
      renderDevices();
      list.querySelector('[aria-checked="true"]')?.focus();
      showToast(`${label} · ${item.value || `System default${fallback ? ` (${fallback})` : ""}`}`, "info");
    });
    list.appendChild(button);
  }
}

function renderDevices() {
  if (!devices) return;
  // Re-rendering replaces the buttons, so keep keyboard focus on the same row.
  const focused = document.activeElement;
  const focusList = focused?.classList.contains("dm-item") ? focused.parentElement : null;
  const focusIndex = focusList ? [...focusList.children].indexOf(focused) : -1;

  renderDeviceList($("dm-inputs"), {
    names: devices.inputs || [], chosen: devices.input, fallback: devices.default_input, active: devices.input_active,
    action: "set_input_device", key: "input", label: "Microphone",
  });
  renderDeviceList($("dm-outputs"), {
    names: devices.outputs || [], chosen: devices.output, fallback: devices.default_output, active: devices.output_active,
    action: "set_output_device", key: "output", label: "Speaker",
  });
  if (focusList) focusList.children[focusIndex]?.focus();
  if (lastStatus) updatePrivacyLive(lastStatus, lastStatus.camera_enabled !== false, lastStatus.mic_enabled !== false);
}

function placeAudioMenu() {
  const bar = $("controls").getBoundingClientRect();
  const anchor = audioMenuBtn.getBoundingClientRect();
  const left = Math.min(window.innerWidth - audioMenu.offsetWidth - 12, Math.max(12, anchor.left - 16));
  audioMenu.style.left = `${Math.round(left)}px`;
  audioMenu.style.bottom = `${Math.round(window.innerHeight - bar.top + 12)}px`;
}

function setAudioMenu(open) {
  if (open === !audioMenu.hidden) return;
  if (!open && audioMenu.contains(document.activeElement)) audioMenuBtn.focus();
  audioMenu.hidden = !open;
  audioMenuBtn.setAttribute("aria-expanded", String(open));
  audioMenuBtn.classList.toggle("open", open);
  if (!open) return;
  $("dm-level").style.width = "0%";
  placeAudioMenu();
  renderDevices();
  sendAction("list_devices"); // refreshes the list, so anything plugged in since the last look shows up
  (audioMenu.querySelector('[aria-checked="true"]') || $("dm-test")).focus({ preventScroll: true });
}

audioMenuBtn.addEventListener("click", () => setAudioMenu(audioMenu.hidden));
document.addEventListener("pointerdown", (e) => {
  if (!audioMenu.hidden && !audioMenu.contains(e.target) && !audioMenuBtn.contains(e.target)) setAudioMenu(false);
});
window.addEventListener("resize", () => { if (!audioMenu.hidden) placeAudioMenu(); });

$("dm-test").addEventListener("click", () => {
  const test = $("dm-test");
  if (!sendAction("test_speaker")) return;
  test.classList.add("playing");
  test.textContent = "Playing…";
  setTimeout(() => {
    test.classList.remove("playing");
    test.textContent = "Test";
  }, 900);
});

// ------------------------------------------------------------------ privacy drawer

function setLive(id, kind, value) {
  const el = $(id);
  el.classList.remove("local", "cloud", "off");
  el.classList.add(kind);
  el.querySelector(".live-value").textContent = value;
}

function updatePrivacyLive(s, cameraOn, micOn) {
  setLive("live-camera", cameraOn ? "local" : "off", cameraOn ? "On · video stays here" : "Off · webcam released");
  setLive("live-mic", micOn ? "cloud" : "off", micOn ? `On · ${devices?.input_active || "voice clips transcribed"}` : "Off · nothing is heard");
  if (s.stt_source === "groq") setLive("live-stt", "cloud", "Groq Whisper (cloud)");
  else if (s.stt_source === "local") setLive("live-stt", "local", "On this computer (offline)");
  else setLive("live-stt", "off", "Not used yet");
}

function fillPrivacy(info) {
  document.querySelectorAll("[data-p]").forEach((el) => {
    const value = info[el.dataset.p];
    if (value !== undefined && value !== null && value !== "") el.textContent = value;
  });
}

function setDrawer(open) {
  const wasOpen = drawer.classList.contains("open");
  drawer.classList.toggle("open", open);
  drawer.setAttribute("aria-hidden", String(!open));
  privacyBtn.setAttribute("aria-expanded", String(open));
  scrim.hidden = !open;
  if (open) $("privacy-close").focus();
  else if (wasOpen) privacyBtn.focus();
}

privacyBtn.addEventListener("click", () => setDrawer(!drawer.classList.contains("open")));
$("privacy-close").addEventListener("click", () => setDrawer(false));
scrim.addEventListener("click", () => setDrawer(false));
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!audioMenu.hidden) setAudioMenu(false);
  else if (drawer.classList.contains("open")) setDrawer(false);
});

// ------------------------------------------------------------------ connection + controls

function setConnected(live) {
  const conn = $("conn");
  conn.textContent = live ? "live" : "offline";
  conn.classList.toggle("live", live);
  actionButtons.forEach((b) => { b.disabled = !live; });
  audioMenuBtn.disabled = !live;
  if (!live) setAudioMenu(false);
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/ws`);
  socket.onopen = () => {
    retries = 0;
    setConnected(true);
    if (!greeted) {
      greeted = true;
      avatar?.react("hello");
    }
  };
  socket.onmessage = (msg) => {
    let data;
    try { data = JSON.parse(msg.data); } catch { return; }
    if (data.type === "level") {
      applyLevel(data);
    } else if (data.type === "status") {
      applyStatus(data);
    } else if (data.type === "devices") {
      devices = data;
      renderDevices();
    } else if (data.type === "notice") {
      showToast(data.text, "info");
    } else if (data.type === "hello") {
      convo.querySelectorAll(".msg, .nudge").forEach((n) => n.remove());
      empty.hidden = false;
      tutorBubble = null;
      data.events.forEach((ev) => onEvent(ev, false));
      endTutor();
      if (data.privacy) fillPrivacy(data.privacy);
      if (data.devices) devices = data.devices;
      applyStatus(data.status);
      renderDevices();
      convo.scrollTop = convo.scrollHeight;
    } else {
      onEvent(data, true);
    }
  };
  socket.onclose = () => {
    setConnected(false);
    avatar?.setLevels(0, 0);
    selfView?.setLevel(0);
    setTimeout(connect, Math.min(5000, 500 * 2 ** retries++));
  };
  socket.onerror = () => socket.close();
}

actionButtons.forEach((button) => button.addEventListener("click", () => {
  if (!sendAction(button.dataset.action)) return;
  button.classList.add("flash");
  setTimeout(() => button.classList.remove("flash"), 260);
}));

connect();
