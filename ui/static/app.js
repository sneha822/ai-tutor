// AI Tutor page: welcome form (name + subject or talk mode), conversation, the AI's emotion, focus, Meet-style call
// controls (with the microphone/speaker menu), self-view and the privacy panel, all driven over one WebSocket. The 3D
// avatar and the self-view load as separate modules; if either fails, the rest still works.

import { EMOTIONS, emotionMeta } from "./emotions.js";

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
const welcome = $("welcome");

const MATH_DELIMITERS = [
  { left: "$$", right: "$$", display: true },
  { left: "\\[", right: "\\]", display: true },
  { left: "$", right: "$", display: false },
  { left: "\\(", right: "\\)", display: false },
];
const STATE_TEXT = { FOCUSED: "Focused", UNFOCUSED: "Drifting", DISTRACTED: "Distracted" };
const MODE_TEXT = { listening: "Listening", user: "Hearing you", thinking: "Thinking", speaking: "Speaking", muted: "Mic off" };
const MODE_NAMES = { tutor: "tutor", friend: "friend", therapist: "therapist", chat: "chat" };
const CHECK_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>';
const WELCOME_KEY = "tutor.welcome.v1";
const MODE_NOTES = {
  tutor: "Teaches you step by step and watches your focus through the camera. It gets stern if you keep getting distracted.",
  friend: "A casual friend with real reactions: happy, sad, playful, even annoyed if you're rude. Camera optional.",
  therapist: "A calm, supportive listener. It's an AI, not a real therapist, and it never gets angry. Camera optional.",
  chat: "Talk about anything with a friendly, curious companion. Camera optional.",
};

let socket = null;
let retries = 0;
let greeted = false;
let tutorBubble = null;
let avatar = null;
let selfView = null;
let toastTimer = null;
let lastStatus = null;
let devices = null;          // latest microphone/speaker list from the app
let session = null;          // the running session, or null until the welcome form is sent
let emotion = "neutral";     // the AI's emotion shown on the face and mood chip
let replyEmotion = "neutral"; // emotion of the reply being streamed, for its label

import("./avatar.js")
  .then(({ createAvatar }) => {
    avatar = createAvatar($("stage"));
    avatar.setEmotion(emotion);
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

const connected = () => socket !== null && socket.readyState === WebSocket.OPEN;

function sendMessage(message) {
  if (!connected()) return false;
  socket.send(JSON.stringify(message));
  return true;
}

function sendAction(action) {
  return sendMessage({ action });
}

// ------------------------------------------------------------------ emotion

function applyEmotion(name) {
  const next = EMOTIONS[name] ? name : "neutral";
  if (next === emotion) return;
  emotion = next;
  const meta = emotionMeta(next);
  root.style.setProperty("--emo", meta.color);
  $("emotion-text").textContent = meta.label;
  avatar?.setEmotion(next);
}

function labelBubble(bubble, name) {
  const meta = emotionMeta(name);
  bubble.style.setProperty("--msg-emo", meta.color);
  bubble.querySelector(".who").textContent = `${bubble.dataset.persona} · ${meta.label}`;
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

function sessionTitle(s) {
  if (s.mode === "tutor") return `Studying ${s.subject || "any subject"}${s.name ? ` with ${s.name}` : ""}`;
  const mode = { friend: "Friend", therapist: "Therapist", chat: "Chat" }[s.mode];
  return `${mode} mode${s.name ? ` with ${s.name}` : ""}`;
}

function onEvent(ev, live) {
  switch (ev.type) {
    case "user": {
      endTutor();
      const el = message("user", session?.name || "You");
      const text = document.createElement("span");
      text.textContent = ev.text;
      el.appendChild(text);
      append(el);
      break;
    }
    case "emotion":
      replyEmotion = ev.emotion;
      if (tutorBubble) labelBubble(tutorBubble, ev.emotion);
      if (live) applyEmotion(ev.emotion);
      break;
    case "tutor_chunk": {
      if (!tutorBubble) {
        tutorBubble = message("tutor", "");
        tutorBubble.dataset.persona = session?.persona || "Tutor";
        labelBubble(tutorBubble, replyEmotion);
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
      chip.textContent = ev.reason;
      append(chip);
      if (live) {
        showToast(ev.reason);
        avatar?.react("nudge");
      }
      break;
    }
    case "session": {
      endTutor();
      const divider = document.createElement("div");
      divider.className = "divider";
      divider.textContent = sessionTitle(ev.session);
      append(divider);
      setSession(ev.session);
      if (live) {
        closeWelcome();
        avatar?.react("hello");
      }
      break;
    }
  }
}

// ------------------------------------------------------------------ welcome form + session

function setSession(next) {
  session = next;
  const chip = $("session-chip");
  chip.hidden = !session;
  if (session) {
    const what = session.mode === "tutor" ? session.subject || "Tutor" : session.persona;
    $("session-text").textContent = [session.name, what].filter(Boolean).join(" · ");
  }
  updateWelcome();
}

function loadWelcomePrefs() {
  try {
    return JSON.parse(localStorage.getItem(WELCOME_KEY) || "{}");
  } catch {
    return {};
  }
}

function saveWelcomePrefs(prefs) {
  try {
    localStorage.setItem(WELCOME_KEY, JSON.stringify(prefs));
  } catch {
    // private window: the form just won't be prefilled next time
  }
}

function readChoice() {
  const value = $("w-choice").value;
  if (!value) return null;
  if (!value.startsWith("tutor:")) return { mode: value, subject: "", other: false };
  const other = value === "tutor:other";
  return { mode: "tutor", subject: other ? $("w-other").value.trim() : value.slice(6), other };
}

function updateWelcome() {
  const choice = readChoice();
  const camera = $("w-camera").checked;
  $("w-other-field").hidden = !choice?.other;
  const note = $("w-note");
  note.hidden = !choice;
  if (choice) {
    note.textContent = MODE_NOTES[choice.mode];
    note.dataset.mode = choice.mode;
  }
  $("w-camera-note").textContent = choice?.mode === "tutor"
    ? (camera ? "Needed for focus tracking and nudges" : "Focus tracking stays off without the camera")
    : (camera ? "I'll notice your expressions and be a little warmer" : "Optional. I'll keep a calmer, more neutral tone");
  const start = $("w-start");
  if (start.dataset.pending) return;
  const ready = $("w-name").value.trim() && choice && (!choice.other || choice.subject);
  start.disabled = !(ready && connected());
  start.textContent = !connected() ? "Connecting…" : session ? "Switch" : "Start";
}

function openWelcome() {
  const prefs = loadWelcomePrefs();
  const select = $("w-choice");
  let choice = prefs.choice || "";
  let other = prefs.other || "";
  if (session) {
    choice = session.mode;
    if (session.mode === "tutor") {
      const preset = [...select.options].some((o) => o.value === `tutor:${session.subject}`);
      choice = preset ? `tutor:${session.subject}` : session.subject ? "tutor:other" : "";
      other = preset ? "" : session.subject;
    }
  }
  $("w-name").value = session?.name ?? prefs.name ?? "";
  select.value = [...select.options].some((o) => o.value === choice) ? choice : "";
  $("w-other").value = other;
  $("w-camera").checked = lastStatus ? lastStatus.camera_enabled !== false : prefs.camera !== false;
  delete $("w-start").dataset.pending;
  welcome.hidden = false;
  for (const el of document.querySelectorAll(".topbar, .stage-ui, .panel, #selfview, #audio-menu")) el.inert = true;
  setAudioMenu(false);
  updateWelcome();
  ($("w-name").value ? select : $("w-name")).focus();
}

function closeWelcome() {
  if (welcome.hidden) return;
  welcome.hidden = true;
  delete $("w-start").dataset.pending;
  for (const el of document.querySelectorAll(".topbar, .stage-ui, .panel, #selfview, #audio-menu")) el.inert = false;
}

for (const id of ["w-name", "w-choice", "w-other", "w-camera"]) {
  $(id).addEventListener("input", updateWelcome);
  $(id).addEventListener("change", updateWelcome);
}
$("w-choice").addEventListener("change", () => {
  if (readChoice()?.other) $("w-other").focus();
});

$("welcome-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const choice = readChoice();
  const start = $("w-start");
  if (start.disabled || !choice) return;
  const name = $("w-name").value.trim();
  const camera = $("w-camera").checked;
  if (!sendMessage({ action: "start_session", name, mode: choice.mode, subject: choice.subject, camera })) return;
  saveWelcomePrefs({ name, choice: $("w-choice").value, other: $("w-other").value.trim(), camera });
  start.dataset.pending = "1";
  start.disabled = true;
  start.textContent = "Starting…"; // the popup closes when the app confirms the new session
});

$("session-chip").addEventListener("click", openWelcome);
$("w-privacy").addEventListener("click", () => setDrawer(true));

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

const capitalize = (text) => text.charAt(0).toUpperCase() + text.slice(1);

function applyFocus(s, cameraOn) {
  const f = s.focus;
  const tutoring = !s.session || s.session.mode === "tutor";
  const seeing = cameraOn && f && f.camera_ok;
  let kind = "nocam";
  $("meter-threshold").style.left = `${s.threshold}%`;

  if (!tutoring) {
    // Friend, therapist and chat: no focus score, just what the camera notices (the same words the AI gets).
    $("focus-label").textContent = "I can see";
    $("focus-state").textContent = seeing ? "Camera on" : "Camera off";
    $("presence-text").textContent = seeing ? capitalize(f.observation || "Looking for you") : "Nothing right now";
    $("focus-detail").textContent = seeing
      ? `No focus tracking in ${MODE_NAMES[s.session.mode]} mode. I just react to how you look.`
      : "I can't see you, so I'll keep things calm and neutral.";
    focusCard.className = "focus-card glass presence";
    $("focus-strikes").hidden = true;
    avatar?.setFocus("nocam");
    return;
  }

  $("focus-label").textContent = "Focus";
  if (seeing) {
    const score = Math.round(f.score);
    kind = f.state === "DISTRACTED" ? "bad" : f.state === "UNFOCUSED" ? "warn" : "good";
    $("focus-score").textContent = score;
    $("meter-fill").style.width = `${score}%`;
    $("focus-state").textContent = (STATE_TEXT[f.state] || f.state) + (f.calibrated ? "" : " · calibrating");
    const onLesson = f.expression ? `Eyes on the lesson · ${f.expression}` : "Eyes on the lesson";
    $("focus-detail").textContent = f.description || (f.face ? onLesson : "No face in view");
  } else {
    $("focus-score").textContent = "--";
    $("meter-fill").style.width = "0%";
    $("focus-state").textContent = "Camera off";
    $("focus-detail").textContent = cameraOn ? "Waiting for the camera" : "Focus tracking is off while your camera is off";
  }
  focusCard.className = `focus-card glass ${kind}`;
  avatar?.setFocus(kind);

  const strikes = s.strikes || 0;
  const badge = $("focus-strikes");
  badge.hidden = !strikes;
  badge.classList.toggle("high", strikes >= 3);
  badge.textContent = strikes === 1 ? "1 distraction in 10 min"
    : `${strikes} distractions in 10 min · ${strikes >= 3 ? "tutor is mad" : "tutor is annoyed"}`;
}

function applyStatus(s) {
  lastStatus = s;
  const cameraOn = s.camera_enabled !== false;
  const micOn = s.mic_enabled !== false;

  if ((s.session?.started_at ?? null) !== (session?.started_at ?? null)) {
    setSession(s.session);
    if (s.session && !$("w-start").dataset.pending && welcome.hidden === false && !lastStatus.session) closeWelcome();
  }
  applyEmotion(s.emotion || "neutral");
  applyFocus(s, cameraOn);

  const mode = s.user_speaking ? "user" : s.tutor_speaking ? "speaking" : s.thinking ? "thinking" : micOn ? "listening" : "muted";
  $("mode").dataset.mode = mode;
  $("mode-text").textContent = MODE_TEXT[mode];
  avatar?.setMode(mode === "muted" ? "listening" : mode);

  setToggle($("mic-btn"), micOn, "microphone", "⌃⌥M");
  setToggle($("cam-btn"), cameraOn, "camera", "⌃⌥V");
  audioMenu.classList.toggle("mic-off", !micOn);
  syncSelfView(s);

  $("paused").hidden = !s.suppressed || !(s.session?.mode === "tutor");
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
  setLive("live-camera", cameraOn ? "local" : "off", cameraOn ? "On · only a few words reach the AI" : "Off · webcam released");
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
  else if (wasOpen) (welcome.hidden ? privacyBtn : $("w-privacy")).focus();
}

privacyBtn.addEventListener("click", () => setDrawer(!drawer.classList.contains("open")));
$("privacy-close").addEventListener("click", () => setDrawer(false));
scrim.addEventListener("click", () => setDrawer(false));
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (drawer.classList.contains("open")) setDrawer(false);
  else if (!audioMenu.hidden) setAudioMenu(false);
  else if (!welcome.hidden && session) closeWelcome(); // switching is optional; the first start isn't
});

// ------------------------------------------------------------------ connection + controls

function setConnected(live) {
  const conn = $("conn");
  conn.textContent = live ? "live" : "offline";
  conn.classList.toggle("live", live);
  actionButtons.forEach((b) => { b.disabled = !live; });
  audioMenuBtn.disabled = !live;
  if (!live) setAudioMenu(false);
  updateWelcome();
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
      convo.querySelectorAll(".msg, .nudge, .divider").forEach((n) => n.remove());
      empty.hidden = false;
      tutorBubble = null;
      data.events.forEach((ev) => onEvent(ev, false));
      endTutor();
      if (data.privacy) fillPrivacy(data.privacy);
      if (data.devices) devices = data.devices;
      setSession(data.session);
      applyStatus(data.status);
      renderDevices();
      convo.scrollTop = convo.scrollHeight;
      if (data.session) closeWelcome();
      else if (welcome.hidden) openWelcome();
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
