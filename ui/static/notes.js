// Notes section: a drawer on the left with your uploaded notes, drag-and-drop anywhere on the page, cards that expand
// into sections, links and suggested questions, and a holographic card beside the avatar while the AI reads a note
// or visits a link.

const $ = (id) => document.getElementById(id);
const ACCEPTED = /\.(pdf|png|jpe?g|webp|gif|bmp)$/i;
const ICON_TRASH = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4.5h6V7M6.5 7l1 13h9l1-13M10 11v6M14 11v6"/></svg>';
const ICON_PAGE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 13h7M9 17h5"/></svg>';
const ICON_LINK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 14a4 4 0 0 0 5.66 0l3-3a4 4 0 0 0-5.66-5.66l-1 1"/><path d="M14 10a4 4 0 0 0-5.66 0l-3 3a4 4 0 0 0 5.66 5.66l1-1"/></svg>';
const ICON_CHEVRON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>';
const LINK_CARD_MS = 7000;

export function createNotes({ send, ask, toast, hasSession, onDrawer }) {
  const drawer = $("notes");
  const button = $("notes-btn");
  const list = $("note-list");
  const overlay = $("drop-overlay");
  const holo = $("note-holo");
  const expanded = new Set();
  let notes = [];
  let statuses = new Map();   // id -> status, to announce "ready" once
  let focus = null;           // note the AI is teaching from
  let focusKey = "";
  let link = null;            // web page the AI is visiting (shown for a few seconds)
  let linkTimer = 0;
  let holoDismissed = "";
  let readingTimer = 0;
  let dragDepth = 0;

  // ------------------------------------------------------------------ drawer

  function setOpen(open) {
    drawer.classList.toggle("open", open);
    drawer.setAttribute("aria-hidden", String(!open));
    button.setAttribute("aria-expanded", String(open));
    button.classList.toggle("active-soft", open);
    onDrawer(open);
    if (open) $("notes-close").focus();
    else if (drawer.contains(document.activeElement)) button.focus();
  }

  button.addEventListener("click", () => setOpen(!drawer.classList.contains("open")));
  $("notes-close").addEventListener("click", () => setOpen(false));
  $("note-file").addEventListener("change", (e) => {
    upload([...e.target.files]);
    e.target.value = "";
  });

  // ------------------------------------------------------------------ upload + drag and drop

  async function upload(files) {
    for (const file of files) {
      if (!ACCEPTED.test(file.name) && !/^(application\/pdf|image\/)/.test(file.type)) {
        toast(`${file.name} isn't a PDF or an image`, "info");
        continue;
      }
      toast(`Adding ${file.name}…`, "info");
      try {
        const res = await fetch("/notes", {
          method: "POST",
          headers: { "X-Filename": encodeURIComponent(file.name), "Content-Type": file.type || "application/octet-stream" },
          body: file,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          toast(err.error || `Couldn't add ${file.name}`, "info");
        }
      } catch {
        toast("Couldn't reach the app to add your notes", "info");
      }
    }
  }

  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  window.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth++;
    overlay.hidden = false;
  });
  window.addEventListener("dragover", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();   // otherwise the browser opens the file instead
    e.dataTransfer.dropEffect = "copy";
  });
  window.addEventListener("dragleave", (e) => {
    if (!hasFiles(e)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) overlay.hidden = true;
  });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth = 0;
    overlay.hidden = true;
    const files = [...e.dataTransfer.files];
    if (files.length) {
      setOpen(true);
      upload(files);
    }
  });

  // ------------------------------------------------------------------ cards

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function linkChip(item) {
    const a = el("a", "note-link");
    a.href = item.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.title = `${item.url} (page ${item.page})`;
    a.innerHTML = ICON_LINK;
    a.appendChild(el("span", "", item.label));
    return a;
  }

  function askFromNote(question) {
    if (!ask(question)) return;
    toast(`Asked: ${question}`, "info");
    setOpen(false);
  }

  function explain(note, section) {
    if (!send({ action: "explain_note", note_id: note.id, ...(section ? { section } : {}) })) return;
    toast(section > 1 ? `Explaining “${note.title}” from section ${section}…` : `Opening “${note.title}”…`, "info");
    setOpen(false);
  }

  function card(note) {
    const ready = note.status === "ready";
    const open = ready && expanded.has(note.id);
    const node = el("article", `note-card ${note.status}${open ? " expanded" : ""}`);
    node.dataset.id = note.id;
    node.classList.toggle("focused", focus?.note_id === note.id);

    const thumb = el("div", "note-thumb");
    thumb.innerHTML = `${ICON_PAGE}<span class="note-scan"></span>`;
    if (note.thumb) {
      const img = el("img");
      img.alt = "";
      img.loading = "lazy";
      img.src = `/notes/${note.id}/pages/1.jpg`;
      img.onerror = () => img.remove();
      thumb.prepend(img);
    }
    if (note.pages) thumb.appendChild(el("span", "note-pages", `${note.pages} page${note.pages === 1 ? "" : "s"}`));

    const info = el("div", "note-info");
    const head = el("button", "note-head");
    head.type = "button";
    head.disabled = !ready;
    head.setAttribute("aria-expanded", String(open));
    head.appendChild(el("span", "note-meta", [note.subject, note.added].filter(Boolean).join(" · ")));
    head.appendChild(el("span", "note-title", note.title || note.filename));
    if (ready) head.insertAdjacentHTML("beforeend", `<span class="note-chev">${ICON_CHEVRON}</span>`);
    head.addEventListener("click", () => {
      if (expanded.has(note.id)) expanded.delete(note.id);
      else expanded.add(note.id);
      render();
    });
    info.appendChild(head);

    if (ready) {
      if (note.summary) info.appendChild(el("p", "note-summary", note.summary));
      if (note.topics?.length) {
        const topics = el("div", "note-topics");
        note.topics.forEach((t) => topics.appendChild(el("span", "", t)));
        info.appendChild(topics);
      }
      if (note.links?.length) {
        const links = el("div", "note-links");
        const shown = open ? note.links : note.links.slice(0, 3);
        shown.forEach((item) => links.appendChild(linkChip(item)));
        if (!open && note.links.length > 3) links.appendChild(el("span", "note-more-links", `+${note.links.length - 3}`));
        info.appendChild(links);
      }
      if (open) info.appendChild(details(note));
      else if (note.questions?.length) {
        const q = el("button", "note-try", `Ask: “${note.questions[0]}”`);
        q.type = "button";
        q.disabled = !hasSession();
        q.addEventListener("click", () => askFromNote(note.questions[0]));
        info.appendChild(q);
      }
    } else {
      const status = note.status === "failed" ? note.error || "Couldn't read this file" : note.progress || "Waiting to be read";
      const dots = note.status === "failed" || status.endsWith("…") ? "" : "…";
      info.appendChild(el("p", "note-status", `${status}${dots}`));
    }

    const actions = el("div", "note-actions");
    if (focus?.note_id === note.id) actions.appendChild(el("span", "note-live", "AI is using this"));
    const explainBtn = el("button", "note-explain", "Explain this");
    explainBtn.type = "button";
    explainBtn.disabled = !ready || !hasSession();
    explainBtn.title = !hasSession() ? "Start a session first" : !ready ? "Still reading this note" : "";
    explainBtn.addEventListener("click", () => explain(note));
    actions.append(explainBtn, deleteButton(note, node));
    info.appendChild(actions);
    node.append(thumb, info);
    return node;
  }

  function details(note) {
    const box = el("div", "note-details");
    if (note.sections?.length) {
      box.appendChild(el("p", "note-label", `${note.sections.length} section${note.sections.length === 1 ? "" : "s"}`));
      const ol = el("ol", "note-sections");
      note.sections.forEach((s, i) => {
        const li = el("li");
        const b = el("button", "note-section");
        b.type = "button";
        b.disabled = !hasSession();
        b.append(el("span", "section-num", String(i + 1)), el("span", "section-title", s.title),
          el("span", "section-page", `p.${s.page}`), el("span", "section-go", "Explain from here"));
        b.addEventListener("click", () => explain(note, i + 1));
        li.appendChild(b);
        ol.appendChild(li);
      });
      box.appendChild(ol);
    }
    if (note.questions?.length) {
      box.appendChild(el("p", "note-label", "Ask about it"));
      const qs = el("div", "note-questions");
      note.questions.forEach((question) => {
        const b = el("button", "note-question", question);
        b.type = "button";
        b.disabled = !hasSession();
        b.addEventListener("click", () => askFromNote(question));
        qs.appendChild(b);
      });
      box.appendChild(qs);
    }
    return box;
  }

  function deleteButton(note, node) {
    const remove = el("button", "note-delete");
    remove.type = "button";
    remove.innerHTML = ICON_TRASH;
    remove.setAttribute("aria-label", `Delete ${note.title || note.filename}`);
    remove.title = "Delete";
    let armed = 0;
    remove.addEventListener("click", async () => {
      if (!armed) {   // two clicks to delete, without a dialog
        remove.classList.add("armed");
        remove.textContent = "Delete?";
        armed = setTimeout(() => {
          armed = 0;
          remove.classList.remove("armed");
          remove.innerHTML = ICON_TRASH;
        }, 3000);
        return;
      }
      clearTimeout(armed);
      node.classList.add("leaving");
      const res = await fetch(`/notes/${note.id}`, { method: "DELETE" }).catch(() => null);
      if (!res || !res.ok) {
        node.classList.remove("leaving");
        toast("Couldn't delete that note", "info");
      }
    });
    return remove;
  }

  function render() {
    list.replaceChildren(...notes.map(card));
    $("note-empty").hidden = notes.length > 0;
    const count = $("notes-count");
    count.hidden = !notes.length;
    count.textContent = String(notes.length);
    button.classList.toggle("busy", notes.some((n) => n.status === "reading" || n.status === "queued"));
  }

  // ------------------------------------------------------------------ holographic card beside the avatar

  function setReading(reading, busyText, idleText) {
    if (reading) {
      holo.classList.add("reading");
      $("holo-state").textContent = busyText;
      clearTimeout(readingTimer);
      readingTimer = setTimeout(() => {
        holo.classList.remove("reading");
        $("holo-state").textContent = idleText;
      }, 2600);
    } else if (!holo.classList.contains("reading")) {
      $("holo-state").textContent = idleText;
    }
  }

  function renderHolo(reading) {
    if (link) {
      holo.hidden = false;
      holo.classList.add("site");
      holo.classList.toggle("failed", Boolean(link.error));
      $("holo-site-name").textContent = link.site;
      $("holo-title").textContent = link.error ? link.site : link.title;
      $("holo-section").textContent = link.error || link.url.replace(/^https?:\/\//, "");
      $("holo-page").textContent = "";
      $("holo-bar").style.width = reading ? "35%" : "100%";
      setReading(reading, "Opening a link", link.error ? "Couldn't open it" : "Read this page");
      return;
    }
    holo.classList.remove("site", "failed");
    const shown = focus && holoDismissed !== focusKey;
    holo.hidden = !shown;
    if (!shown) return;
    const note = notes.find((n) => n.id === focus.note_id);
    $("holo-title").textContent = focus.title;
    $("holo-section").textContent = focus.section
      ? `Section ${focus.section} of ${focus.sections} · ${focus.section_title}`
      : `Overview · ${focus.sections} sections`;
    $("holo-page").textContent = `Page ${focus.page}${note?.pages ? ` of ${note.pages}` : ""}`;
    $("holo-bar").style.width = `${Math.round((Math.max(focus.section, 0.3) / Math.max(1, focus.sections)) * 100)}%`;
    const img = $("holo-img");
    const src = `/notes/${focus.note_id}/pages/${focus.page}.jpg`;
    if (img.getAttribute("src") !== src) {
      img.hidden = false;
      img.onerror = () => { img.hidden = true; };
      img.src = src;
    }
    setReading(reading, "Reading your notes", "Teaching from your notes");
  }

  $("holo-close").addEventListener("click", () => {
    if (link) {
      link = null;
      clearTimeout(linkTimer);
    } else {
      holoDismissed = focusKey;
    }
    renderHolo(false);
  });

  // ------------------------------------------------------------------ API for app.js

  return {
    setNotes(next) {
      for (const note of next) {
        const before = statuses.get(note.id);
        if (before && before !== note.status) {
          if (note.status === "ready") {
            const extra = note.links?.length ? ` · ${note.links.length} link${note.links.length === 1 ? "" : "s"} found` : "";
            toast(`Notes ready · ${note.title}${extra}`, "info");
          }
          if (note.status === "failed") toast(`Couldn't read ${note.filename}: ${note.error}`, "info");
        }
      }
      statuses = new Map(next.map((n) => [n.id, n.status]));
      notes = next;
      render();
      renderHolo(false);
    },
    setFocus(next, reading = false) {
      if (next?.kind === "link") {
        link = next;
        clearTimeout(linkTimer);
        if (!reading) {
          linkTimer = setTimeout(() => {
            link = null;
            renderHolo(false);
          }, LINK_CARD_MS);
        }
        renderHolo(reading);
        return;
      }
      const key = next ? `${next.note_id}:${next.section}:${next.page}` : "";
      if (key === focusKey && !reading) return;
      const noteChanged = (focus?.note_id || "") !== (next?.note_id || "");
      focus = next;
      focusKey = key;
      if (noteChanged) render();
      renderHolo(reading);
    },
    refresh: render,
    isOpen: () => drawer.classList.contains("open"),
    close: () => setOpen(false),
  };
}
