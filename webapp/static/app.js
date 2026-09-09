import "./theme.js";
import { api } from "./api.js";
import { TextChat } from "./text.js";
import { AudioChat } from "./audio.js";
import { Dictation } from "./dictate.js";
import { Settings } from "./settings.js";
import { initSidebarResize } from "./sidebar-resize.js";

const ICON_PENCIL =
  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>';
const ICON_TRASH =
  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>';

const els = {
  sidebar: document.getElementById("sidebar"),
  sidebarResizer: document.getElementById("sidebar-resizer"),
  transcript: document.getElementById("transcript"),
  title: document.getElementById("chat-title"),
  chatList: document.getElementById("chat-list"),
  newChatBtn: document.getElementById("new-chat-btn"),
  modeButtons: document.querySelectorAll(".mode-btn"),
  textInputRow: document.getElementById("text-input-row"),
  textInput: document.getElementById("text-input"),
  dictateBtn: document.getElementById("dictate-btn"),
  audioPanel: document.getElementById("audio-panel"),
  audioStatus: document.getElementById("audio-status"),
  micBar: document.getElementById("mic-level-bar"),
  muteBtn: document.getElementById("mute-btn"),
  audioTextRow: document.getElementById("audio-text-row"),
  audioTextInput: document.getElementById("audio-text-input"),
  settingsPanel: document.getElementById("settings-panel"),
  themeSwatches: document.getElementById("theme-swatches"),
  modelOptions: document.getElementById("model-options"),
  transcriptionOptions: document.getElementById("transcription-options"),
};

let mode = "text";
let currentChatId = null;

const textChat = new TextChat(els.transcript, els.title);
textChat.onTitle = () => refreshSidebar();

const audioChat = new AudioChat({
  transcriptEl: els.transcript,
  titleEl: els.title,
  statusEl: els.audioStatus,
  micBarEl: els.micBar,
  muteBtn: els.muteBtn,
});
audioChat.onTitle = () => refreshSidebar();

const dictation = new Dictation({ button: els.dictateBtn, textarea: els.textInput });

const settings = new Settings({
  themeContainer: els.themeSwatches,
  modelContainer: els.modelOptions,
  transcriptionContainer: els.transcriptionOptions,
});

async function refreshSidebar() {
  const chats = await api.listChats(mode);
  els.chatList.innerHTML = "";
  for (const chat of chats) {
    els.chatList.appendChild(renderChatItem(chat));
  }
}

function renderChatItem(chat) {
  const li = document.createElement("li");
  li.className = "chat-item" + (chat.id === currentChatId ? " active" : "");

  const title = document.createElement("span");
  title.className = "title";
  title.textContent = chat.title;
  li.appendChild(title);

  // Absolutely positioned, not part of the row's flex flow -- so they
  // never take up space (and therefore never squeeze .title's available
  // width) except while actually shown on hover. A gradient behind them
  // fades the truncated title out rather than an abrupt cut.
  const actions = document.createElement("div");
  actions.className = "row-actions";

  const renameBtn = document.createElement("button");
  renameBtn.className = "row-btn";
  renameBtn.type = "button";
  renameBtn.title = "Rename";
  renameBtn.innerHTML = ICON_PENCIL;
  renameBtn.addEventListener("click", async (event) => {
    event.stopPropagation();
    const next = prompt("Rename chat", chat.title);
    if (next && next.trim()) {
      await api.renameChat(mode, chat.id, next.trim());
      if (chat.id === currentChatId) els.title.textContent = next.trim();
      refreshSidebar();
    }
  });
  actions.appendChild(renameBtn);

  const deleteBtn = document.createElement("button");
  deleteBtn.className = "row-btn";
  deleteBtn.type = "button";
  deleteBtn.title = "Delete";
  deleteBtn.innerHTML = ICON_TRASH;
  deleteBtn.addEventListener("click", async (event) => {
    event.stopPropagation();
    if (!confirm(`Delete "${chat.title}"?`)) return;
    await api.deleteChat(mode, chat.id);
    if (chat.id === currentChatId) {
      await startNewChat();
    } else {
      refreshSidebar();
    }
  });
  actions.appendChild(deleteBtn);

  li.appendChild(actions);
  li.addEventListener("click", () => openChat(chat.id));
  return li;
}

async function openChat(chatId) {
  currentChatId = chatId;
  const data = await api.loadChat(mode, chatId).catch(() => null);
  if (mode === "text") {
    await audioChat.close();
    await textChat.open(chatId, data);
  } else {
    textChat.close();
    await audioChat.open(chatId, data);
  }
  refreshSidebar();
}

async function startNewChat() {
  const { id } = await api.newChat(mode);
  await openChat(id);
}

function setActiveTabButtons(tab) {
  els.modeButtons.forEach((btn) => {
    const active = btn.dataset.mode === tab;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });
}

function showPanelsFor(tab) {
  els.transcript.hidden = tab === "settings";
  els.textInputRow.hidden = tab !== "text";
  els.audioPanel.hidden = tab !== "audio";
  els.settingsPanel.hidden = tab !== "settings";
}

// "Settings" isn't a chat mode -- it doesn't touch `mode` (which drives
// chat history + the text/audio WebSocket) at all, just overlays its own
// panel. Text/Audio's chat session keeps running quietly underneath, so
// switching back to whichever one was active resumes it with no reload.
async function switchMode(nextTab) {
  dictation.stop();

  if (nextTab === "settings") {
    setActiveTabButtons("settings");
    showPanelsFor("settings");
    settings.open();
    return;
  }

  setActiveTabButtons(nextTab);
  showPanelsFor(nextTab);

  if (nextTab === mode) return;

  mode = nextTab;
  await startNewChat();
}

els.modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => switchMode(btn.dataset.mode));
});

els.newChatBtn.addEventListener("click", () => startNewChat());

function wireInputRow(row, textarea, sendFn) {
  row.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = textarea.value;
    textarea.value = "";
    textarea.style.height = "auto";
    sendFn(text);
  });

  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      row.requestSubmit();
    }
  });

  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
  });
}

wireInputRow(els.textInputRow, els.textInput, (text) => textChat.send(text));
wireInputRow(els.audioTextRow, els.audioTextInput, (text) => audioChat.send(text));

initSidebarResize(els.sidebar, els.sidebarResizer);

startNewChat();
