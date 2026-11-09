import "./theme.js";
import { api } from "./api.js";
import { TextChat } from "./text.js";
import { AudioChat } from "./audio.js";
import { Dictation } from "./dictate.js";
import { Settings } from "./settings.js";
import { initSidebarResize } from "./sidebar-resize.js";
import { ICON_PENCIL, ICON_TRASH } from "./icons.js";

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
  sendBtn: document.getElementById("send-btn"),
  dictateBtn: document.getElementById("dictate-btn"),
  audioPanel: document.getElementById("audio-panel"),
  audioStatus: document.getElementById("audio-status"),
  emotionDebug: document.getElementById("emotion-debug-body"),
  micBar: document.getElementById("mic-level-bar"),
  muteBtn: document.getElementById("mute-btn"),
  webcamBtn: document.getElementById("webcam-btn"),
  webcamWindow: document.getElementById("webcam-window"),
  webcamHandle: document.getElementById("webcam-handle"),
  webcamPreview: document.getElementById("webcam-preview"),
  webcamOverlay: document.getElementById("webcam-overlay"),
  webcamCanvas: document.getElementById("webcam-canvas"),
  facialPanel: document.getElementById("facial-debug"),
  facialDebug: document.getElementById("facial-debug-body"),
  audioTextRow: document.getElementById("audio-text-row"),
  audioTextInput: document.getElementById("audio-text-input"),
  audioSendBtn: document.getElementById("audio-send-btn"),
  settingsPanel: document.getElementById("settings-panel"),
  themeSwatches: document.getElementById("theme-swatches"),
  modelOptions: document.getElementById("model-options"),
  pullModelForm: document.getElementById("pull-model-form"),
  pullModelInput: document.getElementById("pull-model-input"),
  pullModelStatus: document.getElementById("pull-model-status"),
  transcriptionOptions: document.getElementById("transcription-options"),
  faceOverlayOptions: document.getElementById("face-overlay-options"),
  webcamIntervalSlider: document.getElementById("webcam-interval-slider"),
  webcamIntervalValue: document.getElementById("webcam-interval-value"),
  webcamSizeSlider: document.getElementById("webcam-size-slider"),
  webcamSizeValue: document.getElementById("webcam-size-value"),
};

let mode = "text";
let currentChatId = null;

// Stopping generation before any reply content has streamed back prunes
// the now-childless message it was replying to server-side (see
// run_generation in server.py) rather than leaving a dangling empty
// turn -- so without this, whatever was typed would just be lost when
// the next "sync" makes it disappear from the transcript.
function restoreInput(textarea, text) {
  textarea.value = text;
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight}px`;
  textarea.focus();
}

const textChat = new TextChat(els.transcript, els.title, els.sendBtn);
textChat.onTitle = () => refreshSidebar();
textChat.onEmptyStop = (text) => restoreInput(els.textInput, text);

const audioChat = new AudioChat({
  transcriptEl: els.transcript,
  titleEl: els.title,
  statusEl: els.audioStatus,
  emotionEl: els.emotionDebug,
  micBarEl: els.micBar,
  muteBtn: els.muteBtn,
  sendBtn: els.audioSendBtn,
  webcamBtn: els.webcamBtn,
  webcamWindowEl: els.webcamWindow,
  webcamHandleEl: els.webcamHandle,
  webcamPreviewEl: els.webcamPreview,
  webcamOverlayEl: els.webcamOverlay,
  webcamCanvasEl: els.webcamCanvas,
  facialPanelEl: els.facialPanel,
  facialEl: els.facialDebug,
});
audioChat.onTitle = () => refreshSidebar();
audioChat.onEmptyStop = (text) => restoreInput(els.audioTextInput, text);

const dictation = new Dictation({ button: els.dictateBtn, textarea: els.textInput });

const settings = new Settings({
  themeContainer: els.themeSwatches,
  modelContainer: els.modelOptions,
  pullForm: els.pullModelForm,
  pullInput: els.pullModelInput,
  pullStatus: els.pullModelStatus,
  transcriptionContainer: els.transcriptionOptions,
  faceOverlayContainer: els.faceOverlayOptions,
  webcamIntervalSlider: els.webcamIntervalSlider,
  webcamIntervalValue: els.webcamIntervalValue,
  webcamSizeSlider: els.webcamSizeSlider,
  webcamSizeValue: els.webcamSizeValue,
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
  // Clicking a chat in the sidebar always means "show me this chat" --
  // even if Settings (or the other mode) is currently the visible panel.
  // Without this, loadChat/textChat.open below happened correctly in
  // the background, but the Settings panel stayed on screen covering it.
  setActiveTabButtons(mode);
  showPanelsFor(mode);

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

function wireInputRow(row, textarea, chat) {
  row.addEventListener("submit", (event) => {
    event.preventDefault();
    if (chat.generating) {
      chat.stop();
      return;
    }
    const text = textarea.value;
    textarea.value = "";
    textarea.style.height = "auto";
    chat.send(text);
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

wireInputRow(els.textInputRow, els.textInput, textChat);
wireInputRow(els.audioTextRow, els.audioTextInput, audioChat);

initSidebarResize(els.sidebar, els.sidebarResizer);

startNewChat();
