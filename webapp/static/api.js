async function asJson(response) {
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  getSettings() {
    return fetch("/api/settings").then(asJson);
  },
  listChats(mode) {
    return fetch(`/api/chats/${mode}`).then(asJson);
  },
  newChat(mode) {
    return fetch(`/api/chats/${mode}`, { method: "POST" }).then(asJson);
  },
  loadChat(mode, chatId) {
    return fetch(`/api/chats/${mode}/${chatId}`).then(asJson);
  },
  renameChat(mode, chatId, title) {
    return fetch(`/api/chats/${mode}/${chatId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }).then(asJson);
  },
  deleteChat(mode, chatId) {
    return fetch(`/api/chats/${mode}/${chatId}`, { method: "DELETE" }).then(asJson);
  },
};
