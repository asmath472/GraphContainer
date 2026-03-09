(async function () {
  const nodes = new vis.DataSet([]);
  const edges = new vis.DataSet([]);

  const debugView = document.getElementById("debug-view");
  const chatView = document.getElementById("chat-view");
  const modeDebugBtn = document.getElementById("mode-debug");
  const modeChatBtn = document.getElementById("mode-chat");
  const themeToggleBtn = document.getElementById("theme-toggle");
  const themeToggleLabel = document.getElementById("theme-toggle-label");

  const sidePanel = document.getElementById("sidepanel");
  const toggleBtn = document.getElementById("toggle-panel");
  const toggleIcon = document.getElementById("toggle-icon");
  const nodeLabelToggle = document.getElementById("toggle-node-labels");
  const edgeLabelToggle = document.getElementById("toggle-edge-labels");

  const sessionInput = document.getElementById("session-input");
  const applyBtn = document.getElementById("apply-session");
  const clearBtn = document.getElementById("clear-session");
  const statusEl = document.getElementById("session-status");
  const detailEl = document.getElementById("node-detail");
  const progressFill = document.getElementById("progress-fill");
  const progressText = document.getElementById("progress-text");

  const chatGraphSelect = document.getElementById("chat-graph-select");
  const chatModelSelect = document.getElementById("chat-model-select");
  const chatRetrievalSelect = document.getElementById("chat-retrieval-select");
  const chatSessionListEl = document.getElementById("chat-session-list");
  const newChatBtn = document.getElementById("new-chat-btn");
  const chatTitleEl = document.getElementById("chat-title");
  const chatSubtitleEl = document.getElementById("chat-subtitle");
  const chatThread = document.getElementById("chat-thread");
  const chatInput = document.getElementById("chat-input");
  const chatSendBtn = document.getElementById("chat-send");

  const network = new vis.Network(
    document.getElementById("network"),
    { nodes, edges },
    {
      autoResize: true,
      physics: {
        enabled: true,
        solver: "repulsion",
        repulsion: { nodeDistance: 150, centralGravity: 0.1 },
        stabilization: { iterations: 150, updateInterval: 25 },
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        dragNodes: true,
      },
      edges: {
        smooth: { type: "continuous" },
        arrows: { to: { enabled: true, scaleFactor: 0.5 } },
      },
      nodes: {
        shape: "dot",
      },
    }
  );

  let hops = 2;
  try {
    const configRes = await fetch("/api/config");
    const config = configRes.ok ? await configRes.json() : { default_hops: 2 };
    hops = Number(config.default_hops || 2);
  } catch (_) {
    hops = 2;
  }

  let currentSession = "";
  let eventSource = null;
  let currentMode = "debug";
  let currentTheme = "dark";

  let chatSessions = [];
  let activeChatId = null;
  let isChatInputComposing = false;
  let isChatTitleComposing = false;
  let pendingTitleCommit = false;
  let lastSendSignature = "";
  let lastSendAt = 0;

  const THEME_STORAGE_KEY = "graph-visualizer-theme";
  const CHAT_PLACEHOLDER_RESPONSE =
    "현재 채팅 모드는 UI 프로토타입입니다. 백엔드 연동은 아직 연결되지 않았습니다.";
  const DUPLICATE_SEND_WINDOW_MS = 400;

  function getCssVar(name, fallback = "") {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function getGhostNodeStyle() {
    const size = Number(getCssVar("--ghost-node-size", "8")) || 8;
    const background = getCssVar("--ghost-node-bg", "#6b7280");
    const border = getCssVar("--ghost-node-border", "#9aa1ad");
    return {
      size,
      borderWidth: 1,
      color: {
        background,
        border,
        highlight: { background, border },
      },
    };
  }

  function getRetrievedNodeSize() {
    return Number(getCssVar("--retrieved-node-size", "18")) || 18;
  }

  function isRetrievedNode(node) {
    if (!node || typeof node !== "object") return false;
    const color = node.color;
    const hasOverlayColor = Boolean(color && typeof color === "object" && color.background);
    const hasOverlayBorder = typeof node.borderWidth === "number" && node.borderWidth >= 3;
    return hasOverlayColor || hasOverlayBorder;
  }

  function styleNodeByRetrieval(node, extra = {}) {
    const retrieved = isRetrievedNode(node);
    const common = {
      ...node,
      ...extra,
      _retrieved: retrieved,
      fixed: false,
    };

    if (retrieved) {
      return {
        ...common,
        size: typeof common.size === "number" ? common.size : getRetrievedNodeSize(),
      };
    }

    return {
      ...common,
      ...getGhostNodeStyle(),
    };
  }

  function applyGhostStyleToVisibleNodes() {
    const ghost = getGhostNodeStyle();
    const updates = nodes
      .get({ filter: (item) => !item._retrieved })
      .map((item) => ({ id: item.id, ...ghost }));
    if (updates.length) {
      nodes.update(updates);
    }
  }

  function applyNetworkTheme() {
    const showNodeLabels = nodeLabelToggle ? nodeLabelToggle.checked : true;
    const showEdgeLabels = edgeLabelToggle ? edgeLabelToggle.checked : true;

    const accent = getCssVar("--accent-color", "#2f6feb");
    const edgeColor = getCssVar("--network-edge-color", "rgba(189, 203, 224, 0.28)");
    const nodeText = getCssVar("--graph-node-text", "#ffffff");
    const textMain = getCssVar("--text-main", "#e2e8f0");
    const ghost = getGhostNodeStyle();

    network.setOptions({
      edges: {
        color: { color: edgeColor, highlight: accent },
        font: { color: textMain, size: showEdgeLabels ? 11 : 0, face: "Inter" },
      },
      nodes: {
        size: ghost.size,
        borderWidth: ghost.borderWidth,
        color: ghost.color,
        font: { color: nodeText, size: showNodeLabels ? 12 : 0, face: "Inter" },
      },
    });

    applyGhostStyleToVisibleNodes();
    network.redraw();
  }

  function setTheme(theme) {
    currentTheme = theme === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", currentTheme);

    if (themeToggleLabel) {
      themeToggleLabel.textContent = currentTheme === "dark" ? "Dark Mode" : "Light Mode";
    }
    if (themeToggleBtn) {
      const nextTheme = currentTheme === "dark" ? "Light" : "Dark";
      themeToggleBtn.title = `Switch to ${nextTheme} Mode`;
    }

    try {
      localStorage.setItem(THEME_STORAGE_KEY, currentTheme);
    } catch (_) {}

    applyNetworkTheme();
  }

  function initTheme() {
    let initialTheme = "dark";
    try {
      const stored = localStorage.getItem(THEME_STORAGE_KEY);
      if (stored === "light" || stored === "dark") {
        initialTheme = stored;
      } else if (
        window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: light)").matches
      ) {
        initialTheme = "light";
      }
    } catch (_) {}

    setTheme(initialTheme);

    if (themeToggleBtn) {
      themeToggleBtn.onclick = () => {
        setTheme(currentTheme === "dark" ? "light" : "dark");
      };
    }
  }

  function clearGraph() {
    network.setOptions({ physics: { enabled: false } });
    nodes.clear();
    edges.clear();
  }

  function parseNodeMeta(nodeMeta) {
    if (typeof nodeMeta !== "string") return nodeMeta || {};
    try {
      return JSON.parse(nodeMeta);
    } catch (_) {
      return nodeMeta;
    }
  }

  function restartPhysics() {
    network.setOptions({
      physics: {
        enabled: true,
      },
    });
    if (typeof network.startSimulation === "function") {
      network.startSimulation();
    }
  }

  function applySubgraph(payload, { incremental = false } = {}) {
    const nextNodes = payload.nodes || [];
    const nextEdges = payload.edges || [];

    if (!incremental) {
      clearGraph();
      if (nextNodes.length) {
        nodes.add(
          nextNodes.map((node) =>
            styleNodeByRetrieval(node, {
              physics: true,
            })
          )
        );
      }
      if (nextEdges.length) edges.add(nextEdges);
      restartPhysics();
      return;
    }

    const currentNodeIds = nodes.getIds();
    const currentEdgeIds = edges.getIds();
    const currentNodeIdSet = new Set(currentNodeIds);
    const nextNodeIdSet = new Set(nextNodes.map((node) => node.id));
    const nextEdgeIdSet = new Set(nextEdges.map((edge) => edge.id));
    const currentPositions = currentNodeIds.length ? network.getPositions(currentNodeIds) : {};

    const patchedNodes = nextNodes.map((node) => {
      if (currentNodeIdSet.has(node.id) && currentPositions[node.id]) {
        return styleNodeByRetrieval(node, {
          x: currentPositions[node.id].x,
          y: currentPositions[node.id].y,
          physics: true,
        });
      }
      return styleNodeByRetrieval(node, { physics: true });
    });

    const removedNodeIds = currentNodeIds.filter((id) => !nextNodeIdSet.has(id));
    const removedEdgeIds = currentEdgeIds.filter((id) => !nextEdgeIdSet.has(id));
    if (removedNodeIds.length) nodes.remove(removedNodeIds);
    if (removedEdgeIds.length) edges.remove(removedEdgeIds);

    if (patchedNodes.length) nodes.update(patchedNodes);
    if (nextEdges.length) edges.update(nextEdges);
    restartPhysics();
  }

  async function fetchSessionSubgraph(sessionId) {
    const res = await fetch(`/api/session/${sessionId}/subgraph?hops=${hops}`);
    if (!res.ok) return null;
    return await res.json();
  }

  function updateProgressBar(progress) {
    const p = progress || {};
    let percent = Number(p.percent || 0);
    percent = Math.max(0, Math.min(100, percent));
    progressFill.style.width = `${percent}%`;
    progressText.innerHTML = `<span style="color:var(--accent-color); font-weight:bold;">${percent.toFixed(
      1
    )}%</span> - ${p.message || "Processing"}`;
  }

  async function refreshSessionView({ incremental = false } = {}) {
    if (!currentSession) return;
    const view = await fetchSessionSubgraph(currentSession);
    if (!view) return;
    if (!view.exists) {
      clearGraph();
      statusEl.textContent = `session ${currentSession} not found`;
      updateProgressBar({ percent: 0, current: 0, total: 0, message: "session not found" });
      return;
    }

    applySubgraph(view, { incremental });
    applyGhostStyleToVisibleNodes();
    updateProgressBar(view.progress);

    const nodeCount = (view.nodes || []).length;
    const edgeCount = (view.edges || []).length;
    const hNodeCount = ((view.highlighted || {}).nodes || 0);
    const hEdgeCount = ((view.highlighted || {}).edges || 0);
    statusEl.textContent =
      `session=${currentSession} | displayed(${hops}-hop) nodes=${nodeCount}, edges=${edgeCount} | ` +
      `highlighted nodes=${hNodeCount}, edges=${hEdgeCount}`;
  }

  function stopEventStream() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  }

  function startEventStream(sessionId) {
    stopEventStream();
    eventSource = new EventSource(`/api/session/${sessionId}/events`);

    eventSource.addEventListener("session_update", () => {
      refreshSessionView({ incremental: true });
    });

    eventSource.addEventListener("session_deleted", () => {
      clearGraph();
      statusEl.textContent = `session ${sessionId} deleted`;
      updateProgressBar({ percent: 0, current: 0, total: 0, message: "session deleted" });
      stopEventStream();
    });

    eventSource.onerror = () => {
      if (currentSession) {
        statusEl.textContent = `session=${currentSession} | waiting for updates...`;
      }
    };
  }

  function syncSidePanelUi() {
    const isCollapsed = sidePanel.classList.contains("collapsed");
    toggleIcon.textContent = isCollapsed ? "◀" : "▶";
    toggleBtn.style.right = isCollapsed ? "20px" : "430px";
  }

  function setMode(mode) {
    currentMode = mode === "chat" ? "chat" : "debug";
    const isDebug = currentMode === "debug";

    debugView.classList.toggle("active", isDebug);
    chatView.classList.toggle("active", !isDebug);
    chatView.setAttribute("aria-hidden", isDebug ? "true" : "false");

    modeDebugBtn.classList.toggle("active", isDebug);
    modeChatBtn.classList.toggle("active", !isDebug);

    if (isDebug) {
      requestAnimationFrame(() => {
        network.redraw();
        network.fit({ animation: false });
      });
      syncSidePanelUi();
    }
  }

  function makeChatId() {
    return `chat_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
  }

  function getActiveChatSession() {
    return chatSessions.find((session) => session.id === activeChatId) || null;
  }

  function buildChatSubtitle(session) {
    return `${session.graph} · ${session.model} · ${session.retrieval}`;
  }

  function getSessionPreview(session) {
    const lastMessage =
      [...session.messages].reverse().find((msg) => msg.role === "user") ||
      session.messages[session.messages.length - 1];
    if (!lastMessage || !lastMessage.text) return "대화를 시작해 보세요.";
    return String(lastMessage.text).replace(/\s+/g, " ").trim().slice(0, 42);
  }

  function syncChatControls(session) {
    if (!session) return;
    if (chatGraphSelect) chatGraphSelect.value = session.graph;
    if (chatModelSelect) chatModelSelect.value = session.model;
    if (chatRetrievalSelect) chatRetrievalSelect.value = session.retrieval;
  }

  function renderChatHeader(session) {
    if (!chatTitleEl || !chatSubtitleEl) return;
    if (!session) {
      chatTitleEl.textContent = "Graph Chat";
      chatSubtitleEl.textContent = "No active chat";
      return;
    }
    chatTitleEl.textContent = session.title;
    chatSubtitleEl.textContent = buildChatSubtitle(session);
  }

  function renderChatThread() {
    if (!chatThread) return;
    const session = getActiveChatSession();
    chatThread.innerHTML = "";
    if (!session) return;

    for (const message of session.messages) {
      const msgEl = document.createElement("div");
      msgEl.className = `msg ${message.role === "user" ? "user" : "assistant"}`;

      const roleEl = document.createElement("div");
      roleEl.className = "msg-role";
      roleEl.textContent = message.role === "user" ? "You" : "Assistant";

      const bodyEl = document.createElement("div");
      bodyEl.className = "msg-body";
      bodyEl.textContent = String(message.text || "");

      msgEl.appendChild(roleEl);
      msgEl.appendChild(bodyEl);
      chatThread.appendChild(msgEl);
    }

    chatThread.scrollTop = chatThread.scrollHeight;
  }

  function renderChatSessionList() {
    if (!chatSessionListEl) return;
    chatSessionListEl.innerHTML = "";

    for (const session of chatSessions) {
      const item = document.createElement("div");
      item.className = `chat-session-item${session.id === activeChatId ? " active" : ""}`;

      const openBtn = document.createElement("button");
      openBtn.type = "button";
      openBtn.className = "chat-session-open";
      openBtn.onclick = () => setActiveChatSession(session.id);

      const main = document.createElement("div");
      main.className = "chat-session-main";

      const title = document.createElement("div");
      title.className = "chat-session-title";
      title.textContent = session.title;

      const preview = document.createElement("div");
      preview.className = "chat-session-preview";
      preview.textContent = getSessionPreview(session);

      main.appendChild(title);
      main.appendChild(preview);
      openBtn.appendChild(main);

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "chat-session-remove";
      removeBtn.title = "Delete chat";
      removeBtn.textContent = "×";
      removeBtn.onclick = (event) => {
        event.stopPropagation();
        removeChatSession(session.id);
      };

      item.appendChild(openBtn);
      item.appendChild(removeBtn);
      chatSessionListEl.appendChild(item);
    }
  }

  function createChatSession({ select = true } = {}) {
    const session = {
      id: makeChatId(),
      title: "New Graph Chat",
      graph: (chatGraphSelect && chatGraphSelect.value) || "default",
      model: (chatModelSelect && chatModelSelect.value) || "gpt-4o",
      retrieval: (chatRetrievalSelect && chatRetrievalSelect.value) || "hybrid",
      updatedAt: Date.now(),
      messages: [
        {
          role: "assistant",
          text: "새 Graph Chat이 생성되었습니다. 메시지를 입력하면 여기에 누적됩니다.",
        },
      ],
    };

    chatSessions.unshift(session);
    if (select) {
      activeChatId = session.id;
    }
    return session;
  }

  function setActiveChatSession(chatId) {
    if (!chatSessions.some((session) => session.id === chatId)) return;
    activeChatId = chatId;
    const session = getActiveChatSession();
    syncChatControls(session);
    renderChatHeader(session);
    renderChatSessionList();
    renderChatThread();
  }

  function removeChatSession(chatId) {
    chatSessions = chatSessions.filter((session) => session.id !== chatId);
    if (!chatSessions.length) {
      createChatSession({ select: false });
    }
    if (!chatSessions.some((session) => session.id === activeChatId)) {
      activeChatId = chatSessions[0].id;
    }
    setActiveChatSession(activeChatId);
  }

  function updateActiveChatSetting(key, value) {
    const session = getActiveChatSession();
    if (!session) return;
    session[key] = value;
    session.updatedAt = Date.now();
    renderChatHeader(session);
    renderChatSessionList();
  }

  function maybeRenameNewChat(session, userText) {
    if (!session || session.title !== "New Graph Chat") return;
    const normalized = String(userText || "").replace(/\s+/g, " ").trim();
    if (normalized) {
      session.title = normalized.slice(0, 34);
    }
  }

  function commitActiveChatTitle() {
    if (!chatTitleEl) return;
    const session = getActiveChatSession();
    if (!session) return;

    const normalized = String(chatTitleEl.textContent || "")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 48);
    const nextTitle = normalized || "Untitled Chat";
    session.title = nextTitle;
    session.updatedAt = Date.now();

    chatTitleEl.textContent = nextTitle;
    renderChatSessionList();
  }

  function sendChatMessage() {
    const text = (chatInput && chatInput.value ? chatInput.value : "").trim();
    if (!text) return;

    const now = Date.now();
    const signature = `${activeChatId || "none"}::${text}`;
    if (signature === lastSendSignature && now - lastSendAt < DUPLICATE_SEND_WINDOW_MS) {
      return;
    }
    lastSendSignature = signature;
    lastSendAt = now;

    let session = getActiveChatSession();
    if (!session) {
      session = createChatSession({ select: true });
    }

    session.messages.push({ role: "user", text });
    maybeRenameNewChat(session, text);
    session.messages.push({ role: "assistant", text: CHAT_PLACEHOLDER_RESPONSE });
    session.updatedAt = Date.now();

    if (chatInput) chatInput.value = "";
    renderChatHeader(session);
    renderChatSessionList();
    renderChatThread();
  }

  function initChatUi() {
    createChatSession({ select: true });
    setActiveChatSession(activeChatId);

    if (newChatBtn) {
      newChatBtn.onclick = () => {
        createChatSession({ select: true });
        setActiveChatSession(activeChatId);
      };
    }

    if (chatGraphSelect) {
      chatGraphSelect.onchange = () => updateActiveChatSetting("graph", chatGraphSelect.value);
    }
    if (chatModelSelect) {
      chatModelSelect.onchange = () => updateActiveChatSetting("model", chatModelSelect.value);
    }
    if (chatRetrievalSelect) {
      chatRetrievalSelect.onchange = () =>
        updateActiveChatSetting("retrieval", chatRetrievalSelect.value);
    }

    if (chatSendBtn) {
      chatSendBtn.onclick = sendChatMessage;
    }

    if (chatInput) {
      chatInput.addEventListener("compositionstart", () => {
        isChatInputComposing = true;
      });
      chatInput.addEventListener("compositionend", () => {
        isChatInputComposing = false;
      });
      chatInput.addEventListener("keydown", (event) => {
        if (event.isComposing || isChatInputComposing || event.keyCode === 229) {
          return;
        }
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          sendChatMessage();
        }
      });
    }

    if (chatTitleEl) {
      chatTitleEl.addEventListener("compositionstart", () => {
        isChatTitleComposing = true;
      });
      chatTitleEl.addEventListener("compositionend", () => {
        isChatTitleComposing = false;
        if (pendingTitleCommit) {
          pendingTitleCommit = false;
          commitActiveChatTitle();
        }
      });
      chatTitleEl.addEventListener("keydown", (event) => {
        if (event.isComposing || isChatTitleComposing || event.keyCode === 229) {
          return;
        }
        if (event.key === "Enter") {
          event.preventDefault();
          chatTitleEl.blur();
        }
      });
      chatTitleEl.addEventListener("blur", () => {
        if (isChatTitleComposing) {
          pendingTitleCommit = true;
          return;
        }
        commitActiveChatTitle();
      });
    }
  }

  function toggleSidePanel() {
    sidePanel.classList.toggle("collapsed");
    syncSidePanelUi();
  }

  toggleBtn.onclick = toggleSidePanel;
  syncSidePanelUi();

  applyBtn.onclick = () => {
    currentSession = (sessionInput.value || "").trim();
    if (!currentSession) {
      stopEventStream();
      clearGraph();
      updateProgressBar({ percent: 0, current: 0, total: 0, message: "" });
      statusEl.textContent = "base graph view";
      return;
    }
    refreshSessionView({ incremental: false });
    startEventStream(currentSession);
  };

  clearBtn.onclick = () => {
    stopEventStream();
    currentSession = "";
    sessionInput.value = "";
    clearGraph();
    updateProgressBar({ percent: 0, current: 0, total: 0, message: "" });
    statusEl.textContent = "base graph view";
  };

  network.on("click", function (params) {
    if (!params.nodes.length) {
      detailEl.textContent = "Please select a node.";
      return;
    }

    if (sidePanel.classList.contains("collapsed")) {
      toggleSidePanel();
    }

    const node = nodes.get(params.nodes[0]);
    if (!node) {
      detailEl.textContent = "Selected node is unavailable.";
      return;
    }

    const payload = {
      id: node.id,
      label: node.label || "",
      type: node.node_type || node.group || "",
      text: node.node_text || "",
      metadata: parseNodeMeta(node.node_meta),
    };
    detailEl.textContent = JSON.stringify(payload, null, 2);
  });

  if (nodeLabelToggle) {
    nodeLabelToggle.onchange = () => applyNetworkTheme();
  }
  if (edgeLabelToggle) {
    edgeLabelToggle.onchange = () => applyNetworkTheme();
  }

  if (modeDebugBtn && modeChatBtn) {
    modeDebugBtn.onclick = () => setMode("debug");
    modeChatBtn.onclick = () => setMode("chat");
  }

  initTheme();
  initChatUi();
  clearGraph();
  setMode("chat");
})();
