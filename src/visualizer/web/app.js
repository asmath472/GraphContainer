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
  const replaySlider = document.getElementById("debug-replay-slider");
  const replayLiveBtn = document.getElementById("debug-replay-live");
  const replayStatusEl = document.getElementById("debug-replay-status");

  const chatGraphSelect = document.getElementById("chat-graph-select");
  const chatModelSelect = document.getElementById("chat-model-select");
  const chatEmbeddingSelect = document.getElementById("chat-embedding-select");
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
  const RETRIEVING_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
  const GENERATING_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
  const RETRIEVING_SPINNER_INTERVAL_MS = 90;
  const MINI_SESSION_POLL_MS = 250;
  const DUPLICATE_SEND_WINDOW_MS = 400;
  const chatMiniRenderers = new Map();
  const chatMiniSnapshotCache = new Map();
  const chatMiniLayoutCache = new Map();
  const sessionReplayHistory = new Map();
  const replayCursorBySession = new Map();
  let chatMessageSeq = 0;

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
    network.setOptions({ physics: { enabled: true } });
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

  async function fetchSessionSubgraph(sessionId, customHops = hops) {
    const targetHops = Number.isFinite(Number(customHops)) ? Number(customHops) : hops;
    const res = await fetch(`/api/session/${sessionId}/subgraph?hops=${targetHops}`);
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

  function cloneReplayView(view) {
    return JSON.parse(
      JSON.stringify({
        nodes: view.nodes || [],
        edges: view.edges || [],
        progress: view.progress || {},
        highlighted: view.highlighted || {},
      })
    );
  }

  function getReplayHistory(sessionId) {
    if (!sessionReplayHistory.has(sessionId)) {
      sessionReplayHistory.set(sessionId, []);
    }
    return sessionReplayHistory.get(sessionId);
  }

  function getReplayCursor(sessionId) {
    if (!replayCursorBySession.has(sessionId)) return null;
    return replayCursorBySession.get(sessionId);
  }

  function setReplayStatusText(value) {
    if (!replayStatusEl) return;
    replayStatusEl.textContent = value;
  }

  function syncReplayControls(sessionId) {
    if (!replaySlider) return;
    if (!sessionId) {
      replaySlider.disabled = true;
      replaySlider.min = "0";
      replaySlider.max = "0";
      replaySlider.value = "0";
      if (replayLiveBtn) replayLiveBtn.disabled = true;
      setReplayStatusText("No replay snapshots");
      return;
    }

    const history = getReplayHistory(sessionId);
    if (!history.length) {
      replaySlider.disabled = true;
      replaySlider.min = "0";
      replaySlider.max = "0";
      replaySlider.value = "0";
      if (replayLiveBtn) replayLiveBtn.disabled = true;
      setReplayStatusText("No replay snapshots");
      return;
    }

    const maxIndex = history.length - 1;
    let cursor = getReplayCursor(sessionId);
    if (cursor !== null) {
      cursor = Math.max(0, Math.min(maxIndex, Number(cursor)));
      if (cursor >= maxIndex) {
        replayCursorBySession.delete(sessionId);
        cursor = null;
      } else {
        replayCursorBySession.set(sessionId, cursor);
      }
    }

    const effectiveIndex = cursor === null ? maxIndex : cursor;
    const entry = history[effectiveIndex];
    const progress = entry && entry.view && entry.view.progress ? entry.view.progress : {};
    const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
    const message = String(progress.message || "Processing");
    const mode = cursor === null ? "Live" : "Replay";

    replaySlider.disabled = history.length <= 1;
    replaySlider.min = "0";
    replaySlider.max = String(maxIndex);
    replaySlider.value = String(effectiveIndex);
    if (replayLiveBtn) replayLiveBtn.disabled = cursor === null;

    setReplayStatusText(
      `${mode} ${effectiveIndex + 1}/${history.length} · ${percent.toFixed(1)}% · ${message}`
    );
  }

  function recordReplaySnapshot(sessionId, view) {
    if (!sessionId || !view) return;
    const history = getReplayHistory(sessionId);
    const snapshot = cloneReplayView(view);
    const last = history.length ? history[history.length - 1] : null;
    const currentProgress = snapshot.progress || {};
    const currentPercent = Number(currentProgress.percent || 0);
    const currentMessage = String(currentProgress.message || "");
    const currentNodes = (snapshot.nodes || []).length;
    const currentEdges = (snapshot.edges || []).length;

    if (last) {
      const lastProgress = (last.view && last.view.progress) || {};
      const lastPercent = Number(lastProgress.percent || 0);
      const lastMessage = String(lastProgress.message || "");
      const lastNodes = ((last.view && last.view.nodes) || []).length;
      const lastEdges = ((last.view && last.view.edges) || []).length;
      if (
        Math.abs(currentPercent - lastPercent) < 0.0001 &&
        currentMessage === lastMessage &&
        currentNodes === lastNodes &&
        currentEdges === lastEdges
      ) {
        return;
      }
    }

    history.push({
      ts: Date.now(),
      view: snapshot,
    });

    const maxSnapshots = 240;
    if (history.length > maxSnapshots) {
      const removed = history.length - maxSnapshots;
      history.splice(0, removed);
      const cursor = getReplayCursor(sessionId);
      if (cursor !== null) {
        replayCursorBySession.set(sessionId, Math.max(0, Number(cursor) - removed));
      }
    }
  }

  function applyReplayEntry(sessionId, index) {
    if (!sessionId) return;
    const history = getReplayHistory(sessionId);
    if (!history.length) return;
    const safeIndex = Math.max(0, Math.min(history.length - 1, Number(index)));
    const entry = history[safeIndex];
    if (!entry || !entry.view) return;

    applySubgraph(entry.view, { incremental: false });
    applyGhostStyleToVisibleNodes();
    updateProgressBar(entry.view.progress);

    const nodeCount = (entry.view.nodes || []).length;
    const edgeCount = (entry.view.edges || []).length;
    statusEl.textContent =
      `session=${sessionId} | replay ${safeIndex + 1}/${history.length} | ` +
      `nodes=${nodeCount}, edges=${edgeCount}`;

    syncReplayControls(sessionId);
  }

  function showLatestReplay(sessionId, { incremental = false } = {}) {
    if (!sessionId) return;
    const history = getReplayHistory(sessionId);
    if (!history.length) return;
    replayCursorBySession.delete(sessionId);
    const latest = history[history.length - 1];
    if (!latest || !latest.view) return;

    applySubgraph(latest.view, { incremental });
    applyGhostStyleToVisibleNodes();
    updateProgressBar(latest.view.progress);

    const nodeCount = (latest.view.nodes || []).length;
    const edgeCount = (latest.view.edges || []).length;
    const hNodeCount = ((latest.view.highlighted || {}).nodes || 0);
    const hEdgeCount = ((latest.view.highlighted || {}).edges || 0);
    statusEl.textContent =
      `session=${sessionId} | displayed(${hops}-hop) nodes=${nodeCount}, edges=${edgeCount} | ` +
      `highlighted nodes=${hNodeCount}, edges=${hEdgeCount}`;

    syncReplayControls(sessionId);
  }

  async function refreshSessionView({ incremental = false } = {}) {
    if (!currentSession) return;
    const view = await fetchSessionSubgraph(currentSession);
    if (!view) return;
    if (!view.exists) {
      clearGraph();
      statusEl.textContent = `session ${currentSession} not found`;
      updateProgressBar({ percent: 0, current: 0, total: 0, message: "session not found" });
      syncReplayControls(currentSession);
      return;
    }

    recordReplaySnapshot(currentSession, view);
    const history = getReplayHistory(currentSession);
    const cursor = getReplayCursor(currentSession);
    const hasReplayCursor = cursor !== null && Number(cursor) < history.length - 1;
    if (hasReplayCursor) {
      applyReplayEntry(currentSession, cursor);
      return;
    }

    showLatestReplay(currentSession, { incremental });
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

  function makeChatMessageId() {
    chatMessageSeq += 1;
    return `msg_${Date.now().toString(36)}_${chatMessageSeq.toString(36)}`;
  }

  function ensureMessageId(message) {
    if (!message.id) {
      message.id = makeChatMessageId();
    }
    return message.id;
  }

  function updateMessageBodyText(messageId, text) {
    const body = document.getElementById(`msg-body-${messageId}`);
    if (body) {
      body.textContent = String(text || "");
    }
  }

  function toMiniGraphNode(node) {
    return {
      id: node.id,
      label: "",
      title: node.label || node.id,
      size: typeof node.size === "number" ? Math.max(5, Math.min(16, node.size)) : 10,
      color:
        node.color && typeof node.color === "object"
          ? node.color
          : {
              background: getCssVar("--ghost-node-bg", "#6b7280"),
              border: getCssVar("--ghost-node-border", "#9aa1ad"),
            },
      borderWidth: typeof node.borderWidth === "number" ? node.borderWidth : 1,
    };
  }

  function toMiniGraphEdge(edge) {
    return {
      id: edge.id || `${edge.from}->${edge.to}:${edge.relation || ""}`,
      from: edge.from,
      to: edge.to,
      arrows: "to",
      width: typeof edge.width === "number" ? edge.width : 1.5,
      color: {
        color: getCssVar("--network-edge-color", "rgba(189, 203, 224, 0.28)"),
        highlight: getCssVar("--accent-color", "#2f6feb"),
      },
      smooth: { type: "continuous" },
    };
  }

  function getMiniSnapshotStage(progress) {
    const message = String((progress && progress.message) || "").toLowerCase();
    if (message.includes("answer generated")) return "generated";
    if (message.includes("generating")) return "generating";
    if (message.includes("retrieval complete")) return "retrieved";
    return "retrieving";
  }

  function buildMiniSnapshotSignature(view) {
    const progress = view.progress || {};
    const progressMessage = String(progress.message || "");
    const nodeSnapshotSig = (view.nodes || []).map((node) => String(node.id)).join("|");
    const edgeSnapshotSig = (view.edges || [])
      .map((edge) => `${String(edge.from)}>${String(edge.to)}:${String(edge.relation || "")}`)
      .join("|");
    return `${progressMessage}::${nodeSnapshotSig}::${edgeSnapshotSig}`;
  }

  function cacheMiniRendererLayout(messageId, renderer) {
    if (!renderer || !renderer.network || !renderer.nodes) return;
    const nodeIds = renderer.nodes.getIds();
    if (!nodeIds.length) return;
    let positions = {};
    try {
      positions = renderer.network.getPositions(nodeIds);
    } catch (_) {
      return;
    }
    const viewPosition =
      typeof renderer.network.getViewPosition === "function" ? renderer.network.getViewPosition() : null;
    const scale = typeof renderer.network.getScale === "function" ? renderer.network.getScale() : null;
    chatMiniLayoutCache.set(messageId, {
      positions,
      viewPosition,
      scale,
    });
  }

  function ensureMiniSessionRenderer(messageId) {
    const container = document.getElementById(`msg-mini-graph-${messageId}`);
    if (!container) return null;

    let renderer = chatMiniRenderers.get(messageId);
    if (renderer && renderer.container === container) {
      return renderer;
    }
    if (renderer && renderer.network) {
      cacheMiniRendererLayout(messageId, renderer);
      renderer.network.destroy();
    }

    const miniNodes = new vis.DataSet([]);
    const miniEdges = new vis.DataSet([]);
    const miniNetwork = new vis.Network(
      container,
      { nodes: miniNodes, edges: miniEdges },
      {
        autoResize: true,
        interaction: { dragNodes: false, dragView: true, zoomView: true, hover: true },
        physics: {
          enabled: true,
          solver: "repulsion",
          repulsion: { nodeDistance: 70, centralGravity: 0.25 },
          stabilization: { iterations: 80, updateInterval: 20 },
        },
        nodes: {
          shape: "dot",
          font: { size: 0 },
        },
        edges: {
          arrows: { to: { enabled: true, scaleFactor: 0.45 } },
          smooth: { type: "continuous" },
          font: { size: 0 },
        },
      }
    );

    renderer = {
      container,
      nodes: miniNodes,
      edges: miniEdges,
      network: miniNetwork,
      nodePayloadById: new Map(),
      selectedNodeId: null,
      hasInitialView: false,
      lastLayoutSignature: "",
      lastSnapshotSignature: "",
      frozenAfterRetrieval: false,
    };
    miniNetwork.on("click", (params) => {
      if (!params.nodes.length) {
        renderer.selectedNodeId = null;
        renderMiniNodeDetail(messageId, null);
        return;
      }
      const nodeId = params.nodes[0];
      renderer.selectedNodeId = nodeId;
      renderMiniNodeDetail(messageId, renderer.nodePayloadById.get(nodeId) || null);
    });
    chatMiniRenderers.set(messageId, renderer);
    return renderer;
  }

  function teardownMiniSessionRenderer(messageId) {
    const renderer = chatMiniRenderers.get(messageId);
    if (!renderer) return;
    if (renderer.network) {
      cacheMiniRendererLayout(messageId, renderer);
      renderer.network.destroy();
    }
    chatMiniRenderers.delete(messageId);
  }

  async function renderMiniSessionSnapshot(messageId, sessionId, { allowFetch = true } = {}) {
    if (!sessionId) return;
    const renderer = ensureMiniSessionRenderer(messageId);
    if (!renderer) return;

    const applyViewToRenderer = (view, signature, { liveUpdate = false, stage = "retrieving" } = {}) => {
      renderer.lastSnapshotSignature = signature;
      renderer.frozenAfterRetrieval = stage !== "retrieving";

      renderer.nodePayloadById.clear();
      for (const rawNode of view.nodes || []) {
        if (!rawNode || rawNode.id === undefined || rawNode.id === null) continue;
        renderer.nodePayloadById.set(rawNode.id, rawNode);
      }

      const miniNodes = (view.nodes || []).map(toMiniGraphNode);
      const miniEdges = (view.edges || []).map(toMiniGraphEdge);

      const currentNodeIds = renderer.nodes.getIds();
      const currentPositions = currentNodeIds.length
        ? renderer.network.getPositions(currentNodeIds)
        : {};
      const cachedLayout = chatMiniLayoutCache.get(messageId);
      const cachedPositions =
        cachedLayout && cachedLayout.positions && typeof cachedLayout.positions === "object"
          ? cachedLayout.positions
          : {};

      let anchorX = 0;
      let anchorY = 0;
      let anchorCount = 0;
      for (const pos of Object.values(currentPositions)) {
        if (!pos) continue;
        if (!Number.isFinite(pos.x) || !Number.isFinite(pos.y)) continue;
        anchorX += pos.x;
        anchorY += pos.y;
        anchorCount += 1;
      }
      if (!anchorCount) {
        for (const pos of Object.values(cachedPositions)) {
          if (!pos) continue;
          if (!Number.isFinite(pos.x) || !Number.isFinite(pos.y)) continue;
          anchorX += pos.x;
          anchorY += pos.y;
          anchorCount += 1;
        }
      }
      const anchor =
        anchorCount > 0 ? { x: anchorX / anchorCount, y: anchorY / anchorCount } : null;

      const patchedMiniNodes = miniNodes.map((node) => {
        const knownPos = currentPositions[node.id] || cachedPositions[node.id];
        if (knownPos && Number.isFinite(knownPos.x) && Number.isFinite(knownPos.y)) {
          return {
            ...node,
            x: knownPos.x,
            y: knownPos.y,
          };
        }
        if (anchor) {
          return {
            ...node,
            x: anchor.x,
            y: anchor.y,
          };
        }
        return node;
      });

      const nextNodeIds = new Set(patchedMiniNodes.map((node) => node.id));
      const removedNodeIds = currentNodeIds.filter((id) => !nextNodeIds.has(id));
      if (removedNodeIds.length) renderer.nodes.remove(removedNodeIds);
      if (patchedMiniNodes.length) renderer.nodes.update(patchedMiniNodes);

      const currentEdgeIds = renderer.edges.getIds();
      const nextEdgeIds = new Set(miniEdges.map((edge) => edge.id));
      const removedEdgeIds = currentEdgeIds.filter((id) => !nextEdgeIds.has(id));
      if (removedEdgeIds.length) renderer.edges.remove(removedEdgeIds);
      if (miniEdges.length) renderer.edges.update(miniEdges);

      const nextLayoutSignature = `${patchedMiniNodes.length}:${miniEdges.length}`;
      if (patchedMiniNodes.length && !renderer.hasInitialView) {
        if (cachedLayout && cachedLayout.viewPosition && Number.isFinite(cachedLayout.scale)) {
          renderer.network.moveTo({
            position: cachedLayout.viewPosition,
            scale: cachedLayout.scale,
            animation: false,
          });
        } else {
          renderer.network.fit({ animation: false, padding: 26 });
          renderer.network.moveTo({ scale: 0.72, animation: false });
        }
        renderer.hasInitialView = true;
        renderer.lastLayoutSignature = nextLayoutSignature;
      }

      if (liveUpdate) {
        renderer.network.setOptions({
          physics: {
            enabled: true,
            solver: "repulsion",
            repulsion: { nodeDistance: 70, centralGravity: 0.25 },
            stabilization: { iterations: 80, updateInterval: 20 },
          },
        });
        if (typeof renderer.network.stabilize === "function") {
          renderer.network.stabilize(40);
        }
      }
      renderer.network.setOptions({ physics: { enabled: true } });

      const updatedNodeIds = renderer.nodes.getIds();
      if (updatedNodeIds.length) {
        chatMiniLayoutCache.set(messageId, {
          positions: renderer.network.getPositions(updatedNodeIds),
          viewPosition:
            typeof renderer.network.getViewPosition === "function"
              ? renderer.network.getViewPosition()
              : null,
          scale: typeof renderer.network.getScale === "function" ? renderer.network.getScale() : null,
        });
      }

      if (renderer.selectedNodeId && renderer.nodePayloadById.has(renderer.selectedNodeId)) {
        renderer.network.selectNodes([renderer.selectedNodeId]);
        renderMiniNodeDetail(messageId, renderer.nodePayloadById.get(renderer.selectedNodeId));
      } else {
        renderer.selectedNodeId = null;
        renderMiniNodeDetail(messageId, null);
      }

      const label = document.getElementById(`msg-mini-session-id-${messageId}`);
      if (label) {
        label.textContent = `session: ${sessionId}`;
      }
    };

    const cached = chatMiniSnapshotCache.get(messageId);
    if (cached && cached.sessionId === sessionId) {
      if (renderer.lastSnapshotSignature !== cached.signature) {
        applyViewToRenderer(cached.view, cached.signature, {
          liveUpdate: false,
          stage: cached.stage,
        });
      }
      if (!allowFetch || cached.frozen) {
        return { stage: cached.stage, progress: cached.view.progress || {} };
      }
    } else if (!allowFetch) {
      return;
    }

    const view = await fetchSessionSubgraph(sessionId, 0);
    if (!view || !view.exists) return;

    const progress = view.progress || {};
    const stage = getMiniSnapshotStage(progress);
    const signature = buildMiniSnapshotSignature(view);
    if (renderer.lastSnapshotSignature === signature) {
      return { stage, progress };
    }

    applyViewToRenderer(view, signature, {
      liveUpdate: stage === "retrieving",
      stage,
    });

    chatMiniSnapshotCache.set(messageId, {
      sessionId,
      signature,
      stage,
      frozen: stage !== "retrieving",
      view: JSON.parse(
        JSON.stringify({
          nodes: view.nodes || [],
          edges: view.edges || [],
          progress: view.progress || {},
        })
      ),
    });

    return { stage, progress };
  }

  function getActiveChatSession() {
    return chatSessions.find((session) => session.id === activeChatId) || null;
  }

  function buildChatSubtitle(session) {
    return `${session.graph} · ${session.model} · ${session.embedding} · ${session.retrieval}`;
  }

  function getSessionPreview(session) {
    const lastMessage =
      [...session.messages].reverse().find((msg) => msg.role === "user") ||
      session.messages[session.messages.length - 1];
    if (!lastMessage || !lastMessage.text) return "Start a conversation.";
    return String(lastMessage.text).replace(/\s+/g, " ").trim().slice(0, 42);
  }

  function syncChatControls(session) {
    if (!session) return;
    if (chatGraphSelect) chatGraphSelect.value = session.graph;
    if (chatModelSelect) chatModelSelect.value = session.model;
    if (chatEmbeddingSelect) chatEmbeddingSelect.value = session.embedding;
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

  function ensureElement(parent, selector, tagName, className) {
    let el = parent.querySelector(selector);
    if (el) return el;
    el = document.createElement(tagName);
    el.className = className;
    parent.appendChild(el);
    return el;
  }

  function dropMiniMessageState(messageId) {
    teardownMiniSessionRenderer(messageId);
    chatMiniSnapshotCache.delete(messageId);
    chatMiniLayoutCache.delete(messageId);
  }

  function dropMiniStateForSession(session) {
    if (!session || !Array.isArray(session.messages)) return;
    for (const message of session.messages) {
      const messageId = message && message.id ? String(message.id) : "";
      if (!messageId) continue;
      dropMiniMessageState(messageId);
    }
  }

  function syncChatMessageElement(message) {
    const messageId = ensureMessageId(message);
    const roleClass = message.role === "user" ? "user" : "assistant";
    const kindClass = message.kind ? ` msg-${message.kind}` : "";
    let msgEl = chatThread.querySelector(`.msg[data-message-id="${messageId}"]`);
    if (!msgEl) {
      msgEl = document.createElement("div");
      msgEl.dataset.messageId = messageId;
    }
    msgEl.className = `msg ${roleClass}${kindClass}`;

    const roleEl = ensureElement(msgEl, ".msg-role", "div", "msg-role");
    roleEl.textContent = message.role === "user" ? "You" : "Assistant";

    const bodyEl = ensureElement(msgEl, `.msg-body#msg-body-${messageId}`, "div", "msg-body");
    bodyEl.id = `msg-body-${messageId}`;
    bodyEl.textContent = String(message.text || "");

    const hasMiniGraph = message.role === "assistant" && Boolean(message.visualSessionId);
    let miniWrap = msgEl.querySelector(".msg-mini-session-wrap");
    if (hasMiniGraph) {
      if (!miniWrap) {
        miniWrap = document.createElement("div");
        miniWrap.className = "msg-mini-session-wrap";

        const miniGraph = document.createElement("div");
        miniGraph.className = "msg-mini-session-graph";
        miniGraph.id = `msg-mini-graph-${messageId}`;

        const miniSessionId = document.createElement("div");
        miniSessionId.className = "msg-mini-session-id";
        miniSessionId.id = `msg-mini-session-id-${messageId}`;
        miniSessionId.textContent = `session: ${message.visualSessionId}`;

        const miniNodeDetail = document.createElement("pre");
        miniNodeDetail.className = "msg-mini-session-node-detail is-placeholder";
        miniNodeDetail.id = `msg-mini-node-detail-${messageId}`;
        miniNodeDetail.textContent = "Click a node to inspect node text and metadata.";

        miniWrap.appendChild(miniGraph);
        miniWrap.appendChild(miniSessionId);
        miniWrap.appendChild(miniNodeDetail);
        msgEl.appendChild(miniWrap);
      }
    } else if (miniWrap) {
      miniWrap.remove();
      teardownMiniSessionRenderer(messageId);
    }

    return msgEl;
  }

  function renderChatThread() {
    if (!chatThread) return;
    const session = getActiveChatSession();
    if (!session) {
      const existing = chatThread.querySelectorAll(".msg[data-message-id]");
      for (const el of existing) {
        const messageId = el.dataset.messageId || "";
        if (messageId) teardownMiniSessionRenderer(messageId);
      }
      chatThread.innerHTML = "";
      return;
    }

    const renderedMessageIds = new Set();

    for (const message of session.messages) {
      const messageId = ensureMessageId(message);
      renderedMessageIds.add(messageId);
      const msgEl = syncChatMessageElement(message);
      chatThread.appendChild(msgEl);

      if (message.role === "assistant" && message.visualSessionId) {
        renderMiniSessionSnapshot(messageId, message.visualSessionId, { allowFetch: false });
      }
    }

    const existingMessageEls = chatThread.querySelectorAll(".msg[data-message-id]");
    for (const el of existingMessageEls) {
      const messageId = el.dataset.messageId || "";
      if (!messageId || renderedMessageIds.has(messageId)) continue;
      teardownMiniSessionRenderer(messageId);
      el.remove();
    }

    for (const existingMessageId of chatMiniRenderers.keys()) {
      if (!renderedMessageIds.has(existingMessageId)) {
        teardownMiniSessionRenderer(existingMessageId);
      }
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
      model: (chatModelSelect && chatModelSelect.value) || "gpt-5-nano",
      embedding: (chatEmbeddingSelect && chatEmbeddingSelect.value) || "bge:BAAI/bge-m3",
      retrieval: (chatRetrievalSelect && chatRetrievalSelect.value) || "one-hop",
      updatedAt: Date.now(),
      messages: [
        {
          id: makeChatMessageId(),
          role: "assistant",
          text: "A new graph chat is ready. Send a message to query the loaded graph.",
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
    const removed = chatSessions.find((session) => session.id === chatId) || null;
    chatSessions = chatSessions.filter((session) => session.id !== chatId);
    if (removed) {
      dropMiniStateForSession(removed);
    }
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

  async function sendChatMessage() {
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

    const historyForApi = session.messages.map((message) => ({
      role: message.role,
      content: String(message.text || ""),
    }));

    session.messages.push({ id: makeChatMessageId(), role: "user", text });
    maybeRenameNewChat(session, text);
    const retrievalAssistant = {
      id: makeChatMessageId(),
      role: "assistant",
      kind: "retrieval",
      text: `Retrieving ${RETRIEVING_SPINNER_FRAMES[0]}`,
      visualSessionId: null,
    };
    let answerAssistant = null;
    let generatingSpinnerTimer = null;
    let hasGeneratingStarted = false;

    function ensureAnswerAssistant() {
      if (answerAssistant) return answerAssistant;
      answerAssistant = {
        id: makeChatMessageId(),
        role: "assistant",
        kind: "answer",
        text: `Generating ${GENERATING_SPINNER_FRAMES[0]}`,
      };
      session.messages.push(answerAssistant);
      return answerAssistant;
    }
    session.messages.push(retrievalAssistant);
    session.updatedAt = Date.now();

    if (chatInput) chatInput.value = "";
    renderChatHeader(session);
    renderChatSessionList();
    renderChatThread();

    let spinnerIndex = 0;
    let retrievalSpinnerTimer = setInterval(() => {
      spinnerIndex = (spinnerIndex + 1) % RETRIEVING_SPINNER_FRAMES.length;
      const nextText = `Retrieving ${RETRIEVING_SPINNER_FRAMES[spinnerIndex]}`;
      retrievalAssistant.text = nextText;
      updateMessageBodyText(retrievalAssistant.id, nextText);
    }, RETRIEVING_SPINNER_INTERVAL_MS);

    function startGeneratingPhase() {
      if (hasGeneratingStarted) return;
      hasGeneratingStarted = true;

      if (retrievalSpinnerTimer) {
        clearInterval(retrievalSpinnerTimer);
        retrievalSpinnerTimer = null;
      }
      retrievalAssistant.text = "Retrieval complete.";
      updateMessageBodyText(retrievalAssistant.id, retrievalAssistant.text);

      ensureAnswerAssistant();
      renderChatThread();

      let generatingSpinnerIndex = 0;
      generatingSpinnerTimer = setInterval(() => {
        if (!answerAssistant) return;
        generatingSpinnerIndex = (generatingSpinnerIndex + 1) % GENERATING_SPINNER_FRAMES.length;
        const nextText = `Generating ${GENERATING_SPINNER_FRAMES[generatingSpinnerIndex]}`;
        answerAssistant.text = nextText;
        updateMessageBodyText(answerAssistant.id, nextText);
      }, RETRIEVING_SPINNER_INTERVAL_MS);
    }

    let miniPollTimer = null;
    let miniPollInFlight = false;

    try {
      const createRes = await fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          metadata: {
            graph: session.graph,
            model: session.model,
            retrieval: session.retrieval,
            embedding: session.embedding,
          },
        }),
      });
      const createData = await createRes.json().catch(() => ({}));
      if (!createRes.ok || !createData.session_id) {
        throw new Error(createData.error || "Failed to create a new visual session.");
      }
      session.vizSessionId = String(createData.session_id);

      if (session.vizSessionId) {
        retrievalAssistant.visualSessionId = session.vizSessionId;
        renderChatThread();
        miniPollTimer = setInterval(async () => {
          if (miniPollInFlight) return;
          miniPollInFlight = true;
          try {
            const snapshot = await renderMiniSessionSnapshot(
              retrievalAssistant.id,
              retrievalAssistant.visualSessionId
            );
            if (snapshot && (snapshot.stage === "retrieved" || snapshot.stage === "generating")) {
              startGeneratingPhase();
              if (miniPollTimer) {
                clearInterval(miniPollTimer);
                miniPollTimer = null;
              }
            }
          } finally {
            miniPollInFlight = false;
          }
        }, MINI_SESSION_POLL_MS);
      }

      let embeddingProvider = "bge";
      let embeddingModel = "BAAI/bge-m3";
      if (session.embedding && typeof session.embedding === "string") {
        const rawEmbedding = session.embedding.trim();
        if (rawEmbedding.includes(":")) {
          const [provider, model] = rawEmbedding.split(":", 2);
          embeddingProvider = String(provider || embeddingProvider).trim().toLowerCase();
          embeddingModel = String(model || embeddingModel).trim();
        } else if (rawEmbedding) {
          embeddingProvider = rawEmbedding.toLowerCase();
        }
      }

      const payload = {
        message: text,
        graph: session.graph,
        model: session.model,
        retrieval: session.retrieval,
        embedding_provider: embeddingProvider,
        embedding_model: embeddingModel,
        history: historyForApi,
        top_k: 5,
        session_id: session.vizSessionId || null,
      };

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || `Request failed with status ${res.status}`);
      }

      if (!hasGeneratingStarted) {
        startGeneratingPhase();
      }
      if (generatingSpinnerTimer) {
        clearInterval(generatingSpinnerTimer);
        generatingSpinnerTimer = null;
      }
      ensureAnswerAssistant();
      answerAssistant.text = String(data.answer || "I could not generate a response.");
      if (data.session_id) {
        session.vizSessionId = String(data.session_id);
        retrievalAssistant.visualSessionId = session.vizSessionId;
        currentSession = session.vizSessionId;
        if (sessionInput) sessionInput.value = session.vizSessionId;
        if (currentMode === "debug") {
          await refreshSessionView({ incremental: false });
          startEventStream(session.vizSessionId);
        } else {
          stopEventStream();
        }
      }
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      retrievalAssistant.text = "Retrieval failed.";
      ensureAnswerAssistant();
      answerAssistant.text = `Error: ${message}`;
    } finally {
      if (retrievalSpinnerTimer) {
        clearInterval(retrievalSpinnerTimer);
      }
      if (generatingSpinnerTimer) {
        clearInterval(generatingSpinnerTimer);
      }
      if (miniPollTimer) {
        clearInterval(miniPollTimer);
      }
      session.updatedAt = Date.now();
      renderChatHeader(session);
      renderChatSessionList();
      renderChatThread();
    }
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
    if (chatEmbeddingSelect) {
      chatEmbeddingSelect.onchange = () =>
        updateActiveChatSetting("embedding", chatEmbeddingSelect.value);
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

  function renderMiniNodeDetail(messageId, node) {
    const detail = document.getElementById(`msg-mini-node-detail-${messageId}`);
    if (!detail) return;
    if (!node) {
      detail.classList.add("is-placeholder");
      detail.textContent = "Click a node to inspect node text and metadata.";
      return;
    }

    const payload = {
      id: node.id,
      label: node.label || "",
      type: node.node_type || node.group || "",
      text: node.node_text || "",
      metadata: parseNodeMeta(node.node_meta),
    };
    detail.classList.remove("is-placeholder");
    detail.textContent = JSON.stringify(payload, null, 2);
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
      syncReplayControls("");
      return;
    }
    replayCursorBySession.delete(currentSession);
    syncReplayControls(currentSession);
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
    syncReplayControls("");
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

  if (replaySlider) {
    replaySlider.addEventListener("input", () => {
      if (!currentSession) return;
      const history = getReplayHistory(currentSession);
      if (!history.length) return;
      const targetIndex = Math.max(0, Math.min(history.length - 1, Number(replaySlider.value || 0)));
      if (targetIndex >= history.length - 1) {
        showLatestReplay(currentSession, { incremental: false });
        return;
      }
      replayCursorBySession.set(currentSession, targetIndex);
      applyReplayEntry(currentSession, targetIndex);
    });
  }

  if (replayLiveBtn) {
    replayLiveBtn.onclick = () => {
      if (!currentSession) return;
      showLatestReplay(currentSession, { incremental: false });
    };
  }

  initTheme();
  initChatUi();
  clearGraph();
  syncReplayControls("");
  setMode("chat");
})();
