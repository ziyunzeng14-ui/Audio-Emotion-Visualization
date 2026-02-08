// static/main.js

let globalData = null;        // 整个 JSON（自动 / 自定义 / 上传）
let currentSegmentKey = null; // 当前选中的“段落键”（name + index）
let segmentChart = null;      // Chart.js 柱状图实例
let globalRadarChart = null;  // Chart.js 雷达图实例
let currentPlayTimeout = null; //timeout used to stop playback at the segment end

// 自定义分段存储：内部仍按槽位 original / piano / duet（或当前存在 slots）
let customSegments = {
  original: [],
  piano: [],
  duet: [],
};

// slot -> displayName（用于 UI 显示）
let slotLabelMap = {
  original: "original",
  piano: "piano",
  duet: "duet",
};

// 只用于“显示”，把 "111 (original)" -> "111"
function cleanDisplayName(str) {
  const s = (str || "").toString().trim();
  return s.replace(/\s*\([^)]*\)\s*$/, "").trim();
}

// 当前 data 中存在的 slots（Version slot 下拉只显示这些）
let availableSlots = ["original", "piano", "duet"];

// Segment name 自定义输入框（动态注入）
let segNameCustomInput = null;
let segNameHintSpan = null;

document.addEventListener("DOMContentLoaded", () => {
  console.log(">>> main.js loaded");

  initUploadUI();
  initSegmentationUI();
  initAICommentary(); // 绑定 Ask AI 按钮

  // 默认：加载自动分段 demo 分析
  loadDataFromApi("/api/demo");
});


// ==================== 0. 通用：加载数据后统一刷新 ====================

function handleDataLoaded(data) {
  console.log("Handle data loaded:", data);
  globalData = data;

  // ✅ 每次加载新数据，都刷新 slot 名称映射 + 可用 slots
  refreshSlotLabelMapFromData(data);
  refreshAvailableSlotsFromData(data);
  ensureCustomSegmentsKeys();

  // ✅ UI 同步
  syncCustomSegVersionSelectOptions();
  ensureSegmentNameCustomInputAndHint();
  renderCustomSegList();

  showRawJson(data);
  renderStructureTimeline(data);
  initSegmentUIAndChart(data);
  buildGlobalRadar(data);

  if (window.setVersionsForVis) {
    window.setVersionsForVis(data);
  }
}

function loadDataFromApi(endpoint, payload) {
  const debugDiv = document.getElementById("debug");
  if (debugDiv) {
    debugDiv.textContent = "Loading data from " + endpoint + "...";
  }

  const fetchOptions = payload ? {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload),
  } : {};

  fetch(endpoint, fetchOptions)
    .then((response) => {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    })
    .then((data) => {
      handleDataLoaded(data);
      if (debugDiv) debugDiv.textContent = "Data loaded from " + endpoint + ".";
    })
    .catch((err) => {
      console.error("Error fetching from", endpoint, err);
      if (debugDiv) debugDiv.textContent = "Error loading data: " + err;
    });
}


// ==================== 0.5 上传 UI ====================

function initUploadUI() {
  const btnUpload = document.getElementById("btn-upload-analyse");
  const fileOriginal = document.getElementById("file-original");
  const filePiano = document.getElementById("file-piano");
  const fileDuet = document.getElementById("file-duet");

  const nameOriginal = document.getElementById("name-original");
  const namePiano = document.getElementById("name-piano");
  const nameDuet = document.getElementById("name-duet");

  const statusDiv = document.getElementById("upload-status");

  if (!btnUpload) return;

  btnUpload.addEventListener("click", () => {
    const formData = new FormData();
    let hasFile = false;

    if (fileOriginal && fileOriginal.files[0]) {
      formData.append("original", fileOriginal.files[0]);
      formData.append("label_original", (nameOriginal.value || "original").trim());
      hasFile = true;
    }
    if (filePiano && filePiano.files[0]) {
      formData.append("piano", filePiano.files[0]);
      formData.append("label_piano", (namePiano.value || "piano").trim());
      hasFile = true;
    }
    if (fileDuet && fileDuet.files[0]) {
      formData.append("duet", fileDuet.files[0]);
      formData.append("label_duet", (nameDuet.value || "duet").trim());
      hasFile = true;
    }

    if (!hasFile) {
      alert("Please choose at least one audio file.");
      return;
    }

    if (statusDiv) statusDiv.textContent = "Uploading & analysing uploaded audio ...";

    fetch("/api/upload_analyse", {
        method: "POST",
        body: formData
      })
      .then((res) => {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then((data) => {
        handleDataLoaded(data);

        currentSegmentKey = null;
        initSegmentUIAndChart(globalData);

        if (statusDiv) statusDiv.textContent = "Analysis finished (uploaded audio).";
      })
      .catch((err) => {
        console.error("Upload analyse error:", err);
        if (statusDiv) statusDiv.textContent = "Error: " + err;
      });
  });
}


// ==================== 1. 分段模式 UI（自动 / 自定义） ====================

function initSegmentationUI() {
  const btnAuto = document.getElementById("btn-auto-seg");
  const btnCustom = document.getElementById("btn-custom-seg");
  const panel = document.getElementById("custom-seg-panel");

  if (!btnAuto || !btnCustom || !panel) return;

  btnAuto.addEventListener("click", () => {
    btnAuto.classList.add("primary");
    btnCustom.classList.remove("primary");
    panel.style.display = "none";

    loadDataFromApi("/api/demo");
  });

  btnCustom.addEventListener("click", () => {
    btnCustom.classList.add("primary");
    btnAuto.classList.remove("primary");
    panel.style.display = "block";

    syncCustomSegVersionSelectOptions();
    ensureSegmentNameCustomInputAndHint();
    renderCustomSegList();
  });

  const btnAdd = document.getElementById("btn-add-seg");
  const btnApply = document.getElementById("btn-apply-custom");
  const btnClear = document.getElementById("btn-clear-custom");

  const versionSelect = document.getElementById("seg-version-select");
  const nameSelect = document.getElementById("seg-name-select");
  const startInput = document.getElementById("seg-start-input");
  const endInput = document.getElementById("seg-end-input");

  ensureSegmentNameCustomInputAndHint();
  syncCustomSegVersionSelectOptions();

  if (btnAdd) {
    btnAdd.addEventListener("click", () => {
      const slot = versionSelect.value;

      const segName = getSegmentNameValue(nameSelect);
      const start = parseFloat(startInput.value);
      const end = parseFloat(endInput.value);

      if (!segName) {
        alert("Please enter or select a segment name.");
        return;
      }

      if (Number.isNaN(start) || Number.isNaN(end)) {
        alert("Please enter start and end time in seconds.");
        return;
      }
      if (end <= start) {
        alert("End time must be greater than start time.");
        return;
      }

      if (!customSegments[slot]) customSegments[slot] = [];

      const overlapWith = findOverlapSegment(customSegments[slot], start, end);
      if (overlapWith) {
        alert(
          `Overlap detected in "${slotLabelMap[slot] || slot}".\n` +
          `New: ${start.toFixed(1)}s → ${end.toFixed(1)}s\n` +
          `Conflicts with: "${overlapWith.name}" (${overlapWith.start.toFixed(1)}s → ${overlapWith.end.toFixed(1)}s)\n\n` +
          `Please adjust start/end so segments do not overlap.`
        );
        return;
      }

      customSegments[slot].push({
        name: segName,
        start,
        end
      });
      customSegments[slot].sort((a, b) => a.start - b.start);

      startInput.value = "";
      endInput.value = "";
      if (segNameCustomInput) segNameCustomInput.value = "";

      renderCustomSegList();
    });
  }

  if (btnApply) {
    btnApply.addEventListener("click", () => {
      const versionsPayload = [];
      const labelMap = { ...slotLabelMap };

      const slotsToSubmit = (availableSlots && availableSlots.length) ? availableSlots : ["original", "piano", "duet"];
      slotsToSubmit.forEach((slot) => {
        const segs = customSegments[slot] || [];
        if (segs.length > 0) {
          versionsPayload.push({
            slot,
            label: labelMap[slot] || slot,
            segments: segs,
          });
        }
      });

      if (versionsPayload.length === 0) {
        alert("Please add at least one segment before applying custom segmentation.");
        return;
      }

      loadDataFromApi("/api/custom_segments", { versions: versionsPayload });
    });
  }

  if (btnClear) {
    btnClear.addEventListener("click", () => {
      const slotsToClear = (availableSlots && availableSlots.length) ? availableSlots : Object.keys(customSegments);
      slotsToClear.forEach((slot) => (customSegments[slot] = []));
      renderCustomSegList();
    });
  }

  renderCustomSegList();
}


// ==================== ✅ slots / labels 同步 + UI ====================

function refreshSlotLabelMapFromData(data) {
  const map = { original: "original", piano: "piano", duet: "duet" };
  if (data && Array.isArray(data.versions)) {
    data.versions.forEach((v) => {
      const slot = v.slot || v.name;
      const name = cleanDisplayName(v.name || slot);
      if (slot) map[slot] = name;
    });
  }
  slotLabelMap = map;
}

function refreshAvailableSlotsFromData(data) {
  if (data && Array.isArray(data.versions) && data.versions.length > 0) {
    const slots = data.versions.map((v) => v.slot || v.name).filter(Boolean);
    const seen = new Set();
    const uniq = [];
    slots.forEach((s) => {
      if (!seen.has(s)) {
        seen.add(s);
        uniq.push(s);
      }
    });
    availableSlots = uniq.length ? uniq : ["original", "piano", "duet"];
  } else {
    availableSlots = ["original", "piano", "duet"];
  }
}

function ensureCustomSegmentsKeys() {
  const slots = (availableSlots && availableSlots.length) ? availableSlots : ["original", "piano", "duet"];
  slots.forEach((slot) => {
    if (!customSegments[slot]) customSegments[slot] = [];
  });
}

function syncCustomSegVersionSelectOptions() {
  const versionSelect = document.getElementById("seg-version-select");
  if (!versionSelect) return;

  const prev = versionSelect.value || ((availableSlots && availableSlots[0]) || "original");
  versionSelect.innerHTML = "";

  const slotsToShow = (availableSlots && availableSlots.length) ? availableSlots : ["original", "piano", "duet"];

  slotsToShow.forEach((slot) => {
    const opt = document.createElement("option");
    opt.value = slot;
    const display = slotLabelMap[slot] || slot;
    opt.textContent = `${display}`;
    versionSelect.appendChild(opt);
  });

  versionSelect.value = slotsToShow.includes(prev) ? prev : slotsToShow[0];
}


// ==================== ✅ segment name 自定义输入 + 提示 ====================

function ensureSegmentNameCustomInputAndHint() {
  const nameSelect = document.getElementById("seg-name-select");
  if (!nameSelect) return;

  if (!segNameCustomInput) {
    segNameCustomInput = document.createElement("input");
    segNameCustomInput.type = "text";
    segNameCustomInput.placeholder = "or type a custom name";
    segNameCustomInput.id = "seg-name-custom-input";
    segNameCustomInput.style.padding = "4px 8px";
    segNameCustomInput.style.borderRadius = "6px";
    segNameCustomInput.style.border = "1px solid rgba(148, 163, 184, 0.6)";
    segNameCustomInput.style.background = "#020617";
    segNameCustomInput.style.color = "#e5e7eb";
    segNameCustomInput.style.fontSize = "0.85rem";
    segNameCustomInput.style.width = "180px";
    segNameCustomInput.style.marginLeft = "6px";

    nameSelect.insertAdjacentElement("afterend", segNameCustomInput);

    segNameCustomInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        const btnAdd = document.getElementById("btn-add-seg");
        if (btnAdd) btnAdd.click();
      }
    });
  }

  if (!segNameHintSpan) {
    segNameHintSpan = document.createElement("span");
    segNameHintSpan.textContent = "Custom overrides dropdown";
    segNameHintSpan.style.fontSize = "11px";
    segNameHintSpan.style.opacity = "0.75";
    segNameHintSpan.style.marginLeft = "8px";
    segNameHintSpan.style.whiteSpace = "nowrap";
    segNameHintSpan.style.color = "#cbd5f5";

    segNameCustomInput.insertAdjacentElement("afterend", segNameHintSpan);
  }
}

function getSegmentNameValue(nameSelect) {
  const custom = segNameCustomInput ? segNameCustomInput.value.trim() : "";
  if (custom) return custom;
  const sel = nameSelect ? (nameSelect.value || "").trim() : "";
  return sel || "";
}


// ==================== ✅ 重叠检测 ====================

function findOverlapSegment(existingSegs, start, end) {
  for (const s of existingSegs) {
    const sStart = Number(s.start);
    const sEnd = Number(s.end);
    if (!Number.isFinite(sStart) || !Number.isFinite(sEnd)) continue;
    if (start < sEnd && end > sStart) return s;
  }
  return null;
}


// ==================== 自定义分段列表渲染（显示版本名） ====================

function renderCustomSegList() {
  const container = document.getElementById("seg-list");
  if (!container) return;

  const slots = (availableSlots && availableSlots.length) ? availableSlots : ["original", "piano", "duet"];

  let html = "";
  slots.forEach((slot) => {
    const list = customSegments[slot] || [];
    const display = slotLabelMap[slot] || slot;

    html += `<h4>${display}</h4>`;
    if (list.length === 0) {
      html += `<p style="margin:0 0 4px 4px; opacity:0.6;">(no segments)</p>`;
    } else {
      html += "<ul>";
      list.forEach((seg) => {
        html += `<li><b>${seg.name}</b>: ${seg.start.toFixed(1)}s → ${seg.end.toFixed(1)}s</li>`;
      });
      html += "</ul>";
    }
  });

  container.innerHTML = html;
}


// -------------------------
// 2. 显示原始 JSON 文本
// -------------------------

function showRawJson(data) {
  const jsonContainer = document.getElementById("json-data");
  if (!jsonContainer) return;
  jsonContainer.textContent = JSON.stringify(data, null, 2);
}


// -------------------------
// 3. 结构时间线（可点击播放）
// -------------------------

function onTimelineSegmentClick(event) {
  const block = event.currentTarget;
  const slot = block.dataset.slot;
  const start = parseFloat(block.dataset.start || "0");
  const end = parseFloat(block.dataset.end || "0");

  const audio = document.getElementById("audio-player");
  const labelSpan = document.getElementById("current-play-label");
  if (!audio || !slot) return;

  if (currentPlayTimeout) {
    clearTimeout(currentPlayTimeout);
    currentPlayTimeout = null;
  }

  const srcUrl = `/audio/${slot}?t=${Date.now()}`;
  audio.src = srcUrl;
  audio.dataset.slot = slot;

  if (labelSpan) {
    const displayName = block.dataset.displayName || slot;
    labelSpan.textContent = `${displayName} [${start.toFixed(1)}s → ${end.toFixed(1)}s]`;
  }

  audio.currentTime = start;
  audio.play().catch((err) => console.warn("Audio play error:", err));

  if (end > start) {
    const durMs = (end - start) * 1000;
    currentPlayTimeout = setTimeout(() => {
      if (!audio.paused) audio.pause();
    }, durMs);
  }
}

function renderStructureTimeline(data) {
  const container = document.getElementById("structureTimeline");
  if (!container) return;

  container.innerHTML = "";
  if (!data || !data.versions) return;

  data.versions.forEach((version) => {
    const slot = version.slot || version.name;
    const displayName = version.name || slot;

    const row = document.createElement("div");
    row.className = "timeline-row";

    const label = document.createElement("div");
    label.className = "timeline-label";
    label.textContent = displayName;
    row.appendChild(label);

    const bar = document.createElement("div");
    bar.className = "timeline-bar";

    let totalDuration = 0;
    (version.segments || []).forEach((seg) => {
      const dur = (seg.end || 0) - (seg.start || 0);
      totalDuration += Math.max(dur, 0.1);
    });

    (version.segments || []).forEach((seg, idx) => {
      const dur = (seg.end || 0) - (seg.start || 0);
      const ratio = totalDuration > 0 ? dur / totalDuration : 1 / (version.segments.length || 1);

      const segDiv = document.createElement("div");
      segDiv.className = "segment-block";
      segDiv.style.flexGrow = ratio;
      segDiv.style.flexBasis = ratio * 100 + "%";

      const colors = ["#fbbf24", "#34d399", "#60a5fa", "#f472b6", "#a78bfa", "#f97316", "#22d3ee"];
      segDiv.style.backgroundColor = colors[idx % colors.length];

      segDiv.textContent = seg.name;
      segDiv.dataset.slot = slot;
      segDiv.dataset.start = seg.start ?? 0;
      segDiv.dataset.end = seg.end ?? 0;
      segDiv.dataset.displayName = displayName;

      segDiv.addEventListener("click", onTimelineSegmentClick);
      bar.appendChild(segDiv);
    });

    row.appendChild(bar);
    container.appendChild(row);
  });
}


// -------------------------
// 4. 段落按钮 + 柱状图
// -------------------------

function buildGlobalSegmentKeys(data) {
  const result = [];
  const seen = new Set();
  if (!data || !data.versions) return result;

  data.versions.forEach((version) => {
    const counters = {};
    (version.segments || []).forEach((seg) => {
      const baseName = seg.name || "segment";
      counters[baseName] = (counters[baseName] || 0) + 1;
      const idx = counters[baseName];
      const key = `${baseName}__${idx}`;

      if (!seen.has(key)) {
        seen.add(key);
        result.push({ key, name: baseName, index: idx });
      }
    });
  });

  return result;
}

function findSegmentByKey(version, segKey) {
  if (!version || !version.segments) return null;
  const [baseName, idxStr] = segKey.split("__");
  const targetIndex = parseInt(idxStr, 10);
  if (!baseName || !Number.isFinite(targetIndex)) return null;

  let count = 0;
  for (const seg of version.segments) {
    if (seg.name === baseName) {
      count += 1;
      if (count === targetIndex) return seg;
    }
  }
  return null;
}

function initSegmentUIAndChart(data) {
  if (!data || !data.versions || data.versions.length === 0) return;

  const segmentControls = document.getElementById("segment-controls");
  if (!segmentControls) return;

  segmentControls.innerHTML = "";

  const segments = buildGlobalSegmentKeys(data);
  if (segments.length === 0) return;

  if (currentSegmentKey) {
    const exists = segments.some((s) => s.key === currentSegmentKey);
    if (!exists) currentSegmentKey = null;
  }

  segments.forEach((seg, idx) => {
    const btn = document.createElement("button");
    const label = seg.index > 1 ? `${seg.name} #${seg.index}` : seg.name;

    btn.textContent = label;
    btn.dataset.segmentKey = seg.key;

    if (!currentSegmentKey && idx === 0) {
      btn.classList.add("active");
      currentSegmentKey = seg.key;
    } else if (currentSegmentKey === seg.key) {
      btn.classList.add("active");
    }

    btn.addEventListener("click", () => {
      Array.from(segmentControls.querySelectorAll("button")).forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentSegmentKey = seg.key;
      updateSegmentChart();
    });

    segmentControls.appendChild(btn);
  });

  if (!currentSegmentKey && segments.length > 0) currentSegmentKey = segments[0].key;
  updateSegmentChart();
}

function updateSegmentChart() {
  if (!globalData || !globalData.versions || !currentSegmentKey) return;

  const canvas = document.getElementById("segmentChart");
  if (!canvas) {
    console.warn('Cannot find <canvas id="segmentChart">.');
    return;
  }

  const ctx = canvas.getContext("2d");

  if (segmentChart) {
    segmentChart.destroy();
    segmentChart = null;
  }

  const labels = [];
  const tempoValues = [];
  const energyValues = [];
  const roughnessValues = [];

  globalData.versions.forEach((version) => {
    const displayName = version.name || version.slot || "version";
    labels.push(displayName);

    const seg = findSegmentByKey(version, currentSegmentKey);

    if (seg) {
      const e = typeof seg.energy === "number" ? seg.energy : 0;
      const r = typeof seg.roughness === "number" ? seg.roughness : 0;

      tempoValues.push(seg.tempo ?? null);
      energyValues.push(e * 100);
      roughnessValues.push(r * 100);
    } else {
      tempoValues.push(null);
      energyValues.push(null);
      roughnessValues.push(null);
    }
  });

  segmentChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Tempo (BPM)", data: tempoValues, backgroundColor: "rgba(251, 191, 36, 0.8)" },
        { label: "Energy (0–100)", data: energyValues, backgroundColor: "rgba(34, 197, 94, 0.8)" },
        { label: "Roughness (0–100)", data: roughnessValues, backgroundColor: "rgba(59, 130, 246, 0.8)" },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "white", font: { size: 14 } } }
      },
      scales: {
        x: { ticks: { color: "white", font: { size: 13 } }, grid: { color: "rgba(148, 163, 184, 0.2)" } },
        y: { beginAtZero: true, suggestedMax: 100, ticks: { color: "white", font: { size: 13 } }, grid: { color: "rgba(148, 163, 184, 0.2)" } },
      },
    },
  });
}


// ========================== 5. 全局特征雷达图 ==========================

function summarizeSegmentsForRadar(segments) {
  if (!segments || segments.length === 0) {
    return { energy: 0.5, valence: 0.5, brightness: 2400, roughness: 0.3 };
  }

  let energy = 0, valence = 0, brightness = 0, roughness = 0;
  for (const s of segments) {
    energy += s.energy ?? 0.5;
    valence += s.valence ?? 0.5;
    brightness += s.brightness ?? 2400;
    roughness += s.roughness ?? 0.3;
  }

  const n = segments.length;
  return { energy: energy / n, valence: valence / n, brightness: brightness / n, roughness: roughness / n };
}

function buildGlobalRadar(data) {
  if (!data || !data.versions) return;

  const canvas = document.getElementById("globalRadar");
  if (!canvas) {
    console.warn('Cannot find <canvas id="globalRadar">.');
    return;
  }

  const ctx = canvas.getContext("2d");
  const labels = ["energy", "valence", "brightness (norm.)", "roughness"];

  const summaries = data.versions.map((v) => {
    const stats = summarizeSegmentsForRadar(v.segments || []);
    return { name: v.name || v.slot || "version", ...stats };
  });

  const bValues = summaries.map((v) => (typeof v.brightness === "number" ? v.brightness : 2400));
  let bMin = Math.min(...bValues);
  let bMax = Math.max(...bValues);
  if (!isFinite(bMin) || !isFinite(bMax) || bMax - bMin < 1e-6) {
    bMin = 0; bMax = 1;
  }

  const paletteFill = [
    "rgba(110, 230, 200, 0.4)",
    "rgba(255, 190, 120, 0.4)",
    "rgba(150, 170, 255, 0.4)",
    "rgba(255, 160, 200, 0.4)",
  ];
  const paletteBorder = [
    "rgba(110, 230, 200, 1.0)",
    "rgba(255, 190, 120, 1.0)",
    "rgba(150, 170, 255, 1.0)",
    "rgba(255, 160, 200, 1.0)",
  ];

  const datasets = summaries.map((v, idx) => {
    const fillC = paletteFill[idx % paletteFill.length];
    const borderC = paletteBorder[idx % paletteBorder.length];

    let bNorm = ((v.brightness ?? 2400) - bMin) / (bMax - bMin || 1);
    bNorm = Math.max(0, Math.min(1, bNorm));

    return {
      label: v.name,
      data: [v.energy ?? 0.5, v.valence ?? 0.5, bNorm, v.roughness ?? 0.3],
      backgroundColor: fillC,
      borderColor: borderC,
      borderWidth: 2,
      pointRadius: 3,
      pointBackgroundColor: borderC,
    };
  });

  if (globalRadarChart) {
    globalRadarChart.destroy();
    globalRadarChart = null;
  }

  globalRadarChart = new Chart(ctx, {
    type: "radar",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          suggestedMin: 0,
          suggestedMax: 1,
          angleLines: { color: "rgba(148, 163, 184, 0.4)" },
          grid: { color: "rgba(148, 163, 184, 0.2)" },
          ticks: { display: false },
          pointLabels: { color: "#e5e7eb", font: { size: 12 } },
        },
      },
      plugins: { legend: { labels: { color: "#e5e7eb" } } },
    },
  });
}


// ========================== 6. AI Emotional Commentary (GPT) ==========================

function initAICommentary() {
  const btn = document.getElementById("btn-ai-analyse");
  const status = document.getElementById("ai-status");
  const resultsDiv = document.getElementById("ai-results");

  if (!btn || !status || !resultsDiv) {
    console.log("[AI] UI elements not found, skip AI init.");
    return;
  }

  btn.addEventListener("click", async () => {
    if (!globalData || !globalData.versions || globalData.versions.length === 0) {
      alert("Please run the audio analysis first.");
      return;
    }

    btn.disabled = true;
    status.textContent = "Asking AI to compare all versions...";
    resultsDiv.innerHTML = "";

    try {
      // ✅ 关键：slot 是稳定ID，name 是用户命名（展示用）
      const payload = {
        versions: globalData.versions.map((v) => ({
          slot: v.slot || "",
          name: v.name || v.slot || "version",
          segments: (v.segments || []).map((s) => ({
            name: s.name,
            start: s.start,
            end: s.end,
            tempo: s.tempo,
            energy: s.energy,
            valence: s.valence,
            roughness: s.roughness,
            brightness: s.brightness,
            loudness: s.loudness,
            key: s.key,
          })),
        })),
      };

      const resp = await fetch("/api/ai_compare_all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        throw new Error("HTTP " + resp.status);
      }

      const aiAll = await resp.json();
      const per = aiAll.per_version || {};
      const overall = aiAll.overall_compare || "";

      // ✅ 回写：永远用 slot 对齐
      globalData.versions.forEach((v) => {
        const slot = v.slot || "";
        if (slot && per[slot]) {
          v.aiColor = per[slot].rgb;
          v.aiComment = per[slot].commentary;
        }
      });

      // overall
      if (overall && overall.trim()) {
        const overallBlock = document.createElement("div");
        overallBlock.style.marginBottom = "14px";
        overallBlock.style.padding = "10px 12px";
        overallBlock.style.border = "1px solid rgba(129,140,248,0.35)";
        overallBlock.style.borderRadius = "10px";
        overallBlock.innerHTML = `
          <strong style="color:#a5b4fc;">Overall comparison</strong>
          <p style="margin:6px 0 0; font-size:0.92rem; opacity:0.92; line-height:1.35;">
            ${overall}
          </p>
        `;
        resultsDiv.appendChild(overallBlock);
      }

      // per version blocks: 标题显示用户命名，但内容来自 per[slot]
      globalData.versions.forEach((v) => {
        const slot = v.slot || "";
        const displayName = v.name || slot || "version";
        const info = slot ? per[slot] : null;
        if (!info) return;

        const rgb = info.rgb;
        const colorCss =
          rgb && Array.isArray(rgb) && rgb.length === 3
            ? `rgb(${rgb.join(",")})`
            : "#cbd5f5";

        const block = document.createElement("div");
        block.style.marginBottom = "12px";
        block.style.borderLeft = `4px solid ${colorCss}`;
        block.style.paddingLeft = "10px";
        block.innerHTML = `
          <strong style="color:${colorCss}; font-size:1.02rem;">${displayName}</strong>
          <p style="margin:6px 0 0; font-size:0.92rem; opacity:0.92; line-height:1.35;">
            ${info.commentary || ""}
          </p>
        `;
        resultsDiv.appendChild(block);
      });

      status.textContent = "Done.";

      if (window.setVersionsForVis) {
        window.setVersionsForVis(globalData);
      }
    } catch (e) {
      console.error(e);
      status.textContent = "Error while asking AI.";
    } finally {
      btn.disabled = false;
    }
  });
}
