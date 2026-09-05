/**
 * OmniSentinel Dashboard JS
 * SIH 2026 | AI Network Attack Forecasting
 */

const API_BASE = "http://localhost:8000";

// DOM Elements
const elStatusPill = document.getElementById("status-pill");
const elStatusText = document.getElementById("status-text");
const elClock = document.getElementById("clock");
const elRiskScore = document.getElementById("risk-score-val");
const elRiskLevel = document.getElementById("risk-level-val");
const elMaxProb = document.getElementById("max-prob");
const elMeanProb = document.getElementById("mean-prob");
const elSeqCount = document.getElementById("seq-count");
const elRiskCard = document.getElementById("risk-card");
const elKillChain = document.getElementById("kill-chain");
const elMitreTactic = document.getElementById("mitre-tactic");
const elMitreTechnique = document.getElementById("mitre-technique");
const elMitreSeverity = document.getElementById("mitre-severity");
const elMitreNext = document.getElementById("mitre-next");
const elScenarioBadge = document.getElementById("scenario-badge");
const elAlertBadge = document.getElementById("alert-badge");
const elAlertTbody = document.getElementById("alert-tbody");
const elSaliencyList = document.getElementById("saliency-list");
const elMAuc = document.getElementById("m-auc");
const elMLatency = document.getElementById("m-latency");
const elMAlerts = document.getElementById("m-alerts");
const elGaugeArc = document.getElementById("gauge-arc");
const elGaugeNeedle = document.getElementById("gauge-needle");

// Charts
let timelineChart = null;
let aucChart = null;

// State
let alertsCount = 0;
let totalSequences = 0;

// Clock
setInterval(() => {
  const d = new Date();
  elClock.textContent = d.toLocaleTimeString('en-US', { hour12: false });
}, 1000);

// Init Charts
function initCharts() {
  Chart.defaults.color = "#8fa8c8";
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.plugins.tooltip.backgroundColor = "rgba(14,22,36,0.95)";
  Chart.defaults.plugins.tooltip.borderColor = "rgba(52,152,219,0.3)";
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;

  // Timeline Chart
  const ctxTimeline = document.getElementById("timeline-chart").getContext("2d");
  timelineChart = new Chart(ctxTimeline, {
    type: "line",
    data: {
      labels: ["k=1", "k=2", "k=3", "k=4", "k=5", "k=6", "k=7", "k=8"],
      datasets: [{
        label: "P(Attack)",
        data: [0, 0, 0, 0, 0, 0, 0, 0],
        borderColor: "#3498db",
        backgroundColor: "rgba(52,152,219,0.15)",
        borderWidth: 2.5,
        pointBackgroundColor: "#fff",
        pointBorderColor: "#3498db",
        pointBorderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 1.05, grid: { color: "rgba(255,255,255,0.05)" } },
        x: { grid: { display: false } }
      },
      plugins: {
        legend: { display: false }
      },
      animation: { duration: 600, easing: 'easeOutQuart' }
    }
  });

  // AUC Chart (Static data for demo based on phase10 results)
  const ctxAuc = document.getElementById("auc-chart").getContext("2d");
  aucChart = new Chart(ctxAuc, {
    type: "bar",
    data: {
      labels: ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"],
      datasets: [{
        label: "AUC",
        data: [0.8153, 0.8071, 0.7994, 0.7900, 0.7840, 0.7761, 0.7715, 0.7725],
        backgroundColor: function(context) {
            const chart = context.chart;
            const {ctx, chartArea} = chart;
            if (!chartArea) { return null; }
            const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
            gradient.addColorStop(0, "rgba(52,152,219,0.2)");
            gradient.addColorStop(1, "rgba(52,152,219,0.8)");
            return gradient;
        },
        borderRadius: 4,
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0.70, max: 0.85, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { stepSize: 0.05 } },
        x: { grid: { display: false } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// Gauge Math
function setGauge(score) {
  // SVG arc dasharray logic for gauge filling
  const arcLen = 251.3; // Approx length of arc
  const p = Math.min(Math.max(score, 0), 100) / 100;
  const offset = arcLen - (p * arcLen);
  elGaugeArc.style.strokeDashoffset = offset;
  
  // Rotate needle (180deg sweep, from 0 to 180)
  // Our arc is from x=20 to x=180 on a 200x120 canvas. The center is ~100,100
  // Start angle: -90, end: +90
  const deg = (p * 180) - 90;
  elGaugeNeedle.setAttribute("transform", `rotate(${deg} 100 100)`);
}

// Update Dashboard
function updateDashboard(data) {
  // Top level stats
  totalSequences++;
  elSeqCount.textContent = totalSequences.toLocaleString();
  elRiskScore.textContent = data.risk_score.toFixed(1);
  elRiskLevel.textContent = data.risk_level;
  
  // Colors based on risk
  let color = "#27ae60";
  elRiskLevel.className = "gauge-level level-" + data.risk_level;
  elRiskCard.classList.remove("critical");
  
  if(data.risk_level === "CRITICAL") { color = "#e74c3c"; elRiskCard.classList.add("critical"); }
  else if(data.risk_level === "HIGH") { color = "#e67e22"; }
  else if(data.risk_level === "MEDIUM") { color = "#f1c40f"; }
  else if(data.risk_level === "LOW") { color = "#3498db"; }
  
  setGauge(data.risk_score);
  
  elMaxProb.textContent = data.max_attack_prob.toFixed(3);
  elMeanProb.textContent = (data.attack_probability_timeline.reduce((a,b)=>a+b,0)/8).toFixed(3);
  elScenarioBadge.textContent = data.scenario;
  
  // Chart update
  timelineChart.data.datasets[0].data = data.attack_probability_timeline;
  
  // Dynamic color for chart based on risk
  let chartColor = color;
  if (data.risk_level === "SAFE") chartColor = "#3498db";
  
  timelineChart.data.datasets[0].borderColor = chartColor;
  timelineChart.data.datasets[0].pointBorderColor = chartColor;
  timelineChart.data.datasets[0].backgroundColor = chartColor + "30"; // 30 is hex for approx 20% opacity
  timelineChart.update();
  
  // Mitre
  const t = data.predicted_tactic_k1 || "—";
  elMitreTactic.textContent = t;
  // Mock technique based on tactic for demo
  const techniques = {
    "Impact": "T1498 (DDoS)",
    "Discovery": "T1046 (Port Scan)",
    "Credential Access": "T1110 (Brute Force)",
    "Command & Control": "T1071 (App Layer)",
    "Initial Access": "T1190 (Exploit Public App)"
  };
  elMitreTechnique.textContent = techniques[t] || "—";
  
  let sevStr = "—";
  if (data.risk_level === "CRITICAL") sevStr = "5 - High Impact";
  else if (data.risk_level === "HIGH") sevStr = "4 - Medium-High";
  else if (data.risk_level === "MEDIUM") sevStr = "3 - Medium";
  
  elMitreSeverity.textContent = t !== "—" ? sevStr : "0 - None";
  elMitreNext.textContent = (data.attack_probability_timeline[1] > 0.5) ? "Progression Likely" : "No Progression";

  // Build Kill Chain UI (Mock)
  buildKillChain(t);

  // Saliency updates
  updateSaliency(data.scenario);

  // Alerts
  if (data.alert) {
    alertsCount++;
    elAlertBadge.textContent = `${alertsCount} alerts`;
    elMAlerts.textContent = alertsCount;
    
    // Remove empty row if present
    const emptyRow = elAlertTbody.querySelector('.empty-row');
    if (emptyRow) emptyRow.remove();
    
    const tr = document.createElement("tr");
    tr.className = "new-row";
    tr.innerHTML = `
      <td>${data.timestamp}</td>
      <td>${data.scenario}</td>
      <td style="font-family: var(--font-mono)">${data.risk_score.toFixed(1)}</td>
      <td><span class="level-badge ${data.risk_level}">${data.risk_level}</span></td>
      <td>${t}</td>
      <td>${data.max_attack_prob.toFixed(2)}</td>
    `;
    elAlertTbody.insertBefore(tr, elAlertTbody.firstChild);
    
    // Keep max 15 alerts
    while(elAlertTbody.children.length > 15) {
      elAlertTbody.removeChild(elAlertTbody.lastChild);
    }
  }
}

const ALL_TACTICS = ["Reconnaissance", "Initial Access", "Execution", "Credential Access", "Discovery", "Lateral Movement", "Command & Control", "Impact"];

function buildKillChain(activeTactic) {
  elKillChain.innerHTML = "";
  
  // Map our predicted tactic to a broader stage if needed, for simplicity let's just highlight the exact match if it exists in ALL_TACTICS, or Impact if not found
  let activeIndex = ALL_TACTICS.indexOf(activeTactic);
  if (activeTactic && activeIndex === -1 && activeTactic !== "—") activeIndex = ALL_TACTICS.length - 1; // Default to impact for demo if tactic is weird
  
  ALL_TACTICS.forEach((tac, idx) => {
    const isPast = idx < activeIndex;
    const isActive = tac === activeTactic;
    
    const div = document.createElement("div");
    div.className = "kc-stage" + (isActive ? " active" : "");
    if (isPast) div.style.opacity = "0.7";
    
    div.innerHTML = `
      <div class="kc-dot" ${isPast ? 'style="background:var(--accent-green)"' : ''}></div>
      <div class="kc-num">0${idx+1}</div>
      <div class="kc-name">${tac}</div>
    `;
    elKillChain.appendChild(div);
  });
}

function updateSaliency(scenario) {
  // Mock saliency features that shift based on scenario to make dashboard look alive
  const featurePool = [
    { n: "window_size_ratio", eng: true, v: Math.random() * 0.05 + 0.01 },
    { n: "log1p_pkt_len_mean", eng: true, v: Math.random() * 0.04 + 0.01 },
    { n: "PSH Flag Count", eng: false, v: Math.random() * 0.03 + 0.01 },
    { n: "log1p_flow_duration", eng: true, v: Math.random() * 0.02 + 0.005 },
    { n: "Destination Port", eng: false, v: Math.random() * 0.01 + 0.001 },
    { n: "ACK Flag Count", eng: false, v: Math.random() * 0.02 + 0.005 },
    { n: "byte_fwd_bwd_ratio", eng: true, v: Math.random() * 0.03 + 0.01 },
  ];
  
  featurePool.sort((a,b) => b.v - a.v);
  
  elSaliencyList.innerHTML = "";
  for(let i=0; i<5; i++) {
    const f = featurePool[i];
    const pct = (f.v / 0.06) * 100; // rough scale
    
    elSaliencyList.innerHTML += `
      <div class="sal-row">
        <div class="sal-header">
          <span class="sal-name" title="${f.n}">${f.n}</span>
          <span class="sal-tag ${f.eng ? 'eng' : 'orig'}">${f.eng ? 'ENG' : 'RAW'}</span>
          <span class="sal-val">${f.v.toFixed(4)}</span>
        </div>
        <div class="sal-bar-bg">
          <div class="sal-bar-fill ${f.eng ? 'engineered' : ''}" style="width: ${Math.min(pct, 100)}%"></div>
        </div>
      </div>
    `;
  }
}

// ── PCAP File Upload Logic ──────────────────────────────────────────────────
const elUploadOverlay = document.getElementById("upload-overlay");
const elDropzone = document.getElementById("upload-dropzone");
const elFileInput = document.getElementById("pcap-file-input");
const elUploadStatus = document.getElementById("upload-status");
const elUploadError = document.getElementById("upload-error");

function setupUpload() {
  elDropzone.addEventListener("click", () => elFileInput.click());
  
  elDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    elDropzone.classList.add("dragover");
  });
  
  elDropzone.addEventListener("dragleave", () => {
    elDropzone.classList.remove("dragover");
  });
  
  elDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    elDropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  elFileInput.addEventListener("change", (e) => {
    if (e.target.files.length) {
      handleFileUpload(e.target.files[0]);
    }
  });
}

async function handleFileUpload(file) {
  elUploadError.classList.remove("active");
  elUploadStatus.classList.add("active");
  
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(API_BASE + "/api/analyze-pcap", {
      method: "POST",
      body: formData
    });
    
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Analysis failed");
    }
    
    const data = await res.json();
    
    // Map PCAP response format to dashboard format
    const mappedData = {
      risk_score: data.current_state.risk_score,
      risk_level: data.current_state.risk_level,
      max_attack_prob: Math.max(...data.forecast.map(f => f.attack_prob)),
      attack_probability_timeline: data.forecast.map(f => f.attack_prob),
      scenario: "PCAP Analysis",
      predicted_tactic_k1: data.mitre_progression[0].tactic,
      alert: data.current_state.risk_level === "HIGH" || data.current_state.risk_level === "CRITICAL",
      timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
    };

    updateDashboard(mappedData);
    
    // Update metric cards
    elMAuc.textContent = "—";
    elMLatency.textContent = data.processing_time_ms + " ms";
    
    // Update Saliency directly from backend instead of mock
    if (data.top_saliency_features) {
      elSaliencyList.innerHTML = "";
      data.top_saliency_features.slice(0, 5).forEach(f => {
        const pct = (f.saliency_score / data.top_saliency_features[0].saliency_score) * 100;
        elSaliencyList.innerHTML += `
          <div class="sal-row">
            <div class="sal-header">
              <span class="sal-name" title="${f.feature}">${f.feature}</span>
              <span class="sal-tag orig">RAW</span>
              <span class="sal-val">${f.saliency_score.toFixed(4)}</span>
            </div>
            <div class="sal-bar-bg">
              <div class="sal-bar-fill" style="width: ${Math.min(pct, 100)}%"></div>
            </div>
          </div>
        `;
      });
    }
    
    // Hide overlay
    elUploadOverlay.classList.add("hidden");
    
    elStatusPill.className = "status-pill";
    elStatusPill.innerHTML = '<span class="status-dot"></span><span id="status-text">PCAP Analysis Complete</span>';
    
  } catch (err) {
    elUploadError.textContent = err.message;
    elUploadError.classList.add("active");
  } finally {
    elUploadStatus.classList.remove("active");
  }
}

// Init
window.onload = () => {
  initCharts();
  setupUpload();
};
