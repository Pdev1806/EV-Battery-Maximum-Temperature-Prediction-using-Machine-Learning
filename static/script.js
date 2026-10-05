/**
 * script.js
 * ---------
 * Client-side logic for EV Battery T_max Predictor (KJS-CES-02).
 * Handles slider sync, REST API calls (/predict, /whatif, /api/static-data),
 * Chart.js renders, and responsive tab management.
 */

let whatIfChartInstance = null;
let importanceChartInstance = null;

// Presets mapping
const PRESETS = {
  baseline: { ha: 30.0, phi: 0.030, uin: 0.175 },
  high_flow: { ha: 10.0, phi: 0.050, uin: 0.280 },
  low_flow: { ha: 55.0, phi: 0.010, uin: 0.080 },
};

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // 1. Fetch benchmark data from Flask backend
  loadStaticMetadata();

  // 2. Perform initial prediction and build charts
  submitPrediction();

  // 3. Measure and report initial page load performance
  window.addEventListener("load", () => {
    const timing = window.performance.timing;
    if (timing) {
      const pageLoadTime = timing.loadEventEnd - timing.navigationStart;
      if (pageLoadTime > 0) {
        console.log(`[Performance] Page Load Completed in: ${pageLoadTime} ms`);
      }
    }
  });
});

// -----------------------------------------------------------------------------
// Input Synchronization & Presets
// -----------------------------------------------------------------------------
function syncFromSlider(name) {
  const slider = document.getElementById(`${name}Slider`);
  const num = document.getElementById(`${name}Num`);
  if (slider && num) {
    num.value = slider.value;
  }
}

function syncFromNumber(name) {
  const slider = document.getElementById(`${name}Slider`);
  const num = document.getElementById(`${name}Num`);
  if (slider && num) {
    slider.value = num.value;
  }
}

function applyPreset(presetKey) {
  const p = PRESETS[presetKey];
  if (!p) return;

  document.getElementById("haSlider").value = p.ha;
  document.getElementById("haNum").value = p.ha;

  document.getElementById("phiSlider").value = p.phi;
  document.getElementById("phiNum").value = p.phi;

  document.getElementById("uinSlider").value = p.uin;
  document.getElementById("uinNum").value = p.uin;

  submitPrediction();
}

// -----------------------------------------------------------------------------
// API Calls: /predict
// -----------------------------------------------------------------------------
async function submitPrediction() {
  const predictBtn = document.getElementById("predictBtn");
  const spinner = predictBtn ? predictBtn.querySelector(".spinner") : null;
  const btnText = predictBtn ? predictBtn.querySelector(".btn-text") : null;

  const ha = parseFloat(document.getElementById("haNum").value);
  const phi = parseFloat(document.getElementById("phiNum").value);
  const u_in = parseFloat(document.getElementById("uinNum").value);

  // Set loading state
  if (predictBtn) predictBtn.disabled = true;
  if (spinner) spinner.classList.remove("hidden");
  if (btnText) btnText.textContent = "Computing...";

  dismissError();
  const startTime = performance.now();

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ha, phi, u_in }),
    });

    const elapsedMs = Math.round(performance.now() - startTime);
    const badge = document.getElementById("callTimeBadge");
    if (badge) badge.textContent = `API: ${elapsedMs} ms`;

    const data = await res.json();

    if (!res.ok || data.status === "error") {
      showError(data.error || "Prediction request failed.");
      return;
    }

    // Render result card
    renderPredictionResult(data);

    // Refresh what-if sensitivity curve with new Ha and phi
    loadWhatIfCurve(ha, phi);

  } catch (err) {
    showError("Network or server connection failed. Ensure Flask app is running.");
  } finally {
    if (predictBtn) predictBtn.disabled = false;
    if (spinner) spinner.classList.add("hidden");
    if (btnText) btnText.textContent = "🚀 Predict T_max";
  }
}

function renderPredictionResult(data) {
  const tmaxEl = document.getElementById("tmaxVal");
  const riskBadge = document.getElementById("riskBadge");
  const riskLabel = document.getElementById("riskLabel");
  const riskIcon = document.getElementById("riskIcon");
  const riskDesc = document.getElementById("riskExplanation");
  const resultCard = document.getElementById("resultCard");

  if (tmaxEl) tmaxEl.textContent = data.t_max.toFixed(2);
  if (riskLabel) riskLabel.textContent = data.condition;
  if (riskIcon) riskIcon.textContent = data.icon || "🌡️";
  if (riskDesc) riskDesc.textContent = data.explanation || "";

  // Reset & apply risk classes
  if (riskBadge) {
    riskBadge.className = `risk-badge risk-${data.condition.toLowerCase()}`;
  }

  if (resultCard && data.border) {
    resultCard.style.borderLeftColor = data.border;
  }

  // Update quick stat cell values
  const statHa = document.getElementById("statHa");
  const statPhi = document.getElementById("statPhi");
  const statUin = document.getElementById("statUin");
  const statDelta = document.getElementById("statDelta");

  if (statHa) statHa.textContent = data.inputs.Ha.toFixed(1);
  if (statPhi) statPhi.textContent = data.inputs.phi.toFixed(3);
  if (statUin) statUin.textContent = `${data.inputs.u_in.toFixed(3)} m/s`;

  if (statDelta) {
    const delta = data.delta_mean;
    const sign = delta >= 0 ? "+" : "";
    statDelta.textContent = `${sign}${delta.toFixed(2)} °C`;
    statDelta.style.color = delta > 5 ? "#ef4444" : delta < -3 ? "#10b981" : "#f59e0b";
  }
}

// -----------------------------------------------------------------------------
// API Calls: /whatif
// -----------------------------------------------------------------------------
async function loadWhatIfCurve(haArg, phiArg) {
  const ha = haArg !== undefined ? haArg : parseFloat(document.getElementById("haNum").value);
  const phi = phiArg !== undefined ? phiArg : parseFloat(document.getElementById("phiNum").value);

  try {
    const res = await fetch("/whatif", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ha, phi }),
    });

    const data = await res.json();
    if (!res.ok || data.status === "error") {
      return;
    }

    renderWhatIfChart(data.curve, ha, phi);
  } catch (err) {
    console.error("Error loading what-if curve:", err);
  }
}

function renderWhatIfChart(curveData, ha, phi) {
  const ctx = document.getElementById("whatifChart");
  if (!ctx) return;

  const labels = curveData.map((d) => d.u_in.toFixed(3));
  const temps = curveData.map((d) => d.t_max);

  if (whatIfChartInstance) {
    whatIfChartInstance.destroy();
  }

  whatIfChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: `T_max (°C) at Ha=${ha.toFixed(1)}, phi=${phi.toFixed(3)}`,
          data: temps,
          borderColor: "#881337",
          backgroundColor: "rgba(136, 19, 55, 0.08)",
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointBackgroundColor: "#881337",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: { font: { family: "Inter", weight: 600 } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Predicted T_max: ${ctx.parsed.y.toFixed(2)} °C`,
          },
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Inlet Flow Velocity (u_in) [m/s]",
            font: { family: "Inter", weight: 600 },
          },
          grid: { color: "#f1f5f9" },
        },
        y: {
          title: {
            display: true,
            text: "Predicted Maximum Temp (T_max) [°C]",
            font: { family: "Inter", weight: 600 },
          },
          grid: { color: "#f1f5f9" },
        },
      },
    },
  });
}

// -----------------------------------------------------------------------------
// API Calls: /api/static-data
// -----------------------------------------------------------------------------
async function loadStaticMetadata() {
  try {
    const res = await fetch("/api/static-data");
    const json = await res.json();
    if (!res.ok || json.status !== "success") return;

    const data = json.data;

    // 1. Populate Benchmark table
    renderBenchmarkTable(data.model_comparison);

    // 2. Populate Sweep table
    renderSweepTable(data.rf_config_sweep);

    // 3. Populate Sample predictions table
    renderSampleTable(data.sample_predictions);

    // 4. Render Feature Importance Bar Chart
    renderFeatureImportanceChart(data.feature_importance);

  } catch (err) {
    console.warn("Could not load static benchmark metadata:", err);
  }
}

function renderBenchmarkTable(rows) {
  const tbody = document.getElementById("benchmarkTableBody");
  if (!tbody || !rows || rows.length === 0) return;

  tbody.innerHTML = rows
    .map(
      (r) => `
      <tr>
        <td><strong>${r.Model || r.model}</strong></td>
        <td>${parseFloat(r.MAE).toFixed(4)}</td>
        <td>${parseFloat(r.MSE).toFixed(4)}</td>
        <td>${parseFloat(r.RMSE).toFixed(4)}</td>
        <td><strong style="color: #881337;">${parseFloat(r.R2).toFixed(4)}</strong></td>
      </tr>
    `
    )
    .join("");
}

function renderSweepTable(rows) {
  const tbody = document.getElementById("sweepTableBody");
  if (!tbody || !rows || rows.length === 0) return;

  tbody.innerHTML = rows
    .map(
      (r) => `
      <tr>
        <td><strong>${r.n_estimators} trees</strong></td>
        <td>${parseFloat(r.MAE).toFixed(4)}</td>
        <td>${parseFloat(r.MSE).toFixed(4)}</td>
        <td>${parseFloat(r.RMSE).toFixed(4)}</td>
        <td>${parseFloat(r.R2).toFixed(4)}</td>
      </tr>
    `
    )
    .join("");
}

function renderSampleTable(rows) {
  const tbody = document.getElementById("sampleTableBody");
  if (!tbody || !rows || rows.length === 0) return;

  tbody.innerHTML = rows
    .map(
      (r) => `
      <tr>
        <td>${parseFloat(r.Actual_T_max).toFixed(3)} °C</td>
        <td>${parseFloat(r.Predicted_T_max).toFixed(3)} °C</td>
        <td><span style="font-weight: 600; color: ${parseFloat(r.Absolute_Error) > 1.5 ? '#ef4444' : '#10b981'};">
          ${parseFloat(r.Absolute_Error).toFixed(3)} °C
        </span></td>
      </tr>
    `
    )
    .join("");
}

function renderFeatureImportanceChart(items) {
  const ctx = document.getElementById("importanceChart");
  if (!ctx || !items || items.length === 0) return;

  const labels = items.map((i) => i.feature);
  const vals = items.map((i) => (i.importance * 100).toFixed(2));

  if (importanceChartInstance) {
    importanceChartInstance.destroy();
  }

  importanceChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Importance (%)",
          data: vals,
          backgroundColor: ["#881337", "#b91c1c", "#fb7185"],
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Relative Impact: ${ctx.parsed.x}%`,
          },
        },
      },
      scales: {
        x: {
          max: 100,
          title: { display: true, text: "Gini Impurity Importance (%)" },
          grid: { color: "#f1f5f9" },
        },
      },
    },
  });
}

// -----------------------------------------------------------------------------
// UI Utilities: Tabs, Plots, Error Banner
// -----------------------------------------------------------------------------
function switchTab(tabId) {
  // Update button active states
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.classList.remove("active");
    if (btn.getAttribute("onclick").includes(tabId)) {
      btn.classList.add("active");
    }
  });

  // Update content active states
  const contents = document.querySelectorAll(".tab-content");
  contents.forEach((c) => c.classList.remove("active"));

  const target = document.getElementById(tabId);
  if (target) {
    target.classList.add("active");
  }

  // Trigger chart resize if navigating to chart tabs
  if (tabId === "tab-whatif" && whatIfChartInstance) {
    setTimeout(() => whatIfChartInstance.resize(), 50);
  }
  if (tabId === "tab-importance" && importanceChartInstance) {
    setTimeout(() => importanceChartInstance.resize(), 50);
  }
}

function changePlot(filename) {
  const img = document.getElementById("diagnosticImg");
  const caption = document.getElementById("imgCaption");
  if (img) {
    img.src = `/static/${filename}`;
  }
  if (caption) {
    caption.textContent = `results/${filename}`;
  }
}

function showError(msg) {
  const banner = document.getElementById("errorBanner");
  const text = document.getElementById("errorMessage");
  if (banner && text) {
    text.textContent = msg;
    banner.classList.remove("hidden");
  }
}

function dismissError() {
  const banner = document.getElementById("errorBanner");
  if (banner) {
    banner.classList.add("hidden");
  }
}
