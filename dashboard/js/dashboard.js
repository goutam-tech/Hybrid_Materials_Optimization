/**
 * dashboard.js
 * Shared API client + Chart.js helpers for the Hybrid Materials
 * Optimization System dashboard pages.
 */

const API_BASE = "/api";

const PALETTE = {
  classical: "#38bdf8",
  quantum: "#a855f7",
  good: "#34d399",
  warn: "#fbbf24",
  bad: "#f87171",
  grid: "rgba(255,255,255,0.06)",
  text: "#8c98b8",
};

// Chart.defaults.color = PALETTE.text;
// Chart.defaults.borderColor = PALETTE.grid;
// Chart.defaults.font.family = "'Inter', 'Segoe UI', system-ui, sans-serif";

if (window.Chart) {
  Chart.defaults.color = PALETTE.text;
  Chart.defaults.borderColor = PALETTE.grid;
  Chart.defaults.font.family = "'Inter', 'Segoe UI', system-ui, sans-serif";
} else {
  console.error("Chart.js failed to load");
}

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  const payload = await res.json();
  if (payload.status !== "success") {
    throw new Error(payload.message || "Request failed");
  }
  return payload.data;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

function formatSeconds(value) {
  if (value === null || value === undefined) return "—";
  return `${Number(value).toFixed(4)}s`;
}

function barChart(ctx, labels, datasets, options = {}) {
  if (!ctx) {
    console.error("Chart canvas not found");
    return;
  }

  return new Chart(ctx, {
    type: "bar",
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: PALETTE.text } } },
      scales: {
        x: { grid: { color: PALETTE.grid }, ticks: { color: PALETTE.text } },
        y: { grid: { color: PALETTE.grid }, ticks: { color: PALETTE.text } },
      },
      ...options,
    },
  });
}

function lineChart(ctx, labels, datasets, options = {}) {
  return new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: PALETTE.text } } },
      scales: {
        x: { grid: { color: PALETTE.grid }, ticks: { color: PALETTE.text } },
        y: { grid: { color: PALETTE.grid }, ticks: { color: PALETTE.text } },
      },
      ...options,
    },
  });
}

function radarChart(ctx, labels, datasets) {
  return new Chart(ctx, {
    type: "radar",
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: PALETTE.text } } },
      scales: {
        r: {
          grid: { color: PALETTE.grid },
          angleLines: { color: PALETTE.grid },
          pointLabels: { color: PALETTE.text },
          ticks: { display: false },
        },
      },
    },
  });
}

function setActiveNav() {
  const path = window.location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".sidebar .nav-link").forEach((link) => {
    const href = link.getAttribute("href");
    if (href === path) link.classList.add("active");
  });
}

document.addEventListener("DOMContentLoaded", setActiveNav);
