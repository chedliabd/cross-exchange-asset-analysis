/* -----------------------------------------
   Markets, Decoded — Demo wiring (placeholders)
   ----------------------------------------- */

   (function () {
    // ---------- tiny helpers ----------
    const $ = (sel) => document.querySelector(sel);
  
    function setLoading(id, isLoading) {
      const el = document.getElementById(id);
      if (!el) return;
      el.classList.toggle("is-loading", !!isLoading);
    }
  
    function clear(el) {
      if (!el) return;
      el.innerHTML = "";
    }
  
    function svgWrap({ w = 900, h = 260, label = "" } = {}) {
      return `
        <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" preserveAspectRatio="none" role="img" aria-label="${label}">
          <rect x="0" y="0" width="${w}" height="${h}" fill="rgba(255,255,255,0.02)" />
        </svg>
      `;
    }
  
    function drawTitle(el, title, subtitle) {
      const w = 900, h = 260;
      el.innerHTML = `
        <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" preserveAspectRatio="none">
          <rect x="0" y="0" width="${w}" height="${h}" fill="rgba(255,255,255,0.02)"/>
          <text x="24" y="54" fill="rgba(232,238,255,0.92)" font-size="28" font-weight="700">${escapeXml(title)}</text>
          <text x="24" y="88" fill="rgba(169,179,216,0.95)" font-size="16">${escapeXml(subtitle || "")}</text>
          <text x="24" y="${h-26}" fill="rgba(111,122,168,0.95)" font-size="14">Demo placeholder (replace with real chart)</text>
        </svg>
      `;
    }
  
    function escapeXml(s) {
      return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  
    // ---------- Demo charts ----------
    function demoHeatmap(el, title = "Correlation heatmap") {
      const w = 900, h = 260;
      const rows = 8, cols = 14;
      const pad = 18;
      const gridW = w - pad * 2, gridH = h - 90;
      const cellW = gridW / cols, cellH = gridH / rows;
  
      let rects = "";
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const v = Math.sin((r + 1) * 0.7 + c * 0.25) * 0.5 + 0.5; // 0..1
          const alpha = 0.08 + v * 0.55;
          rects += `<rect x="${pad + c * cellW}" y="${70 + r * cellH}" width="${cellW - 2}" height="${cellH - 2}"
                     fill="rgba(124,92,255,${alpha.toFixed(3)})" />`;
        }
      }
  
      el.innerHTML = `
        <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" preserveAspectRatio="none">
          <rect x="0" y="0" width="${w}" height="${h}" fill="rgba(255,255,255,0.02)"/>
          <text x="24" y="48" fill="rgba(232,238,255,0.92)" font-size="22" font-weight="700">${escapeXml(title)}</text>
          ${rects}
        </svg>
      `;
    }
  
    function demoLine(el, title = "Rolling correlation") {
      const w = 900, h = 260;
      const left = 50, right = 30, top = 60, bottom = 36;
      const W = w - left - right;
      const H = h - top - bottom;
  
      const n = 80;
      const pts = [];
      for (let i = 0; i < n; i++) {
        const t = i / (n - 1);
        const v = 0.5 + 0.18 * Math.sin(2 * Math.PI * (t * 1.15)) + 0.08 * Math.sin(2 * Math.PI * (t * 3.4));
        const x = left + t * W;
        const y = top + (1 - v) * H;
        pts.push([x, y]);
      }
  
      const path = pts.map((p, i) => (i === 0 ? `M ${p[0].toFixed(2)} ${p[1].toFixed(2)}` : `L ${p[0].toFixed(2)} ${p[1].toFixed(2)}`)).join(" ");
  
      el.innerHTML = `
        <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" preserveAspectRatio="none">
          <rect x="0" y="0" width="${w}" height="${h}" fill="rgba(255,255,255,0.02)"/>
          <text x="24" y="48" fill="rgba(232,238,255,0.92)" font-size="22" font-weight="700">${escapeXml(title)}</text>
  
          <line x1="${left}" y1="${top+H}" x2="${w-right}" y2="${top+H}" stroke="rgba(255,255,255,0.12)" />
          <line x1="${left}" y1="${top}" x2="${left}" y2="${top+H}" stroke="rgba(255,255,255,0.12)" />
  
          <path d="${path}" fill="none" stroke="rgba(45,212,191,0.85)" stroke-width="3.2" />
        </svg>
      `;
    }
  
    function demoScatter(el, title = "UMAP map") {
      const w = 900, h = 260;
      const left = 40, right = 30, top = 60, bottom = 30;
      const W = w - left - right;
      const H = h - top - bottom;
  
      const k = 22;
      let dots = "";
      for (let i = 0; i < k; i++) {
        const x = left + Math.random() * W;
        const y = top + Math.random() * H;
        const r = 4 + Math.random() * 3;
        const a = 0.25 + Math.random() * 0.55;
        const useAccent = i % 3 === 0;
        const fill = useAccent ? `rgba(124,92,255,${a.toFixed(3)})` : `rgba(45,212,191,${a.toFixed(3)})`;
        dots += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${fill}" />`;
      }
  
      el.innerHTML = `
        <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" preserveAspectRatio="none">
          <rect x="0" y="0" width="${w}" height="${h}" fill="rgba(255,255,255,0.02)"/>
          <text x="24" y="48" fill="rgba(232,238,255,0.92)" font-size="22" font-weight="700">${escapeXml(title)}</text>
          ${dots}
        </svg>
      `;
    }
  
    function demoBars(el, title = "Top-N contribution") {
      const w = 900, h = 260;
      const left = 60, right = 30, top = 60, bottom = 40;
      const W = w - left - right;
      const H = h - top - bottom;
  
      const labels = ["Top 1", "Top 3", "Top 5", "Top 10"];
      const values = [0.48, 0.62, 0.74, 0.85]; // demo shares
      const barW = W / labels.length;
  
      let bars = "";
      for (let i = 0; i < labels.length; i++) {
        const x = left + i * barW + barW * 0.18;
        const bw = barW * 0.64;
        const v = values[i];
        const bh = v * H;
        const y = top + (H - bh);
        bars += `
          <rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="10" fill="rgba(124,92,255,0.55)"/>
          <text x="${x + bw/2}" y="${top+H+26}" text-anchor="middle" fill="rgba(169,179,216,0.95)" font-size="14">${labels[i]}</text>
        `;
      }
  
      el.innerHTML = `
        <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" preserveAspectRatio="none">
          <rect x="0" y="0" width="${w}" height="${h}" fill="rgba(255,255,255,0.02)"/>
          <text x="24" y="48" fill="rgba(232,238,255,0.92)" font-size="22" font-weight="700">${escapeXml(title)}</text>
          <line x1="${left}" y1="${top+H}" x2="${w-right}" y2="${top+H}" stroke="rgba(255,255,255,0.12)" />
          ${bars}
        </svg>
      `;
    }
  
    function demoTable(el, title = "Scenario table") {
      const w = 900, h = 260;
      const x0 = 24, y0 = 64;
      const cellW = 260, cellH = 46;
  
      const rows = [
        ["Conservative", "Bear: Defensive / low vol", "Bull: Balanced participation"],
        ["Balanced", "Bear: Diversified clusters", "Bull: Sector tilt + ETF"],
        ["Aggressive", "Bear: Controlled risk-on", "Bull: Growth cluster rotation"],
      ];
  
      let cells = "";
      // header
      const headers = ["Profile", "Bear / Stress", "Bull / Expansion"];
      headers.forEach((t, j) => {
        cells += `
          <rect x="${x0 + j*cellW}" y="${y0}" width="${cellW-10}" height="${cellH-6}" rx="10" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.10)" />
          <text x="${x0 + j*cellW + 14}" y="${y0 + 28}" fill="rgba(232,238,255,0.92)" font-size="14" font-weight="700">${escapeXml(t)}</text>
        `;
      });
  
      rows.forEach((r, i) => {
        r.forEach((t, j) => {
          const y = y0 + (i+1)*cellH;
          cells += `
            <rect x="${x0 + j*cellW}" y="${y}" width="${cellW-10}" height="${cellH-6}" rx="10" fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.08)" />
            <text x="${x0 + j*cellW + 14}" y="${y + 28}" fill="rgba(169,179,216,0.95)" font-size="13">${escapeXml(t)}</text>
          `;
        });
      });
  
      el.innerHTML = `
        <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" preserveAspectRatio="none">
          <rect x="0" y="0" width="${w}" height="${h}" fill="rgba(255,255,255,0.02)"/>
          <text x="24" y="44" fill="rgba(232,238,255,0.92)" font-size="22" font-weight="700">${escapeXml(title)}</text>
          ${cells}
        </svg>
      `;
    }
  
    // ---------- Nav highlight ----------
    function setupNavActive() {
      const links = Array.from(document.querySelectorAll(".nav__links a"));
      const targets = links
        .map((a) => document.querySelector(a.getAttribute("href")))
        .filter(Boolean);
  
      if (!targets.length) return;
  
      const obs = new IntersectionObserver(
        (entries) => {
          // pick the most visible intersecting entry
          const visible = entries
            .filter((e) => e.isIntersecting)
            .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
          if (!visible) return;
  
          const id = "#" + visible.target.id;
          links.forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === id));
        },
        { root: null, threshold: [0.2, 0.35, 0.5, 0.65] }
      );
  
      targets.forEach((t) => obs.observe(t));
    }
  
    // ---------- Strategy picker (demo outputs) ----------
    function setupStrategyPicker() {
      const btn = $("#btn_run");
      if (!btn) return;
  
      btn.addEventListener("click", () => {
        const profile = $("#ctl_profile")?.value || "balanced";
        const regime = $("#ctl_regime")?.value || "neutral";
        const exposure = $("#ctl_exposure")?.value || "etf";
  
        const baseReturn = { conservative: 0.06, balanced: 0.10, aggressive: 0.14 }[profile] ?? 0.10;
        const baseVol = { conservative: 0.10, balanced: 0.16, aggressive: 0.23 }[profile] ?? 0.16;
  
        const regimeAdj = {
          bull: { r: +0.03, v: +0.02 },
          neutral: { r: 0.0, v: 0.0 },
          bear: { r: -0.04, v: +0.05 },
        }[regime] ?? { r: 0.0, v: 0.0 };
  
        const exposureAdj = {
          etf: { r: 0.0, v: 0.0 },
          sector_basket: { r: +0.005, v: -0.01 },
          market_category: { r: -0.01, v: -0.02 },
        }[exposure] ?? { r: 0.0, v: 0.0 };
  
        const expR = baseReturn + regimeAdj.r + exposureAdj.r;
        const expV = baseVol + regimeAdj.v + exposureAdj.v;
  
        $("#out_return").textContent = (expR * 100).toFixed(1) + "%";
        $("#out_vol").textContent = (expV * 100).toFixed(1) + "%";
  
        // demo visuals
        const wEl = document.getElementById("viz_strategy_weights");
        const pEl = document.getElementById("viz_strategy_perf");
        if (wEl) demoBars(wEl, "Recommended weights (demo)");
        if (pEl) demoLine(pEl, "Simulated performance (demo)");
      });
    }
  
    // ---------- Render all placeholders ----------
    function renderAll() {
      const map = [
        ["viz_hero_heatmap", () => demoHeatmap(document.getElementById("viz_hero_heatmap"), "Sector co-movement (preview)")],
        ["viz_hero_corr_line", () => demoLine(document.getElementById("viz_hero_corr_line"), "Avg rolling correlation (preview)")],
  
        ["viz_c1_heatmap", () => demoHeatmap(document.getElementById("viz_c1_heatmap"), "Correlation heatmap (full/20y/5y)")],
        ["viz_c1_rollcorr", () => demoLine(document.getElementById("viz_c1_rollcorr"), "Rolling correlation (60d vs 120d)")],
        ["viz_c1_umap", () => demoScatter(document.getElementById("viz_c1_umap"), "UMAP sector map")],
  
        ["viz_c2_umap", () => demoScatter(document.getElementById("viz_c2_umap"), "UMAP inside a sector (sub-clusters)")],
        ["viz_c2_dendrogram", () => drawTitle(document.getElementById("viz_c2_dendrogram"), "Dendrogram", "Hierarchical clustering inside sector")],
        ["viz_c2_norm_index", () => demoLine(document.getElementById("viz_c2_norm_index"), "Proxy vs representative stock (base=100)")],
        ["viz_c2_topn", () => demoBars(document.getElementById("viz_c2_topn"), "Top-N contribution (concentration)")],
        ["viz_c2_te", () => demoLine(document.getElementById("viz_c2_te"), "Tracking error (rolling)")],
  
        ["viz_c3_perf", () => demoLine(document.getElementById("viz_c3_perf"), "ETF vs DIY basket (base=100)")],
        ["viz_c3_te", () => demoLine(document.getElementById("viz_c3_te"), "ETF tracking error (rolling)")],
  
        ["viz_c4_spy_regimes", () => drawTitle(document.getElementById("viz_c4_spy_regimes"), "SPY + regime shading", "Bull / Bear / Neutral timeline")],
        ["viz_c4_winners", () => demoBars(document.getElementById("viz_c4_winners"), "Regime winners/losers (ranked)")],
  
        ["viz_c5_risk_spectrum", () => drawTitle(document.getElementById("viz_c5_risk_spectrum"), "Risk spectrum", "Conservative → Aggressive allocation logic")],
        ["viz_c5_scenario_table", () => demoTable(document.getElementById("viz_c5_scenario_table"), "Profile × Regime (demo)")],
      ];
  
      // mark loading briefly so it feels “alive”
      map.forEach(([id]) => setLoading(id, true));
  
      setTimeout(() => {
        map.forEach(([id, fn]) => {
          const el = document.getElementById(id);
          if (!el) return;
          clear(el);
          fn();
          setLoading(id, false);
        });
      }, 180);
    }
  
    // ---------- boot ----------
    window.addEventListener("DOMContentLoaded", () => {
      setupNavActive();
      setupStrategyPicker();
      renderAll();
    });
  })();


    // Chapter 2 regime toggle
    (function () {
      const buttons = document.querySelectorAll('#c2 .toggle__btn');
      const panels = {
        static: document.getElementById('regime_static'),
        bear: document.getElementById('regime_bear'),
        neutral: document.getElementById('regime_neutral'),
        bull: document.getElementById('regime_bull'),
      };

      if (!buttons.length) return;

      function show(regime) {
        Object.values(panels).forEach(p => p && p.classList.add('hidden'));
        if (panels[regime]) panels[regime].classList.remove('hidden');

        buttons.forEach(b => b.classList.toggle('is-active', b.dataset.regime === regime));
      }

      buttons.forEach(btn => {
        btn.addEventListener('click', () => show(btn.dataset.regime));
      });

      show('static');
    })();


    // Chapter 3 sector toggle
    (function () {
      const root = document.getElementById('c3_right');
      if (!root) return;

      const buttons = root.querySelectorAll('.toggle__btn');

      const panels = {
        basic_materials: document.getElementById('c3_sector_basic_materials'),
        communication_services: document.getElementById('c3_sector_communication_services'),
        consumer_cyclical: document.getElementById('c3_sector_consumer_cyclical'),
        consumer_defensive: document.getElementById('c3_sector_consumer_defensive'),
        energy: document.getElementById('c3_sector_energy'),
        financial_services: document.getElementById('c3_sector_financial_services'),
        healthcare: document.getElementById('c3_sector_healthcare'),
        industrials: document.getElementById('c3_sector_industrials'),
        real_estate: document.getElementById('c3_sector_real_estate'),
        technology: document.getElementById('c3_sector_technology'),
        utilities: document.getElementById('c3_sector_utilities'),
      };

      function show(sector) {
        Object.values(panels).forEach(p => p && p.classList.add('hidden'));
        if (panels[sector]) panels[sector].classList.remove('hidden');

        buttons.forEach(b => b.classList.toggle('is-active', b.dataset.sector === sector));
      }

      buttons.forEach(btn => {
        btn.addEventListener('click', () => show(btn.dataset.sector));
      });

      show('basic_materials');
    })();

    // ----------------------------
// Chapter 3: sector toggle + interpretation load
// ----------------------------
let c3Writeups = null;

async function c3LoadWriteups() {
  // Adjust path if you place the JSON elsewhere
  const res = await fetch("content/chapter3_writeups.json");
  if (!res.ok) {
    throw new Error(`Failed to load chapter3_writeups.json (${res.status})`);
  }
  c3Writeups = await res.json();
}

function c3SetInterpretation(sector) {
  const box = document.getElementById("c3_interpretation");
  if (!box) return;

  const entry = c3Writeups && c3Writeups[sector];

  if (entry && entry.html) {
    // Your JSON contains HTML already (converted from markdown)
    box.innerHTML = `
      <p><strong>Interpretation:</strong></p>
      ${entry.html}
    `;
  } else {
    box.innerHTML = `
      <p><strong>Interpretation:</strong></p>
      <p>Interpretation not ready for sector <strong>${sector}</strong>.</p>
    `;
  }
}

function c3ShowSector(sector) {
  // Toggle panels
  document.querySelectorAll("#c3 .sector-panel").forEach((p) => p.classList.add("hidden"));
  const panel = document.getElementById(`c3_sector_${sector}`);
  if (panel) panel.classList.remove("hidden");

  // Update interpretation text
  c3SetInterpretation(sector);
}

function c3WireButtons() {
  const root = document.getElementById("c3");
  if (!root) return;

  const buttons = root.querySelectorAll(".toggle__btn[data-sector]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const sector = btn.getAttribute("data-sector");
      if (!sector) return;

      // active button state
      buttons.forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");

      c3ShowSector(sector);
    });
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  // Wire toggles immediately (works even if writeups fail to load)
  c3WireButtons();

  // Default sector = currently active button (fallback: first button)
  const activeBtn =
    document.querySelector("#c3 .toggle__btn.is-active") ||
    document.querySelector("#c3 .toggle__btn[data-sector]");
  const defaultSector = activeBtn ? activeBtn.getAttribute("data-sector") : null;

  // Load writeups + render
  try {
    await c3LoadWriteups();
  } catch (err) {
    console.error(err);
    // still allow showing panels; interpretation will fall back to placeholder
  }

  if (defaultSector) c3ShowSector(defaultSector);
});


  