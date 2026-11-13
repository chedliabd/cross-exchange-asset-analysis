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


  

// ----------------------------
// Chapter 5: load story + plots + sector toggle (scoped)
// ----------------------------
// Embedded fallback so the page still works when opened via file://
const c5EmbeddedData = {
  chapter_title: "Chapter 5 — Sector Proxy vs Stock Representativeness",
  chapter_story: [
    "# Sector Proxy vs Stock Representativeness",
    "If an investor wants sector exposure, is one representative stock enough, or do we need a small basket to avoid idiosyncratic risk? We test this directly: build a sector proxy from weighted stock returns, filter illiquid names, score representativeness with rolling correlations, apply a lifetime-aware ranking rule, and validate with tracking error through time.",
    "2.1 Intra-Sector Structure: We compute pairwise-overlap correlations (no forced common window), transform to distances, run UMAP and agglomerative clustering, and explicitly tag Missing Data. Most sectors collapse into a dominant cluster (e.g., Technology 2 clusters/325 stocks; main cluster 97.2%), while some show real segmentation (Financial Services 6 clusters/598; Basic Materials main cluster 73.4%; Utilities 3 clusters/28 with only 57% in the largest). Structure tells us where simplification is plausible.",
    "2.2 From Structure to Candidates: Representativeness = average rolling correlation to the sector proxy (60d), after liquidity filtering. We weight by lifetime coverage: effective_score = rep_score × lifetime_fraction^alpha (alpha=2) so short-lived tickers are penalized. Volatility breaks ties. Output: one top representative and a Top-5 basket per sector.",
    "2.3 Return Path Check: We compare normalized indices (base=100) for proxy vs single vs basket. Some sectors are well represented by one stock (e.g., Energy single XOM corr 0.725 vs basket 0.751; TE similar). Others need the basket (Technology single HPQ corr 0.612 / TE 0.328 vs basket corr 0.802 / TE 0.228; Healthcare single 0.605 / TE 0.284 vs basket 0.800 / TE 0.199). Communication Services shows marginal basket benefit (DIS vs basket nearly identical).",
    "2.4 Tracking Error: TE_t = rolling std(active return). Energy and Communication Services show nearly identical TE single vs basket. Basic Materials and Technology show meaningful TE reduction with the basket (e.g., Basic Materials single p95 0.028 → basket 0.021; Tech single p95 0.033 → basket 0.023). Consumer Cyclical shows large spikes even with a basket (single p95 0.060, max 0.370; basket p95 0.054, max 0.368).",
    "2.5 Takeaways: Energy (and close to it Communication Services) can be represented by a single stock. Technology, Healthcare, Financial Services, Industrials, Consumer Defensive, Utilities, Basic Materials are safer with a basket. Consumer Cyclical is highly regime-dependent; a basket helps but cannot fully tame deviations.",
    "2.6 Investor Lens: Simplification is conditional. Use a single-stock proxy only where TE stays close to the basket; otherwise a small basket is not optional. The Top-5 basket, lifetime-weighted and equal-weighted, captures most diversification while staying implementable."
  ],
  sectors: {
    basic_materials: {
      plot1: { src: "chapter_5_plots/plots/basic_materials/basic_materials_umap.html", alt: "Basic Materials – UMAP", interpretation: "2 clusters / 128 stocks; largest cluster 94 (73.4%). Structure less collapsed than most, signaling internal diversification (metals vs chemicals/value chains)." },
      plot2: { src: "chapter_5_plots/plots/basic_materials/basic_materials_index.html", alt: "Basic Materials – Sector proxy vs reps", interpretation: "Single AA corr 0.681, TE 0.262; basket (AA, DD, OLN, NUE, PPG) corr 0.801, TE 0.174. Basket clearly improves alignment." },
      plot3: { src: "chapter_5_plots/plots/basic_materials/basic_materials_te.html", alt: "Basic Materials – Tracking error", interpretation: "TE(60d): single p95 0.028 (max 0.047) vs basket p95 0.021 (max 0.035). Basket materially lowers TE; single is inadequate." }
    },
    communication_services: {
      plot1: { src: "chapter_5_plots/plots/communication_services/communication_services_umap.html", alt: "Communication Services – UMAP", interpretation: "2 clusters / 117 stocks; largest cluster 105 (89.7%). Concentrated structure (telcos vs platforms) but mostly one dominant cluster." },
      plot2: { src: "chapter_5_plots/plots/communication_services/communication_services_index.html", alt: "Communication Services – Sector proxy vs reps", interpretation: "Single DIS corr 0.674, TE 0.277; basket (DIS, CMCSA, NYT, IPG, OMC) corr 0.665, TE 0.271. Basket adds little." },
      plot3: { src: "chapter_5_plots/plots/communication_services/communication_services_te.html", alt: "Communication Services – Tracking error", interpretation: "TE(60d): single p95 0.032 (max 0.074) vs basket p95 0.029 (max 0.074). Nearly identical; single stock largely sufficient here." }
    },
    consumer_cyclical: {
      plot1: { src: "chapter_5_plots/plots/consumer_cyclical/consumer_cyclical_umap.html", alt: "Consumer Cyclical – UMAP", interpretation: "2 clusters / 277 stocks; largest cluster 265 (95.7%). Highly collapsed structure." },
      plot2: { src: "chapter_5_plots/plots/consumer_cyclical/consumer_cyclical_index.html", alt: "Consumer Cyclical – Sector proxy vs reps", interpretation: "Single IP corr 0.583, TE 0.295; basket (IP, MCD, GT, F, HD) corr 0.786, TE 0.226. Basket improves alignment but drift remains." },
      plot3: { src: "chapter_5_plots/plots/consumer_cyclical/consumer_cyclical_te.html", alt: "Consumer Cyclical – Tracking error", interpretation: "TE(60d): single p95 0.060 (max 0.370) vs basket p95 0.054 (max 0.368). Large spikes persist; single stock inadequate, basket only partially tames volatility." }
    },
    consumer_defensive: {
      plot1: { src: "chapter_5_plots/plots/consumer_defensive/consumer_defensive_umap.html", alt: "Consumer Defensive – UMAP", interpretation: "2 clusters / 116 stocks; largest cluster 105 (90.5%). Strongly concentrated staples core." },
      plot2: { src: "chapter_5_plots/plots/consumer_defensive/consumer_defensive_index.html", alt: "Consumer Defensive – Sector proxy vs reps", interpretation: "Single PG corr 0.598, TE 0.207; basket (PG, WMT, PEP, TGT, CL) corr 0.789, TE 0.152. Basket improves path stability." },
      plot3: { src: "chapter_5_plots/plots/consumer_defensive/consumer_defensive_te.html", alt: "Consumer Defensive – Tracking error", interpretation: "TE(60d): single p95 0.029 (max 0.104) vs basket p95 0.025 (max 0.105). Basket modestly lowers TE; single is usable but basket more robust." }
    },
    energy: {
      plot1: { src: "chapter_5_plots/plots/energy/energy_umap.html", alt: "Energy – UMAP", interpretation: "3 clusters / 133 stocks; largest cluster 118 (88.7%). Mostly one dominant group." },
      plot2: { src: "chapter_5_plots/plots/energy/energy_index.html", alt: "Energy – Sector proxy vs reps", interpretation: "Single XOM corr 0.725, TE 0.169; basket (XOM, CVX, BP, SLB, NBR) corr 0.751, TE 0.168. Basket adds little." },
      plot3: { src: "chapter_5_plots/plots/energy/energy_te.html", alt: "Energy – Tracking error", interpretation: "TE(60d): single p95 0.026 (max 0.078) vs basket p95 0.027 (max 0.074). Nearly identical; single stock sufficient here." }
    },
    financial_services: {
      plot1: { src: "chapter_5_plots/plots/financial_services/financial_services_umap.html", alt: "Financial Services – UMAP", interpretation: "6 clusters / 598 stocks; largest cluster 508 (84.9%). Rich segmentation (banks/insurers/fintech)." },
      plot2: { src: "chapter_5_plots/plots/financial_services/financial_services_index.html", alt: "Financial Services – Sector proxy vs reps", interpretation: "Single WFC corr 0.545, TE 0.263; basket (WFC, C, MMC, CNA, PNC) corr 0.696, TE 0.178. Basket clearly better." },
      plot3: { src: "chapter_5_plots/plots/financial_services/financial_services_te.html", alt: "Financial Services – Tracking error", interpretation: "TE(60d): single p95 0.032 (max 0.114) vs basket p95 0.023 (max 0.113). Basket lowers TE, especially in stress." }
    },
    healthcare: {
      plot1: { src: "chapter_5_plots/plots/healthcare/healthcare_umap.html", alt: "Healthcare – UMAP", interpretation: "2 clusters / 442 stocks; largest cluster 432 (97.7%). Extremely concentrated structure." },
      plot2: { src: "chapter_5_plots/plots/healthcare/healthcare_index.html", alt: "Healthcare – Sector proxy vs reps", interpretation: "Single JNJ corr 0.605, TE 0.284; basket (JNJ, MRK, PFE, LLY, MDT) corr 0.800, TE 0.199. Basket improves path." },
      plot3: { src: "chapter_5_plots/plots/healthcare/healthcare_te.html", alt: "Healthcare – Tracking error", interpretation: "TE(60d): single p95 0.043 (max 0.074) vs basket p95 0.034 (max 0.073). Basket consistently lowers TE; single not sufficient." }
    },
    industrials: {
      plot1: { src: "chapter_5_plots/plots/industrials/industrials_umap.html", alt: "Industrials – UMAP", interpretation: "In the UMAP, the sector splits into 2 clusters over 307 stocks. The largest cluster (Cluster 1, 295 stocks) contains 96% of the sector." },
      plot2: { src: "chapter_5_plots/plots/industrials/industrials_index.html", alt: "Industrials – Sector proxy vs reps", interpretation: "Single representative (GE) corr 0.531 with annualized TE 0.171; Top-5 basket improves stability." },
      plot3: { src: "chapter_5_plots/plots/industrials/industrials_te.html", alt: "Industrials – Tracking error", interpretation: "Basket keeps TE lower and steadier; single name spikes during volatility shocks." }
    },
    real_estate: {
      plot1: { src: "chapter_5_plots/plots/real_estate/real_estate_umap.html", alt: "Real Estate – UMAP", interpretation: "2 clusters / 152 stocks; largest cluster 134 (88.2%)." },
      plot2: { src: "chapter_5_plots/plots/real_estate/real_estate_index.html", alt: "Real Estate – Sector proxy vs reps", interpretation: "Single WY corr 0.431, TE 0.243; basket (WY, HST, FRT, VNO, PCH) corr 0.578, TE 0.173. Basket improves tracking." },
      plot3: { src: "chapter_5_plots/plots/real_estate/real_estate_te.html", alt: "Real Estate – Tracking error", interpretation: "TE(60d): single p95 0.041 (max 0.199) vs basket p95 0.032 (max 0.199). Basket clearly better." }
    },
    technology: {
      plot1: { src: "chapter_5_plots/plots/technology/technology_umap.html", alt: "Technology – UMAP", interpretation: "2 clusters / 325 stocks; largest cluster 316 (97.2%). Extremely collapsed structure." },
      plot2: { src: "chapter_5_plots/plots/technology/technology_index.html", alt: "Technology – Sector proxy vs reps", interpretation: "Single HPQ corr 0.612, TE 0.328; basket (HPQ, IBM, INTC, AAPL, TER) corr 0.802, TE 0.228. Basket markedly better." },
      plot3: { src: "chapter_5_plots/plots/technology/technology_te.html", alt: "Technology – Tracking error", interpretation: "TE(60d): single p95 0.033 (max 0.078) vs basket p95 0.023 (max 0.059). Basket significantly reduces TE." }
    },
    utilities: {
      plot1: { src: "chapter_5_plots/plots/utilities/utilities_umap.html", alt: "Utilities – UMAP", interpretation: "3 clusters / 28 stocks; largest cluster 16 (57.1%). Least concentrated structure." },
      plot2: { src: "chapter_5_plots/plots/utilities/utilities_index.html", alt: "Utilities – Sector proxy vs reps", interpretation: "Single ED corr 0.549, TE 0.206; basket (ED, DTE, AEP, CNP, NEE) corr 0.673, TE 0.170. Basket improves stability." },
      plot3: { src: "chapter_5_plots/plots/utilities/utilities_te.html", alt: "Utilities – Tracking error", interpretation: "TE(60d): single p95 0.026 (max 0.069) vs basket p95 0.020 (max 0.070). Basket consistently lowers TE." }
    }
  }
};

let c5Data = null;

async function c5LoadData() {
  // If opened via file://, fetch is often blocked; fall back to embedded data.
  if (window.location.protocol === "file:") {
    c5Data = c5EmbeddedData;
    return c5Data;
  }
  const res = await fetch("./chapter_5_plots/chapter5_text.json");
  if (!res.ok) throw new Error(`Failed to load chapter5_text.json (${res.status})`);
  c5Data = await res.json();
  return c5Data;
}

function c5PopulateStory() {
  const storyCard = document.getElementById("c5_story");
  if (!storyCard || !c5Data) return;

  storyCard.innerHTML = "";
  const h4 = document.createElement("h4");
  h4.textContent = "Story";
  storyCard.appendChild(h4);

  const story = Array.isArray(c5Data.chapter_story) ? c5Data.chapter_story : [c5Data.chapter_story || ""];
  story.forEach((para) => {
    if (!para) return;
    const p = document.createElement("p");
    p.textContent = para;
    storyCard.appendChild(p);
  });
}

function c5PopulateSector(sector) {
  if (!c5Data || !c5Data.sectors) return;
  const sectorData = c5Data.sectors[sector];
  if (!sectorData) return;

  const panel = document.querySelector(`#c5 [data-sector-panel="${sector}"]`);
  if (!panel) return;

  ["plot1", "plot2", "plot3"].forEach((key) => {
    const plotInfo = sectorData[key];
    const img = panel.querySelector(`img[data-plot="${key}"]`);
    const note = panel.querySelector(`.note[data-note="${key}"]`);

    if (img && plotInfo) {
      const viz = img.parentElement;
      const src = plotInfo.src || "";
      const alt = plotInfo.alt || `${sector} ${key}`;
      if (src.toLowerCase().endsWith(".html") && viz) {
        viz.innerHTML = `<iframe src="${src}" title="${alt}" style="width:100%; height:260px; border:0;" loading="lazy"></iframe>`;
      } else {
        img.src = src;
        img.alt = alt;
      }
    }

    if (note) {
      const txt = plotInfo && plotInfo.interpretation;
      note.innerHTML = txt ? `<p>${txt}</p>` : "<p>Interpretation coming soon.</p>";
    }
  });
}

function c5ShowSector(sector) {
  const root = document.getElementById("c5_right");
  if (!root) return;

  const panels = root.querySelectorAll("[data-sector-panel]");
  panels.forEach((p) => p.classList.add("hidden"));

  const target = root.querySelector(`[data-sector-panel="${sector}"]`);
  if (target) {
    target.classList.remove("hidden");
    c5PopulateSector(sector);
  }

  const buttons = root.querySelectorAll(".toggle__btn");
  buttons.forEach((b) => b.classList.toggle("is-active", b.dataset.sector === sector));
}

function c5WireButtons() {
  const root = document.getElementById("c5_right");
  if (!root) return;
  const buttons = root.querySelectorAll(".toggle__btn[data-sector]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const sector = btn.dataset.sector;
      if (!sector) return;
      c5ShowSector(sector);
    });
  });
}

async function c5Init() {
  const fallback = document.getElementById("c5_fallback");
  try {
    await c5LoadData();
    if (c5Data?.chapter_title) {
      const titleEl = document.querySelector("#c5 .block__title h3");
      if (titleEl) titleEl.textContent = c5Data.chapter_title;
    }
    c5WireButtons();
    c5ShowSector("technology");
    if (fallback) fallback.classList.add("hidden");
  } catch (err) {
    console.error(err);
    if (fallback) fallback.classList.remove("hidden");
  }
}
// ----------------------------
// Chapter 1: funnel metric toggle
// ----------------------------
function c1ShowMetric(metric) {
  // Toggle panels
  document
    .querySelectorAll("#c1_right .funnel-panel")
    .forEach((p) => p.classList.add("hidden"));

  const panel = document.getElementById(`c1_funnel_${metric}`);
  if (panel) panel.classList.remove("hidden");

  // Update active button
  document
    .querySelectorAll("#c1_right .toggle__btn")
    .forEach((b) =>
      b.classList.toggle("is-active", b.dataset.funnelMetric === metric)
    );
}

function c1WireButtons() {
  const root = document.getElementById("c1_right");
  if (!root) return;

  const buttons = root.querySelectorAll(".toggle__btn[data-funnel-metric]");
  if (!buttons.length) return;

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const metric = btn.dataset.funnelMetric;
      if (!metric) return;
      c1ShowMetric(metric);
    });
  });
}

window.addEventListener("DOMContentLoaded", () => {
  // --- Chapter 1 Init ---
  c1WireButtons();

  const activeBtn =
    document.querySelector("#c1_right .toggle__btn.is-active") ||
    document.querySelector("#c1_right .toggle__btn[data-funnel-metric]");

  const defaultMetric = activeBtn
    ? activeBtn.dataset.funnelMetric
    : "returns";

  c1ShowMetric(defaultMetric);

  // --- Chapter 5 Init (The Fix) ---
  // This was missing!
  c5Init(); 
});

function c1WirePCAButtons() {
  const root = document.getElementById("c1_pca_right");
  if (!root) return;

  // Track current selections
  let currentMetric = "returns";
  let currentWindow = "full";

  const metricBtns = document.querySelectorAll("[data-pca-metric]");
  const windowBtns = document.querySelectorAll("[data-pca-window]");

  function updatePCAView() {
    // 1. Hide ALL image panels
    document.querySelectorAll(".pca-panel").forEach(p => p.classList.add("hidden"));
    
    // 2. Show the specific image for the combination (e.g., pca_returns_5y)
    const targetId = `pca_${currentMetric}_${currentWindow}`;
    const targetPanel = document.getElementById(targetId);
    if (targetPanel) targetPanel.classList.remove("hidden");

    // 3. Toggle Interpretation (Only dependent on Metric)
    document.getElementById("pca_interp_returns").classList.toggle("hidden", currentMetric !== "returns");
    document.getElementById("pca_interp_volatility").classList.toggle("hidden", currentMetric !== "volatility");
  }

  // Metric Click Logic
  metricBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      currentMetric = btn.dataset.pcaMetric;
      metricBtns.forEach(b => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      updatePCAView();
    });
  });

  // Window Click Logic
  windowBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      currentWindow = btn.dataset.pcaWindow;
      windowBtns.forEach(b => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      updatePCAView();
    });
  });
}

window.addEventListener("DOMContentLoaded", () => {
  c1WireButtons();    // Existing Funnel buttons
  c1WirePCAButtons(); // New PCA buttons
  c5Init();           // Chapter 5
});




// Chapter 6 figure toggle
(function () {
  const root = document.getElementById("c6_right");
  if (!root) return;
  const buttons = root.querySelectorAll(".toggle__btn[data-c6]");
  const panels = root.querySelectorAll(".c6-panel");
  function show(name) {
    panels.forEach(p => p.classList.toggle("hidden", p.id !== `c6_${name}`));
    buttons.forEach(b => b.classList.toggle("is-active", b.dataset.c6 === name));
  }
  buttons.forEach(b => b.addEventListener("click", () => show(b.dataset.c6)));
  show("performance");
})();
