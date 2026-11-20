// Chapter 1 — interactive charts (diversifier dot plot + clustered correlation heatmap).
// Data: content/chapter1.json, built by chapter1_sector_diversification.ipynb.
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  // Validated categorical slots (dark surface) for the three windows
  const WINDOW_KEYS = ["full", "20y", "5y"];
  const WINDOW_COLORS = { full: "#3987e5", "20y": "#d95926", "5y": "#199e70" };
  const WINDOW_SHORT = { full: "Full sample", "20y": "Last 20Y", "5y": "Last 5Y" };
  const INK = { text: "#E8EEFF", muted: "#A9B3D8", faint: "#6F7AA8", grid: "rgba(255,255,255,0.08)" };
  // Brighter blue = sectors move together more, in every mode.
  // Sequential (returns, all positive): one blue hue, dark -> bright.
  // Diverging (volatility, change): red (negative) <- neutral gray -> blue (positive).
  const SEQ = { lo: [22, 36, 74], hi: [158, 197, 244] };
  const DIV = { neg: [230, 103, 103], mid: [56, 56, 53], pos: [109, 167, 236] };

  const state = { dotMetric: "returns", heatMetric: "returns", heatWindow: "full" };
  let DATA = null;

  // ---------- helpers ----------
  function el(name, attrs = {}, parent) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }
  function text(parent, x, y, str, attrs = {}) {
    const t = el("text", { x, y, fill: INK.muted, "font-size": 12, ...attrs }, parent);
    t.textContent = str;
    return t;
  }
  const fmt = (v) => (v >= 0 ? "" : "−") + Math.abs(v).toFixed(2);
  const fmtSigned = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2);

  function mix(a, b, t) {
    const c = a.map((x, i) => Math.round(x + (b[i] - x) * t));
    return { css: `rgb(${c[0]},${c[1]},${c[2]})`, lum: (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255 };
  }
  // scale = { kind: "seq", max } or { kind: "div", max }
  function cellColor(v, scale) {
    if (scale.kind === "seq") return mix(SEQ.lo, SEQ.hi, Math.max(0, Math.min(1, v / scale.max)));
    const t = Math.max(-1, Math.min(1, v / scale.max));
    return mix(DIV.mid, t >= 0 ? DIV.pos : DIV.neg, Math.abs(t));
  }

  // One tooltip per chart container; values lead, labels follow
  function makeTooltip(container) {
    const tip = document.createElement("div");
    tip.className = "c1-tip hidden";
    container.appendChild(tip);
    return {
      show(evt, title, rows) {
        tip.replaceChildren();
        const h = document.createElement("div");
        h.className = "c1-tip__title";
        h.textContent = title;
        tip.appendChild(h);
        for (const r of rows) {
          const row = document.createElement("div");
          row.className = "c1-tip__row";
          if (r.color) {
            const key = document.createElement("span");
            key.className = "c1-tip__key";
            key.style.background = r.color;
            row.appendChild(key);
          }
          const val = document.createElement("strong");
          val.textContent = r.value;
          const lab = document.createElement("span");
          lab.textContent = r.label;
          row.append(val, lab);
          tip.appendChild(row);
        }
        tip.classList.remove("hidden");
        const box = container.getBoundingClientRect();
        let x = evt.clientX - box.left + 14;
        let y = evt.clientY - box.top + 14;
        if (x + tip.offsetWidth > box.width) x = evt.clientX - box.left - tip.offsetWidth - 14;
        tip.style.left = `${Math.max(0, x)}px`;
        tip.style.top = `${y}px`;
      },
      hide() { tip.classList.add("hidden"); },
    };
  }

  // ---------- chart 1: diversifier dot plot ----------
  function renderDots() {
    const host = document.getElementById("c1_dots");
    const wrap = host.querySelector(".c1-chart");
    const m = DATA.metrics[state.dotMetric];
    const S = DATA.sectors;

    const rows = S.map((name, i) => {
      const vals = Object.fromEntries(WINDOW_KEYS.map(w => [w, m.avg_corr[w][i]]));
      return { name, vals };
    }).sort((a, b) => a.vals.full - b.vals.full);   // most independent sector first

    const W = wrap.clientWidth || 700;
    const narrow = W < 560;
    const labelW = narrow ? 118 : 178, rowH = 32, top = 14, bottom = 30;
    const plotW = W - labelW - 16;
    const all = rows.flatMap(r => WINDOW_KEYS.map(w => r.vals[w]));
    const lo = Math.min(0, Math.floor((Math.min(...all) - 0.02) * 20) / 20);
    const hi = Math.ceil((Math.max(...all) + 0.02) * 20) / 20;
    const x = (v) => labelW + ((v - lo) / (hi - lo)) * plotW;
    const H = top + rows.length * rowH + bottom;

    wrap.replaceChildren();
    const svg = el("svg", { width: W, height: H, role: "img",
      "aria-label": `Average correlation of each sector with the others, by window (${state.dotMetric})` }, wrap);

    // x grid + axis
    for (let v = lo; v <= hi + 1e-9; v += 0.05) {
      const major = Math.abs(Math.round(v * 100) % 10) === 0;
      el("line", { x1: x(v), x2: x(v), y1: top - 6, y2: top + rows.length * rowH,
        stroke: INK.grid, "stroke-width": major ? 1 : 0.5 }, svg);
      if (major || !narrow) text(svg, x(v), H - 10, fmt(v), { "text-anchor": "middle", fill: INK.faint, "font-size": 11 });
    }

    const tip = host._tip || (host._tip = makeTooltip(host));
    rows.forEach((r, k) => {
      const cy = top + k * rowH + rowH / 2;
      const g = el("g", { tabindex: 0, class: "c1-row" }, svg);
      el("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: "transparent", class: "c1-row__hit" }, g);
      text(g, labelW - 10, cy + 4, r.name, { "text-anchor": "end", fill: INK.text, "font-size": narrow ? 11 : 12.5 });
      const vs = WINDOW_KEYS.map(w => r.vals[w]);
      el("line", { x1: x(Math.min(...vs)), x2: x(Math.max(...vs)), y1: cy, y2: cy,
        stroke: "rgba(255,255,255,0.28)", "stroke-width": 2, "stroke-linecap": "round" }, g);
      const dy = { full: -5, "20y": 0, "5y": 5 };  // keeps equal values visible
      WINDOW_KEYS.forEach((w) => {
        el("circle", { cx: x(r.vals[w]), cy: cy + dy[w], r: 5.5, fill: WINDOW_COLORS[w], stroke: "#0E1630", "stroke-width": 1.5 }, g);
      });
      const show = (evt) => tip.show(evt, `${r.name} — average correlation with the other sectors`,
        WINDOW_KEYS.map((w) => ({ color: WINDOW_COLORS[w], value: fmt(r.vals[w]), label: WINDOW_SHORT[w] })));
      g.addEventListener("pointermove", show);
      g.addEventListener("pointerleave", () => tip.hide());
      g.addEventListener("focus", () => {
        const b = g.getBoundingClientRect();
        show({ clientX: b.left + labelW, clientY: b.top + rowH });
      });
      g.addEventListener("blur", () => tip.hide());
    });

    // table view (same numbers, no hover needed)
    const table = host.querySelector(".c1-table");
    table.replaceChildren();
    const head = table.insertRow();
    ["Sector", ...WINDOW_KEYS.map(w => WINDOW_SHORT[w])].forEach(h => {
      const th = document.createElement("th"); th.textContent = h; head.appendChild(th);
    });
    rows.forEach(r => {
      const tr = table.insertRow();
      [r.name, ...WINDOW_KEYS.map(w => fmt(r.vals[w]))]
        .forEach(v => { tr.insertCell().textContent = v; });
    });

    host.querySelectorAll(".c1-dot-note").forEach(n => n.classList.toggle("hidden", n.dataset.metric !== state.dotMetric));
  }

  // ---------- chart 2: clustered correlation heatmap ----------
  function renderHeat() {
    const host = document.getElementById("c1_heat");
    const wrap = host.querySelector(".c1-chart");
    const m = DATA.metrics[state.heatMetric];
    const order = m.order;
    const S = order.map(i => DATA.sectors[i]);
    const isChange = state.heatWindow === "change";
    const value = (i, j) => isChange
      ? m.corr["5y"][order[i]][order[j]] - m.corr.full[order[i]][order[j]]
      : m.corr[state.heatWindow][order[i]][order[j]];
    const scale = isChange ? { kind: "div", max: 0.4 }
      : state.heatMetric === "returns" ? { kind: "seq", max: 0.6 } : { kind: "div", max: 0.8 };

    const W = wrap.clientWidth || 700;
    const narrow = W < 560;
    const labelW = narrow ? 112 : 170, colLabelH = narrow ? 92 : 120;
    const n = S.length - 1;                       // lower triangle without the diagonal
    const cell = Math.min(54, Math.floor((W - labelW - 4) / n));
    const H = n * cell + colLabelH + 8;

    wrap.replaceChildren();
    const svg = el("svg", { width: W, height: H, role: "img",
      "aria-label": `Sector correlation heatmap, ${state.heatMetric}, ${isChange ? "change from full sample to last 5 years" : WINDOW_SHORT[state.heatWindow]}` }, wrap);
    const tip = host._tip || (host._tip = makeTooltip(host));

    for (let i = 1; i < S.length; i++) {
      const y = (i - 1) * cell;
      text(svg, labelW - 8, y + cell / 2 + 4, S[i], { "text-anchor": "end", fill: INK.text, "font-size": narrow ? 10.5 : 12 });
      for (let j = 0; j < i; j++) {
        const v = value(i, j);
        const c = cellColor(v, scale);
        const xx = labelW + j * cell;
        const g = el("g", { tabindex: 0, class: "c1-cell" }, svg);
        el("rect", { x: xx + 1, y: y + 1, width: cell - 2, height: cell - 2, rx: 4, fill: c.css }, g);
        if (cell >= 34) {
          text(g, xx + cell / 2, y + cell / 2 + 4, isChange ? fmtSigned(v) : fmt(v),
            { "text-anchor": "middle", "font-size": 11, fill: c.lum > 0.45 ? "#0b0b0b" : INK.text });
        }
        const rowsTip = isChange
          ? [{ value: fmtSigned(v), label: "change, last 5Y vs full sample" },
             { value: fmt(m.corr.full[order[i]][order[j]]), label: "full sample" },
             { value: fmt(m.corr["5y"][order[i]][order[j]]), label: "last 5 years" }]
          : [{ value: fmt(v), label: `correlation, ${WINDOW_SHORT[state.heatWindow]}` }];
        const show = (evt) => tip.show(evt, `${S[i]} × ${S[j]}`, rowsTip);
        g.addEventListener("pointermove", show);
        g.addEventListener("pointerleave", () => tip.hide());
        g.addEventListener("focus", () => { const b = g.getBoundingClientRect(); show({ clientX: b.right, clientY: b.bottom }); });
        g.addEventListener("blur", () => tip.hide());
      }
    }
    // column labels (rotated)
    for (let j = 0; j < n; j++) {
      const cx = labelW + j * cell + cell / 2, cy = n * cell + 10;
      text(svg, cx, cy, S[j], { "text-anchor": "end", fill: INK.text, "font-size": narrow ? 10.5 : 12,
        transform: `rotate(-45 ${cx} ${cy})`, dy: "0.6em" });
    }

    // legend bar
    const legend = host.querySelector(".c1-legend");
    const ts = scale.kind === "seq" ? [0, 0.25, 0.5, 0.75, 1] : [-1, -0.5, 0, 0.5, 1];
    const stops = ts.map(t => cellColor(t * scale.max, scale).css);
    legend.querySelector(".c1-legend__bar").style.background = `linear-gradient(90deg, ${stops.join(",")})`;
    const lo = scale.kind === "seq" ? "0 (independent)" : isChange ? `−${scale.max} (moved apart)` : `−${scale.max} (opposite)`;
    const hi = isChange ? `+${scale.max} (moved together)` : `${scale.max} (move together)`;
    legend.querySelector(".c1-legend__lo").textContent = lo;
    legend.querySelector(".c1-legend__hi").textContent = hi;

    const mean = isChange ? null : m.mean_pair_corr[state.heatWindow];
    host.querySelector(".c1-heat-stat").textContent = isChange
      ? "Blue = the pair moves together more in 2015–2020 than over the full sample; red = it decoupled."
      : `Average pair correlation in this window: ${fmt(mean)} · window: ${DATA.windows[state.heatWindow]}`;
    host.querySelectorAll(".c1-heat-note").forEach(n => n.classList.toggle("hidden", n.dataset.metric !== state.heatMetric));
  }

  // ---------- wiring ----------
  function wireToggle(selector, key, render) {
    const btns = document.querySelectorAll(selector);
    btns.forEach(b => b.addEventListener("click", () => {
      state[key] = b.dataset.value;
      btns.forEach(x => x.classList.toggle("is-active", x === b));
      render();
    }));
  }

  async function init() {
    if (!document.getElementById("c1_dots")) return;
    try {
      const res = await fetch("content/chapter1.json", { cache: "no-store" });
      DATA = await res.json();
    } catch (e) {
      console.error("Chapter 1 data failed to load", e);
      return;
    }
    wireToggle("#c1_dots [data-key='dotMetric']", "dotMetric", renderDots);
    wireToggle("#c1_heat [data-key='heatMetric']", "heatMetric", renderHeat);
    wireToggle("#c1_heat [data-key='heatWindow']", "heatWindow", renderHeat);
    renderDots();
    renderHeat();
    let t, lastWidth = 0;
    new ResizeObserver((entries) => {
      const w = Math.round(entries[0].contentRect.width);
      if (w === lastWidth) return;            // re-render on width changes only
      lastWidth = w;
      clearTimeout(t);
      t = setTimeout(() => { renderDots(); renderHeat(); }, 120);
    }).observe(document.getElementById("c1"));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
