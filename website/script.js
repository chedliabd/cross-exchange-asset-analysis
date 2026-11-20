/* -----------------------------------------
   Markets, Decoded: page interactions
   (Chapter 1 charts live in chapter1.js, the Strategy Picker in index.html)
   ----------------------------------------- */

// Generic toggle: clicking a button shows its panel and hides the others.
function wireToggle({ buttons, panels, key, matches, initial, onShow }) {
  function show(value) {
    panels.forEach((p) => p.classList.toggle("hidden", !matches(p, value)));
    buttons.forEach((b) => b.classList.toggle("is-active", b.dataset[key] === value));
    if (onShow) onShow(value);
  }
  buttons.forEach((b) => b.addEventListener("click", () => show(b.dataset[key])));
  if (initial) show(initial);
}

// ---------- Navigation: highlight the section in view ----------
(function () {
  const links = Array.from(document.querySelectorAll(".nav__links a"));
  const targets = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  if (!targets.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((e) => e.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      const id = "#" + visible.target.id;
      links.forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === id));
    },
    { threshold: [0.2, 0.35, 0.5, 0.65] }
  );
  targets.forEach((t) => observer.observe(t));
})();

// ---------- Chapter 2: regime toggle ----------
wireToggle({
  buttons: document.querySelectorAll("#c2 .toggle__btn[data-regime]"),
  panels: document.querySelectorAll("#c2 .regime-panel"),
  key: "regime",
  matches: (p, regime) => p.id === `regime_${regime}`,
  initial: "static",
});

// ---------- Chapter 4: sector toggle + hand-written interpretation ----------
(async function () {
  const box = document.getElementById("c4_interpretation");
  let writeups = {};

  function setInterpretation(sector) {
    if (!box) return;
    const entry = writeups[sector];
    box.innerHTML = `<p><strong>Interpretation:</strong></p>` +
      (entry && entry.html ? entry.html : `<p>Interpretation not available for this sector.</p>`);
  }

  try {
    const res = await fetch("content/chapter4_writeups.json");
    if (!res.ok) throw new Error(`Failed to load chapter4_writeups.json (${res.status})`);
    writeups = await res.json();
  } catch (err) {
    console.error(err);
  }

  const active = document.querySelector("#c4 .toggle__btn.is-active[data-sector]");
  wireToggle({
    buttons: document.querySelectorAll("#c4 .toggle__btn[data-sector]"),
    panels: document.querySelectorAll("#c4 .sector-panel"),
    key: "sector",
    matches: (p, sector) => p.id === `c4_sector_${sector}`,
    initial: active ? active.dataset.sector : "basic_materials",
    onShow: setInterpretation,
  });
})();

// ---------- Chapter 5: sector toggle ----------
wireToggle({
  buttons: document.querySelectorAll("#c5_right .toggle__btn[data-sector]"),
  panels: document.querySelectorAll("#c5_right [data-sector-panel]"),
  key: "sector",
  matches: (p, sector) => p.dataset.sectorPanel === sector,
  initial: "technology",
});

// ---------- Chapter 6: figure toggle ----------
wireToggle({
  buttons: document.querySelectorAll("#c6_right .toggle__btn[data-c6]"),
  panels: document.querySelectorAll("#c6_right .c6-panel"),
  key: "c6",
  matches: (p, name) => p.id === `c6_${name}`,
  initial: "performance",
});
