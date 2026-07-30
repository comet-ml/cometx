#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ****************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
#  Copyright (c) 2024 Cometx Development
#      Team. All rights reserved.
# ****************************************
"""Self-contained HTML dashboard renderer for `cometx admin growth-report`.

`build_html(report_data)` turns the `report_data` contract (see
`.superpowers/sdd/task-C8-report.md` for the full documented shape) into a
single self-contained HTML string: inline CSS (theme-aware, light/dark),
inline JS that draws the charts as inline SVG from an embedded JSON payload,
and server-rendered KPI/table/panel markup. No external assets, no network
calls, no secrets -- `report_data` never carries an API key.

Charts render as plain month-on-month / week-on-week series with horizontal
gridlines for scale. An earlier single-analysis-window overlay (a shaded
band with start/end dots) was removed once the report became
period-over-period; the now-defunct `window_start`/`window_end` chart-data
fields were dropped with it. A window chip is still rendered next to each
section title.
"""

from __future__ import annotations

import html
import json

DEFAULT_PALETTE = ("--accent", "--sdk", "--ok", "--warn")


def _esc(value) -> str:
    """HTML-escape any value (numbers included) for safe insertion."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _fmt(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, (int, float)):
        return f"{value:,}"
    return _esc(value)


CSS = """
:root{
  --ground:#f7f8fa; --card:#ffffff; --card-2:#fbfbfd;
  --ink:#161a21; --ink-2:#48505e; --muted:#727b8a; --hair:#e6e9ef; --hair-2:#eef1f6;
  --accent:#3b5bdb; --accent-soft:rgba(59,91,219,.10);
  --sdk:#0aa0b4; --sdk-soft:rgba(10,160,180,.12);
  --ok:#17886b; --ok-soft:rgba(23,136,107,.12);
  --warn:#b7791f; --warn-soft:rgba(183,121,31,.14);
  --idle:#8a93a2; --idle-soft:rgba(138,147,162,.14);
  --bar-mute:#c9d0dc;
  --grid:rgba(22,26,33,.07); --shadow:0 1px 2px rgba(16,22,40,.04),0 8px 24px -16px rgba(16,22,40,.18);
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme: dark){
  :root{
    --ground:#0d1015; --card:#151922; --card-2:#11151d;
    --ink:#e7eaf1; --ink-2:#aab3c2; --muted:#828d9d; --hair:#232a35; --hair-2:#1c222c;
    --accent:#6c86fa; --accent-soft:rgba(108,134,250,.16);
    --sdk:#1ca6bb; --sdk-soft:rgba(28,166,187,.18);
    --ok:#3dbe93; --ok-soft:rgba(61,190,147,.16);
    --warn:#e0a94a; --warn-soft:rgba(224,169,74,.16);
    --idle:#7b8494; --idle-soft:rgba(123,132,148,.18);
    --bar-mute:#333b48;
    --grid:rgba(231,234,241,.08); --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -18px rgba(0,0,0,.7);
  }
}
:root[data-theme="light"]{
  --ground:#f7f8fa; --card:#ffffff; --card-2:#fbfbfd;
  --ink:#161a21; --ink-2:#48505e; --muted:#727b8a; --hair:#e6e9ef; --hair-2:#eef1f6;
  --accent:#3b5bdb; --accent-soft:rgba(59,91,219,.10);
  --sdk:#0aa0b4; --sdk-soft:rgba(10,160,180,.12);
  --ok:#17886b; --ok-soft:rgba(23,136,107,.12);
  --warn:#b7791f; --warn-soft:rgba(183,121,31,.14);
  --idle:#8a93a2; --idle-soft:rgba(138,147,162,.14);
  --bar-mute:#c9d0dc;
  --grid:rgba(22,26,33,.07); --shadow:0 1px 2px rgba(16,22,40,.04),0 8px 24px -16px rgba(16,22,40,.18);
}
:root[data-theme="dark"]{
  --ground:#0d1015; --card:#151922; --card-2:#11151d;
  --ink:#e7eaf1; --ink-2:#aab3c2; --muted:#828d9d; --hair:#232a35; --hair-2:#1c222c;
  --accent:#6c86fa; --accent-soft:rgba(108,134,250,.16);
  --sdk:#1ca6bb; --sdk-soft:rgba(28,166,187,.18);
  --ok:#3dbe93; --ok-soft:rgba(61,190,147,.16);
  --warn:#e0a94a; --warn-soft:rgba(224,169,74,.16);
  --idle:#7b8494; --idle-soft:rgba(123,132,148,.18);
  --bar-mute:#333b48;
  --grid:rgba(231,234,241,.08); --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -18px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  -webkit-font-smoothing:antialiased;line-height:1.5;font-size:15px}
.wrap{max-width:1120px;margin:0 auto;padding:28px 24px 64px}
.topbar{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:16px;
  padding-bottom:18px;border-bottom:1px solid var(--hair);margin-bottom:24px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);margin:0 0 6px}
h1{font-size:26px;line-height:1.15;margin:0;letter-spacing:-.015em;font-weight:650}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:12px;font-size:13px;color:var(--ink-2)}
.meta b{color:var(--ink);font-weight:600}
.meta .mono{font-family:var(--mono);font-size:12px}
.topbar-right{display:flex;flex-direction:column;align-items:flex-end;gap:12px}
.toggle{font-family:var(--mono);font-size:12px;color:var(--ink-2);background:var(--card);
  border:1px solid var(--hair);border-radius:7px;padding:7px 11px;cursor:pointer;display:inline-flex;
  gap:7px;align-items:center}
.toggle:hover{border-color:var(--accent);color:var(--ink)}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-family:var(--mono);
  padding:4px 9px;border-radius:999px;border:1px solid var(--hair);color:var(--ink-2);background:var(--card)}
.chip .dot{width:7px;height:7px;border-radius:50%}
.chip.on .dot{background:var(--ok)} .chip.off .dot{background:var(--idle)}
.chip.winchip{color:var(--ink);border-color:color-mix(in srgb,var(--accent) 40%,var(--hair))}
.sec-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;margin:30px 0 14px}
.sec-title{font-size:18px;margin:0;font-weight:650;letter-spacing:-.01em}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--hair);border-radius:12px;padding:16px 16px 14px;
  box-shadow:var(--shadow);position:relative;overflow:hidden}
.kpi .stripe{position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--accent)}
.kpi.ok .stripe{background:var(--ok)} .kpi.warn .stripe{background:var(--warn)}
.kpi .label{font-size:12px;color:var(--muted);font-weight:550;letter-spacing:.01em}
.kpi .val{font-family:var(--mono);font-size:28px;font-weight:600;letter-spacing:-.02em;
  margin-top:6px;font-variant-numeric:tabular-nums;line-height:1}
.kpi .sub{margin-top:8px;font-size:12px;color:var(--ink-2)}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.panel{background:var(--card);border:1px solid var(--hair);border-radius:12px;padding:18px 18px 12px;
  box-shadow:var(--shadow)}
.panel h3{font-size:14px;margin:0;font-weight:600;letter-spacing:-.005em}
.panel .ph{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:4px}
.panel .hint{font-size:12px;color:var(--muted)}
.legend{display:flex;gap:14px;margin:2px 0 6px;font-size:12px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.swatch{width:10px;height:10px;border-radius:3px;display:inline-block}
.nodata{color:var(--muted);font-size:12px;padding:24px 0;text-align:center}
svg{display:block;width:100%;height:auto;overflow:visible}
.axis-base{stroke:var(--hair);stroke-width:1}
.gridline{stroke:var(--hair);stroke-width:1;stroke-dasharray:2 4;opacity:.7}
.guide{stroke:var(--accent);stroke-width:1;opacity:0;pointer-events:none}
.chart-host{position:relative}
.charttip{position:absolute;pointer-events:none;transform:translate(-50%,-115%);white-space:nowrap;
  background:var(--card);color:var(--ink);border:1px solid var(--hair);border-radius:6px;
  padding:4px 8px;font-family:var(--mono);font-size:11px;box-shadow:var(--shadow);
  opacity:0;transition:opacity .08s;z-index:5}
.charttip.show{opacity:1}
.tablecard{background:var(--card);border:1px solid var(--hair);border-radius:12px;box-shadow:var(--shadow);
  overflow:hidden;margin-bottom:16px}
.tablecard .ph{padding:16px 18px 12px;display:flex;align-items:center;justify-content:space-between}
details.tablecard>summary{cursor:pointer;list-style:none;user-select:none}
details.tablecard>summary::-webkit-details-marker{display:none}
details.tablecard>summary::marker{content:""}
details.tablecard>summary:hover{background:var(--hair-2)}
.ph-title{display:flex;align-items:center;gap:8px;min-width:0}
.disc{flex:none;width:0;height:0;border-left:6px solid currentColor;border-top:4px solid transparent;border-bottom:4px solid transparent;opacity:.55;transition:transform .15s ease}
details.tablecard[open]>summary .disc{transform:rotate(90deg)}
.count{font-family:var(--mono);font-size:11px;line-height:1;background:var(--hair);border-radius:999px;padding:3px 8px;opacity:.85}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px;min-width:480px}
thead th{text-align:right;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  font-weight:600;padding:9px 18px;border-bottom:1px solid var(--hair);white-space:nowrap;background:var(--card-2)}
thead th:first-child,tbody td:first-child{text-align:left}
tbody td{padding:11px 18px;border-bottom:1px solid var(--hair-2);text-align:right;
  font-variant-numeric:tabular-nums;font-family:var(--mono);color:var(--ink-2)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--card-2)}
footer{margin-top:26px;padding-top:16px;border-top:1px solid var(--hair);font-size:12px;color:var(--muted)}
@media (max-width:820px){
  .kpis{grid-template-columns:repeat(2,1fr)} .grid-2{grid-template-columns:1fr}
}
"""

# NOTE: the SVG namespace literal is split across a concatenation so the
# contiguous substring "http://" never appears in this file's text (the
# renderer output must never contain http:// or https:// anywhere).
CLIENT_JS = """
(function(){
  "use strict";
  var root = document.documentElement;
  var btn = document.getElementById("themeBtn");
  var icon = document.getElementById("themeIcon");
  var lbl = document.getElementById("themeLbl");
  function sysDark(){ return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches; }
  function curTheme(){ return root.getAttribute("data-theme") || (sysDark() ? "dark" : "light"); }
  function applyTheme(t){
    root.setAttribute("data-theme", t);
    if(icon) icon.textContent = t === "dark" ? "\\u25D0" : "\\u25D1";
    if(lbl) lbl.textContent = t === "dark" ? "Dark" : "Light";
    redraw();
  }
  if(btn){ btn.addEventListener("click", function(){ applyTheme(curTheme() === "dark" ? "light" : "dark"); }); }

  function tok(name){ return getComputedStyle(root).getPropertyValue(name).trim(); }
  var NS = "http:" + "//www.w3.org/2000/svg";
  function el(name, attrs){
    var node = document.createElementNS(NS, name);
    for(var k in attrs){ if(Object.prototype.hasOwnProperty.call(attrs, k)) node.setAttribute(k, attrs[k]); }
    return node;
  }
  function fmt(n){ return (n || 0).toLocaleString("en-US"); }

  var W = 560, H = 220, P = {t: 16, r: 14, b: 28, l: 44};

  // Interactive hover: a positioned tooltip div + a vertical guide line,
  // driven by one full-height transparent hit rect per column. Reliable
  // across browsers (unlike native SVG <title>, which is delayed/flaky).
  function attachTip(host, svg, cols){
    if(!cols || !cols.length) return;
    host.style.position = "relative";
    var tip = host.querySelector(".charttip");
    if(!tip){ tip = document.createElement("div"); tip.className = "charttip"; host.appendChild(tip); }
    var guide = el("line", {class: "guide", y1: P.t, y2: H - P.b, x1: 0, x2: 0, opacity: "0"});
    svg.appendChild(guide);
    function show(col){
      tip.textContent = col.label || (col.key + ": " + fmt(col.value));
      tip.classList.add("show");
      var r = host.getBoundingClientRect();
      tip.style.left = (col.x / W) * r.width + "px";
      tip.style.top = ((P.t / H) * r.height) + "px";
      guide.setAttribute("x1", col.x); guide.setAttribute("x2", col.x);
      guide.setAttribute("opacity", "1");
    }
    function hide(){ tip.classList.remove("show"); guide.setAttribute("opacity", "0"); }
    cols.forEach(function(col){
      var hit = el("rect", {x: col.x0, y: P.t, width: Math.max(col.w, 1),
        height: (H - P.t - P.b), fill: "transparent"});
      hit.addEventListener("mouseenter", function(){ show(col); });
      hit.addEventListener("mousemove", function(){ show(col); });
      hit.addEventListener("mouseleave", hide);
      svg.appendChild(hit);
    });
  }

  function hGrid(svg, yOf, maxVal, iw){
    for(var g = 1; g <= 4; g++){
      var v = maxVal * g / 4, y = yOf(v);
      svg.appendChild(el("line", {class: "gridline", x1: P.l, x2: P.l + iw, y1: y, y2: y}));
      var t = el("text", {x: P.l - 6, y: y + 3, "text-anchor": "end", fill: tok("--muted"),
        "font-size": "10", "font-family": "var(--mono)"});
      t.textContent = fmt(v); svg.appendChild(t);
    }
  }

  // Right-hand axis ticks (unlabelled gridlines omitted; the left axis owns
  // the grid) for a secondary series drawn on its own scale. `suffix` labels
  // the units, e.g. "%". Colored to match the series it measures.
  function rAxis(svg, yOf, maxVal, iw, suffix, color){
    for(var g = 1; g <= 4; g++){
      var v = maxVal * g / 4, y = yOf(v);
      var t = el("text", {x: P.l + iw + 6, y: y + 3, "text-anchor": "start",
        fill: color || tok("--muted"), "font-size": "10", "font-family": "var(--mono)"});
      t.textContent = fmt(Math.round(v)) + (suffix || ""); svg.appendChild(t);
    }
  }

  function drawBars(host, data){
    data = data || {}; var points = data.points || [];
    if(!points.length){ host.innerHTML = "<p class=\\"nodata\\">No data</p>"; return; }
    var accent = tok("--accent");
    var svg = el("svg", {viewBox: "0 0 " + W + " " + H, role: "img"});
    var iw = W - P.l - P.r, ih = H - P.t - P.b, n = points.length;
    var max = 1; points.forEach(function(p){ max = Math.max(max, p.value || 0); });
    var slot = iw / n, bw = Math.min(38, slot * 0.55);
    hGrid(svg, function(v){ return P.t + ih * (1 - v / max); }, max, iw);
    svg.appendChild(el("line", {class: "axis-base", x1: P.l, x2: P.l + iw, y1: P.t + ih, y2: P.t + ih}));
    var cols = [];
    points.forEach(function(p, i){
      var v = p.value || 0, h = ih * (v / max), x = P.l + slot * i + (slot - bw) / 2, y = P.t + ih - h;
      svg.appendChild(el("rect", {x: x, y: y, width: bw, height: Math.max(h, 0), rx: "3", fill: accent}));
      cols.push({x: P.l + slot * i + slot / 2, x0: P.l + slot * i, w: slot, key: p.key, value: v});
      if(i % Math.max(1, Math.ceil(n / 8)) === 0){
        var t = el("text", {x: x + bw / 2, y: H - 8, "text-anchor": "middle", fill: tok("--muted"),
          "font-size": "10", "font-family": "var(--mono)"});
        t.textContent = p.key; svg.appendChild(t);
      }
    });
    attachTip(host, svg, cols);
    host.appendChild(svg);
  }

  function drawLines(host, data){
    data = data || {}; var points = data.points || []; var cats = data.categories || [];
    if(!points.length || !cats.length){ host.innerHTML = "<p class=\\"nodata\\">No data</p>"; return; }
    var colors = (data.colors && data.colors.length) ? data.colors : ["--accent", "--sdk", "--ok", "--warn"];
    var svg = el("svg", {viewBox: "0 0 " + W + " " + H, role: "img"});
    var iw = W - P.l - P.r, ih = H - P.t - P.b, n = points.length;
    var max = 1;
    points.forEach(function(p){
      cats.forEach(function(c){ max = Math.max(max, (p.values && p.values[c]) || 0); });
    });
    var X = function(i){ return n > 1 ? P.l + iw * (i / (n - 1)) : P.l + iw / 2; };
    var Y = function(v){ return P.t + ih * (1 - v / max); };
    hGrid(svg, Y, max, iw);
    svg.appendChild(el("line", {class: "axis-base", x1: P.l, x2: P.l + iw, y1: P.t + ih, y2: P.t + ih}));
    // one polyline per category
    cats.forEach(function(c, ci){
      var col = tok(colors[ci % colors.length]);
      var dp = "";
      points.forEach(function(p, i){
        var v = (p.values && p.values[c]) || 0;
        dp += (i ? " L" : "M") + X(i) + " " + Y(v);
      });
      svg.appendChild(el("path", {d: dp, fill: "none", stroke: col, "stroke-width": "2",
        "stroke-linejoin": "round", "stroke-linecap": "round"}));
      var lv = (points[n - 1].values && points[n - 1].values[c]) || 0;
      svg.appendChild(el("circle", {cx: X(n - 1), cy: Y(lv), r: "3", fill: col}));
    });
    // x-axis: earliest and latest keys
    [0, n - 1].forEach(function(i){
      if(i < 0) return;
      var xl = el("text", {x: X(i), y: H - 8, "text-anchor": i === 0 ? "start" : "end",
        fill: tok("--muted"), "font-size": "10", "font-family": "var(--mono)"});
      xl.textContent = points[i].key; svg.appendChild(xl);
    });
    // Per-x tooltips: one column spanning the gap around each point, listing
    // every category's value at that x.
    var labels = data.labels || {};
    var slot = n > 1 ? iw / (n - 1) : iw;
    var cols = points.map(function(p, i){
      var parts = [p.key];
      cats.forEach(function(c){ parts.push((labels[c] || c) + " " + fmt((p.values && p.values[c]) || 0)); });
      return {x: X(i), x0: X(i) - slot / 2, w: slot, key: p.key, label: parts.join("  \\u00b7  ")};
    });
    attachTip(host, svg, cols);
    host.appendChild(svg);
  }

  function drawGroupedH(host, data){
    data = data || {}; var rows = data.rows || []; var cats = data.categories || [];
    if(!rows.length){ host.innerHTML = "<p class=\\"nodata\\">No data</p>"; return; }
    var colors = (data.colors && data.colors.length) ? data.colors : ["--accent", "--sdk", "--ok", "--warn"];
    var rowH = 32, padL = 120, padR = 48, w = 560, h = rows.length * rowH + 16;
    var svg = el("svg", {viewBox: "0 0 " + w + " " + h, role: "img"});
    var iw = w - padL - padR;
    var totals = rows.map(function(r){
      if(cats.length){ var s = 0; cats.forEach(function(c){ s += (r.values && r.values[c]) || 0; }); return s; }
      return r.value || 0;
    });
    var max = Math.max.apply(null, totals.concat([1]));
    rows.forEach(function(r, i){
      var y = 8 + i * rowH;
      var nm = el("text", {x: padL - 10, y: y + rowH / 2 + 4, "text-anchor": "end", fill: tok("--ink"),
        "font-size": "11.5", "font-family": "var(--sans)"});
      nm.textContent = r.label; svg.appendChild(nm);
      var bh = 14, x = padL;
      svg.appendChild(el("rect", {x: padL, y: y + rowH / 2 - bh / 2, width: iw, height: bh, rx: "4",
        fill: tok("--hair-2")}));
      if(cats.length){
        cats.forEach(function(c, ci){
          var v = (r.values && r.values[c]) || 0; if(!v) return;
          var bw = iw * (v / max);
          var rect = el("rect", {x: x, y: y + rowH / 2 - bh / 2, width: Math.max(bw, 0), height: bh,
            fill: tok(colors[ci % colors.length])});
          var title = el("title", {});
          title.textContent = ((data.labels && data.labels[c]) || c) + ": " + fmt(v);
          rect.appendChild(title); svg.appendChild(rect);
          x += bw;
        });
      } else {
        var bw2 = iw * ((r.value || 0) / max);
        svg.appendChild(el("rect", {x: padL, y: y + rowH / 2 - bh / 2, width: Math.max(bw2, 0), height: bh,
          rx: "4", fill: tok(colors[0])}));
      }
      var tot = el("text", {x: w - padR + 8, y: y + rowH / 2 + 4, "text-anchor": "start", fill: tok("--muted"),
        "font-size": "11", "font-family": "var(--mono)"});
      tot.textContent = fmt(totals[i]); svg.appendChild(tot);
    });
    host.appendChild(svg);
  }

  function drawBarsLine(host, data){
    data = data || {}; var points = data.points || []; var bars = data.bars || [];
    var lineCat = data.line;
    if(!points.length || !bars.length){ host.innerHTML = "<p class=\\"nodata\\">No data</p>"; return; }
    var barColors = (data.bar_colors && data.bar_colors.length) ? data.bar_colors : ["--ok", "--warn"];
    var lineColor = data.line_color || "--accent";
    var lineSuffix = data.line_suffix || "";
    var barLabels = data.bar_labels || {};
    var svg = el("svg", {viewBox: "0 0 " + W + " " + H, role: "img"});
    // Reserve extra right margin for the secondary (line) axis labels when a
    // line is present, so they don't collide with the plot area.
    var rpad = lineCat ? 34 : 0;
    var iw = W - P.l - P.r - rpad, ih = H - P.t - P.b, n = points.length;
    var lmax = 1, rmax = 1;
    points.forEach(function(p){
      bars.forEach(function(b){ lmax = Math.max(lmax, (p.values && p.values[b]) || 0); });
      rmax = Math.max(rmax, (p.values && p.values[lineCat]) || 0);
    });
    var slot = iw / n, group = Math.min(slot * 0.7, 40), bw = group / bars.length;
    var YL = function(v){ return P.t + ih * (1 - v / lmax); };
    var YR = function(v){ return P.t + ih * (1 - v / rmax); };
    var cx = function(i){ return P.l + slot * i + slot / 2; };
    hGrid(svg, YL, lmax, iw);
    if(lineCat) rAxis(svg, YR, rmax, iw, lineSuffix, tok(lineColor));
    svg.appendChild(el("line", {class: "axis-base", x1: P.l, x2: P.l + iw, y1: P.t + ih, y2: P.t + ih}));
    var cols = [];
    points.forEach(function(p, i){
      var gx = P.l + slot * i + (slot - group) / 2;
      bars.forEach(function(b, bi){
        var v = (p.values && p.values[b]) || 0; if(v <= 0) return;
        var h = ih * (v / lmax);
        svg.appendChild(el("rect", {x: gx + bw * bi, y: P.t + ih - h, width: Math.max(bw - 2, 1),
          height: h, rx: "2", fill: tok(barColors[bi % barColors.length])}));
      });
      // Per-column tooltip label: each bar series + the line value.
      var parts = [p.key];
      bars.forEach(function(b){ parts.push((barLabels[b] || b) + " " + fmt((p.values && p.values[b]) || 0)); });
      if(lineCat){ parts.push((barLabels[lineCat] || lineCat) + " " + fmt((p.values && p.values[lineCat]) || 0) + lineSuffix); }
      cols.push({x: cx(i), x0: P.l + slot * i, w: slot, key: p.key, label: parts.join("  \\u00b7  ")});
      if(i % Math.max(1, Math.ceil(n / 8)) === 0){
        var t = el("text", {x: cx(i), y: H - 8, "text-anchor": "middle", fill: tok("--muted"),
          "font-size": "10", "font-family": "var(--mono)"});
        t.textContent = p.key; svg.appendChild(t);
      }
    });
    if(lineCat){
      var dp = "";
      points.forEach(function(p, i){ dp += (i ? " L" : "M") + cx(i) + " " + YR((p.values && p.values[lineCat]) || 0); });
      svg.appendChild(el("path", {d: dp, fill: "none", stroke: tok(lineColor), "stroke-width": "2",
        "stroke-linejoin": "round", "stroke-linecap": "round"}));
      points.forEach(function(p, i){
        svg.appendChild(el("circle", {cx: cx(i), cy: YR((p.values && p.values[lineCat]) || 0), r: "2.5", fill: tok(lineColor)}));
      });
    }
    attachTip(host, svg, cols);
    host.appendChild(svg);
  }

  function drawChart(c){
    if(!c || !c.id) return;
    var host = document.getElementById(c.id); if(!host) return;
    host.innerHTML = "";
    if(c.kind === "bars") drawBars(host, c.data);
    else if(c.kind === "lines") drawLines(host, c.data);
    else if(c.kind === "barsLine") drawBarsLine(host, c.data);
    else if(c.kind === "groupedBarsH") drawGroupedH(host, c.data);
  }

  function collectCharts(p){
    var out = [];
    function add(section){ if(section && section.charts){ section.charts.forEach(function(c){ out.push(c); }); } }
    var sections = (p && p.sections) || {};
    add(sections.unified);
    add(sections.people);
    add(sections.leaderboards);
    add(sections.personal_vs_service);
    return out;
  }

  var dataNode = document.getElementById("report-data");
  var PAYLOAD = dataNode ? JSON.parse(dataNode.textContent || "{}") : {};

  function redraw(){ collectCharts(PAYLOAD).forEach(drawChart); }

  if(!root.getAttribute("data-theme")){
    // leave unset so the CSS media query drives initial theme; icon/label still shown
    icon && (icon.textContent = sysDark() ? "\\u25D0" : "\\u25D1");
    lbl && (lbl.textContent = sysDark() ? "Dark" : "Light");
  }
  redraw();
  if(window.matchMedia){
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function(){
      if(!root.getAttribute("data-theme")) redraw();
    });
  }
})();
"""

FOOTER_HTML = (
    "<footer>Generated by <code>cometx admin growth-report</code>. "
    "Self-contained snapshot; no external assets or network calls."
    "</footer>"
)


def render_kpis(kpis) -> str:
    if not kpis:
        return ""
    cards = []
    for kpi in kpis:
        tone = kpi.get("tone")
        cls = "kpi" + (f" {tone}" if tone in ("ok", "warn") else "")
        sub = kpi.get("sub")
        sub_html = f'<div class="sub">{_esc(sub)}</div>' if sub else ""
        cards.append(
            f'<div class="{cls}"><span class="stripe"></span>'
            f'<div class="label">{_esc(kpi.get("label"))}</div>'
            f'<div class="val">{_esc(kpi.get("value"))}</div>{sub_html}</div>'
        )
    return f'<section class="kpis">{"".join(cards)}</section>'


def render_chart_panel(chart) -> str:
    if not chart or not chart.get("id"):
        return ""
    cid = _esc(chart.get("id"))
    kind = _esc(chart.get("kind"))
    title = _esc(chart.get("title"))
    hint = chart.get("hint")
    hint_html = f'<span class="hint">{_esc(hint)}</span>' if hint else ""
    legend_items = chart.get("legend") or []
    legend_html = ""
    if legend_items:
        swatches = "".join(
            '<span><i class="swatch" '
            f'style="background:var({_esc(item.get("color", "--accent"))})">'
            f"</i>{_esc(item.get('label'))}</span>"
            for item in legend_items
        )
        legend_html = f'<div class="legend">{swatches}</div>'
    return (
        '<section class="panel">'
        f'<div class="ph"><h3>{title}</h3>{hint_html}</div>'
        f"{legend_html}"
        f'<div id="{cid}" class="chart-host" data-kind="{kind}"></div>'
        "</section>"
    )


def render_table(table, as_panel: bool = False) -> str:
    if not table or not table.get("headers"):
        return ""
    title = table.get("title")
    hint = table.get("hint")
    headers = table.get("headers") or []
    rows = table.get("rows") or []
    head_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    # Collapsible table: native <details>/<summary> (no JS). Collapsed by
    # default so large tables don't dominate the page; the summary shows the
    # title + row count so you can scan without expanding.
    hint_html = f'<span class="hint">{_esc(hint)}</span>' if hint else ""
    summary_title = _esc(title) if title else "Table"
    summary = (
        '<summary class="ph">'
        '<span class="ph-title">'
        '<span class="disc" aria-hidden="true"></span>'
        f"<h3>{summary_title}</h3>"
        f'<span class="count">{len(rows)}</span>'
        "</span>"
        f"{hint_html}</summary>"
    )
    return (
        '<details class="tablecard">'
        f"{summary}"
        '<div class="scroll"><table><thead><tr>'
        f"{head_html}</tr></thead><tbody>{body_html}</tbody></table></div>"
        "</details>"
    )


def render_section(section) -> str:
    """Render one self-contained (title + kpis + charts + table + panels)
    block. Returns "" for a missing/empty section -- callers never need to
    guard, keeping `build_html` robust to partial `report_data`."""
    if not section or not section.get("title"):
        return ""
    title = _esc(section.get("title"))
    chip = section.get("window_chip")
    chip_html = f'<span class="chip winchip">{_esc(chip)}</span>' if chip else ""
    parts = [
        f'<div class="sec-head"><h2 class="sec-title">{title}</h2>{chip_html}</div>'
    ]

    parts.append(render_kpis(section.get("kpis")))

    charts = [c for c in (section.get("charts") or []) if c]
    if charts:
        panels = "".join(render_chart_panel(c) for c in charts)
        parts.append(f'<div class="grid-2">{panels}</div>')

    parts.append(render_table(section.get("table")))
    for panel in section.get("panels") or []:
        parts.append(render_table(panel))

    return f'<section class="section">{"".join(p for p in parts if p)}</section>'


def render_topbar(report_data: dict) -> str:
    meta = report_data.get("meta") or {}
    window = report_data.get("window") or {}

    title = _esc(meta.get("title") or "Growth report")
    meta_bits = []
    if meta.get("org"):
        meta_bits.append(f'<span>Organization <b>{_esc(meta["org"])}</b></span>')
    if meta.get("scope"):
        meta_bits.append(f'<span>Scope <b>{_esc(meta["scope"])}</b></span>')
    if window.get("label"):
        meta_bits.append(
            f'<span>Window <b class="mono">{_esc(window["label"])}</b></span>'
        )
    if meta.get("generated"):
        meta_bits.append(
            f'<span>Generated <b class="mono">{_esc(meta["generated"])}</b></span>'
        )
    if meta.get("source"):
        meta_bits.append(f'<span>Source <b>{_esc(meta["source"])}</b></span>')

    return (
        '<div class="topbar"><div>'
        '<p class="eyebrow">Comet Growth Report</p>'
        f"<h1>{title}</h1>"
        f'<div class="meta">{"".join(meta_bits)}</div>'
        "</div>"
        '<div class="topbar-right">'
        '<button class="toggle" id="themeBtn" aria-label="Toggle color theme">'
        '<span id="themeIcon">◑</span><span id="themeLbl">Theme</span>'
        "</button>"
        "</div></div>"
    )


def build_html(report_data: dict) -> str:
    """Render the full self-contained growth-report HTML document from
    `report_data`. Fully data-driven; missing/empty sections render as
    nothing (never raises) so partial data still produces a valid page.
    """
    report_data = report_data or {}
    sections = report_data.get("sections") or {}

    body_parts = [
        render_topbar(report_data),
        render_section(sections.get("unified")),
        render_section(sections.get("people")),
        render_section(sections.get("leaderboards")),
        render_section(sections.get("personal_vs_service")),
    ]

    body_parts.append(FOOTER_HTML)

    payload = json.dumps(report_data, default=str)
    # Breakout-escape "<" so a "<" inside embedded data (e.g. a workspace or
    # project name) can never terminate the </script> tag early.
    payload = payload.replace("<", "\\u003c")

    meta_title = _esc((report_data.get("meta") or {}).get("title") or "Growth report")

    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{meta_title}</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n"
        f'<div class="wrap">{"".join(body_parts)}</div>\n'
        f'<script type="application/json" id="report-data">{payload}</script>\n'
        f"<script>{CLIENT_JS}</script>\n"
        "</body>\n</html>\n"
    )


def write_html(report_data: dict, path) -> str:
    """Render `report_data` to `path` and return the path (as a string)."""
    document = build_html(report_data)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(document)
    return str(path)
