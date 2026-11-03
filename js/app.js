/*
 * app.js — HW2 多元線性回歸網頁介面（純 JavaScript）
 * 資料載入、控制項、表格與 Chart.js 圖表
 */
(function () {
  "use strict";
  const S = window.Stats, A = window.Analysis;
  const $ = (id) => document.getElementById(id);
  const BLUE = "#2E6FB7", ORANGE = "#E8833A", TEAL = "#087f83", RED = "#d23c3c";
  const METHOD_COLORS = { SelectKBest: "#2E6FB7", RFE: "#0f9d8a", LassoCV: "#E8833A" };

  const PROBLEMS = {
    flood: {
      file: "data/flood_train_sample.csv.gz", target: "FloodProbability",
      name: "Regression with a Flood Prediction Dataset（Kaggle Playground Series S4E5）",
      link: "https://www.kaggle.com/competitions/playground-series-s4e5",
      note: "原始 train.csv 共 1,117,957 筆；網頁使用 random_state=42 隨機抽樣的 50,000 筆，完整資料的分析結果見 Notebook 與報告。",
      business: [
        "<b>問題</b>：根據 20 項環境、氣候與人為風險因子（季風強度、排水系統、都市化、森林砍伐等，每項為 0～18 的整數分數），預測一個地區的洪水發生機率 <code>FloodProbability</code>。",
        "<b>價值</b>：協助政府與防災單位找出高風險區域與關鍵風險因子，作為防災資源配置、排水與河川治理優先順序、基礎設施改善的分析參考。",
        "<b>成功標準</b>：Kaggle 以 R² 評分；本研究目標是建立可解釋的多元線性回歸模型，R² 達 0.84 以上。",
        "<b>限制</b>：資料為 Kaggle 合成資料，係數與風險排序僅供方法研究，不能直接作為真實洪水預警依據。",
      ],
    },
    startups: {
      file: "data/50_Startups.csv", target: "Profit",
      name: "50 Startups（100-Days-Of-ML-Code）",
      link: "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv",
      note: "共 50 家新創公司，特徵為研發支出、行政支出、行銷支出與所在州別（類別變數，自動做 One-hot 編碼）。",
      business: [
        "<b>問題</b>：根據新創公司的研發支出（R&amp;D Spend）、行政支出（Administration）、行銷支出（Marketing Spend）與所在州別（State），預測公司年度利潤 <code>Profit</code>。",
        "<b>價值</b>：協助創投評估投資標的的獲利潛力，也幫助經營者了解哪一類支出與利潤最相關，作為預算分配參考。",
        "<b>成功標準</b>：測試集 R² 越高越好，並能指出影響利潤的關鍵支出項目。",
        "<b>限制</b>：樣本只有 50 筆，交叉驗證與測試集結果的變異較大，結論需謹慎解讀。",
      ],
    },
  };

  const state = { raw: null, cfg: null, prep: null, fsCache: {}, res: null, charts: {}, dirty: new Set(), activeTab: "t1", dataId: 0 };

  // ------------------------------------------------------------------
  // 工具
  // ------------------------------------------------------------------
  function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  function fmt(x, d = 4) {
    if (x === null || x === undefined || Number.isNaN(x)) return "—";
    if (!isFinite(x)) return x > 0 ? "∞" : "−∞";
    const a = Math.abs(x);
    if (a >= 1e4) return Math.round(x).toLocaleString("en-US");
    if (a >= 100) return x.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (a !== 0 && a < 1e-4) return x.toExponential(3);
    return x.toFixed(d);
  }
  function fmtP(p) { return p < 1e-4 ? (p === 0 ? "< 1e-300" : p.toExponential(2)) : p.toFixed(4); }
  function setStatus(msg, kind) { const el = $("status"); el.textContent = msg; el.className = "status" + (kind ? " " + kind : ""); }
  function kpiRow(el, items) {
    el.innerHTML = items.map(([l, v]) => `<div class="kpi"><div class="lab">${esc(l)}</div><div class="val">${esc(v)}</div></div>`).join("");
  }
  function table(headers, rows, opts = {}) {
    const numCols = opts.num || [];
    const head = "<tr>" + headers.map((h) => `<th>${esc(h)}</th>`).join("") + "</tr>";
    const body = rows.map((r, i) => `<tr class="${opts.highlight && opts.highlight(i) ? "rec" : ""}">` +
      r.map((v, j) => `<td class="${numCols.includes(j) ? "num" : ""}${opts.wrap && opts.wrap.includes(j) ? " wrap" : ""}">${opts.raw && opts.raw.includes(j) ? v : esc(v)}</td>`).join("") + "</tr>").join("");
    return `<table>${head}${body}</table>`;
  }
  function sampleIdx(n, k, seed) {
    const perm = S.permutation(n, seed);
    return Array.from(perm.slice(0, Math.min(k, n)));
  }
  function chart(id, config) {
    if (state.charts[id]) state.charts[id].destroy();
    config.options = Object.assign({ responsive: true, maintainAspectRatio: false, animation: false, plugins: {} }, config.options || {});
    state.charts[id] = new Chart($(id), config);
  }
  function axis(title, extra) { return Object.assign({ title: { display: !!title, text: title }, grid: { color: "#edf1f4" } }, extra || {}); }

  // ------------------------------------------------------------------
  // 資料載入
  // ------------------------------------------------------------------
  function toRawUrl(url) {
    const m = url.trim().match(/^https?:\/\/github\.com\/([^/]+)\/([^/]+)\/blob\/(.+)$/);
    return m ? `https://raw.githubusercontent.com/${m[1]}/${m[2]}/${m[3]}` : url.trim();
  }

  async function fetchText(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    if (/\.gz($|\?)/.test(url)) {
      const buf = new Uint8Array(await r.arrayBuffer());
      if (buf[0] === 0x1f && buf[1] === 0x8b) {
        const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
        return await new Response(stream).text();
      }
      return new TextDecoder().decode(buf);
    }
    return await r.text();
  }

  function parseCsv(text) {
    const res = Papa.parse(text.replace(/^﻿/, ""), { header: true, dynamicTyping: true, skipEmptyLines: "greedy", transformHeader: (h) => h.trim() });
    const headers = (res.meta.fields || []).filter((h) => h !== "");
    if (!headers.length || !res.data.length) throw new Error("CSV 沒有資料");
    return { rows: res.data, headers };
  }

  async function loadProblem(key) {
    $("customBox").classList.toggle("hidden", key !== "custom");
    if (key === "custom") return loadUrl();
    const cfg = PROBLEMS[key];
    setStatus("讀取資料中…", "busy");
    try {
      const t0 = performance.now();
      const { rows, headers } = parseCsv(await fetchText(cfg.file));
      setData(rows, headers, { ...cfg, key }, performance.now() - t0);
    } catch (e) { setStatus("資料載入失敗：" + e.message, "bad"); }
  }

  async function loadUrl() {
    const url = toRawUrl($("url").value);
    if (!url) return;
    setStatus("下載資料中…", "busy");
    try {
      const t0 = performance.now();
      const { rows, headers } = parseCsv(await fetchText(url));
      const name = decodeURIComponent(url.split("/").pop().split("?")[0]) || url;
      setData(rows, headers, customCfg(name, url), performance.now() - t0);
    } catch (e) {
      setStatus(`無法讀取資料：${e.message}\n網頁在瀏覽器中執行，只能讀取允許跨網域存取（CORS）的網址，例如 raw.githubusercontent.com；也可以改用「上傳本機 CSV」。`, "bad");
    }
  }

  function loadFile(file) {
    setStatus("讀取檔案中…", "busy");
    const reader = new FileReader();
    reader.onload = () => {
      try { const { rows, headers } = parseCsv(reader.result); setData(rows, headers, customCfg(file.name, null), 0); }
      catch (e) { setStatus("無法讀取檔案：" + e.message, "bad"); }
    };
    reader.readAsText(file);
  }

  function customCfg(name, link) {
    return {
      key: "custom", target: null, name, link,
      note: "使用者自訂資料集：系統自動排除 ID 欄位、以中位數／眾數填補缺失值，文字欄位做 One-hot 編碼。",
      business: [
        "<b>問題</b>：以資料集中的其他欄位預測選定的目標欄位。",
        "<b>做法</b>：依 CRISP-DM 流程完成資料理解、前處理、特徵選擇、多元線性回歸建模與評估。",
        "<b>成功標準</b>：測試集 R² 越高、RMSE 越低越好，並找出對目標影響最大的特徵。",
      ],
    };
  }

  function setData(rows, headers, cfg, loadMs) {
    const types = A.columnTypes(rows, headers);
    const numeric = headers.filter((h) => types[h] !== "text");
    if (!numeric.length) { setStatus("資料中沒有數值欄位，無法進行回歸。", "bad"); return; }
    state.raw = { rows, headers, types };
    state.cfg = cfg;
    state.fsCache = {};
    state.dataId++;
    const target = cfg.target && numeric.includes(cfg.target) ? cfg.target : numeric[numeric.length - 1];
    $("target").innerHTML = numeric.map((h) => `<option ${h === target ? "selected" : ""}>${esc(h)}</option>`).join("");
    renderExclude(A.guessIdColumns(rows, headers, types, target));
    state.loadMs = loadMs;
    analyze();
  }

  function renderExclude(checked) {
    const target = $("target").value;
    const set = new Set(checked);
    $("exclude").innerHTML = state.raw.headers.filter((h) => h !== target)
      .map((h) => `<label class="check"><input type="checkbox" value="${esc(h)}" ${set.has(h) ? "checked" : ""}>${esc(h)}</label>`).join("");
    updateExcludeSummary();
  }
  function excluded() { return Array.from($("exclude").querySelectorAll("input:checked")).map((el) => el.value); }
  function updateExcludeSummary() {
    const ex = excluded();
    $("excludeSummary").textContent = `排除的欄位（ID 等）：${ex.length ? ex.join("、") : "無"}　（點此展開調整）`;
  }

  // ------------------------------------------------------------------
  // 分析
  // ------------------------------------------------------------------
  let pending = null;
  function analyze() {
    if (!state.raw) return;
    setStatus("分析中…", "busy");
    clearTimeout(pending);
    pending = setTimeout(runAnalysis, 30);
  }

  function runAnalysis() {
    const t0 = performance.now();
    try {
      const target = $("target").value;
      const testSize = parseFloat($("testSize").value);
      const seed = parseInt($("seed").value, 10) || 0;
      const prep = A.prepare(state.raw.rows, state.raw.headers, { target, exclude: excluded(), testSize, seed });
      if (!prep.X.length || prep.rows.length < 10) throw new Error("可用的特徵或資料筆數不足（至少需要 1 個特徵與 10 筆資料）。");
      const key = [state.dataId, target, excluded().join("|"), testSize, seed].join("#");
      if (!state.fsCache[key]) state.fsCache[key] = A.compareFeatureSelection(prep);
      const fs = state.fsCache[key];
      const rec = A.recommend(fs.results);
      const p = prep.X.length;

      const choice = $("fsMethod").value;
      const kEl = $("k");
      kEl.max = p;
      let method, k;
      if (choice === "auto") { method = rec.method; k = rec.k; kEl.disabled = true; kEl.value = k; }
      else if (choice === "all") { method = "全部特徵"; k = p; kEl.disabled = true; kEl.value = p; }
      else {
        method = choice; kEl.disabled = false;
        if (state.lastKKey !== key) kEl.value = Math.min(rec.k, p);
        k = Math.min(parseInt(kEl.value, 10), p);
      }
      state.lastKKey = key;
      $("kVal").textContent = `${k} / ${p}`;
      const feats = A.selectFeatures(prep, method, k);
      const trained = A.trainModels(prep, feats);
      const diag = A.diagnostics(prep, trained, feats);
      const finalName = $("finalModel").value;
      state.prep = prep;
      state.res = { fs, rec, method, k, feats, trained, diag, finalName, final: trained.models[finalName] };
      renderAll();
      const ms = performance.now() - t0;
      setStatus(`完成：${prep.rows.length.toLocaleString()} 筆資料、${p} 個特徵；分析耗時 ${(ms / 1000).toFixed(2)} 秒` +
        (state.loadMs ? `（讀取資料 ${(state.loadMs / 1000).toFixed(2)} 秒）` : "") + "。", "ok");
      state.loadMs = 0;
    } catch (e) {
      console.error(e);
      setStatus("分析失敗：" + e.message, "bad");
    }
  }

  // ------------------------------------------------------------------
  // 繪製
  // ------------------------------------------------------------------
  function renderAll() {
    const { prep, res, cfg } = { ...state, cfg: state.cfg };
    const link = cfg.link ? `<a href="${esc(cfg.link)}" target="_blank" rel="noopener">${esc(cfg.name)}</a>` : esc(cfg.name);
    $("datasetInfo").innerHTML = `<b>Dataset</b>：${link}　｜　<b>目標</b>：<code>${esc(prep.target)}</code><br>${esc(cfg.note)}`;
    const m = res.final.metrics;
    kpiRow($("kpis"), [
      ["資料筆數", prep.rows.length.toLocaleString()],
      ["原始特徵數", String(prep.numCols.length + prep.catCols.length)],
      ["選用特徵數", `${res.feats.length} / ${prep.X.length}`],
      [`測試 R²（${res.finalName}）`, m.R2.toFixed(4)],
      ["95% PI 涵蓋率", (res.trained.coverage * 100).toFixed(1) + "%"],
    ]);
    $("business").innerHTML = cfg.business.map((b) => `<li>${b}</li>`).join("") +
      (cfg.key === "custom" ? `<li><b>目標欄位</b>：<code>${esc(prep.target)}</code></li>` : "");
    renderTables();
    state.dirty = new Set(["t2", "t4", "t5", "t6"]);
    renderTab(state.activeTab);
  }

  function renderTab(id) {
    if (!state.res || !state.dirty.has(id)) return;
    state.dirty.delete(id);
    if (id === "t2") chartsDU();
    if (id === "t4") chartsModeling();
    if (id === "t5") chartsEval();
    if (id === "t6") renderPredictForm();
  }

  function renderTables() {
    const { prep, res } = state;
    const { rows, headers, types } = state.raw;
    // Data Understanding
    let missing = 0;
    const info = headers.map((h) => {
      let miss = 0;
      const u = new Set();
      for (const r of rows) { const v = r[h]; if (A.isMissing(v)) miss++; else u.add(v); }
      missing += miss;
      return [h, types[h] === "text" ? "文字" : types[h] === "int" ? "整數" : "數值", miss, u.size];
    });
    kpiRow($("duKpis"), [["列數 × 欄數（原始）", `${rows.length.toLocaleString()} × ${headers.length}`], ["缺失值總數", missing.toLocaleString()], ["重複列（排除後欄位）", prep.nDup.toLocaleString()]]);
    $("headTable").innerHTML = table(headers, rows.slice(0, 10).map((r) => headers.map((h) => (r[h] ?? ""))));
    $("infoTable").innerHTML = table(["欄位", "型態", "缺失值", "唯一值"], info, { num: [2, 3] });
    const desc = headers.filter((h) => types[h] !== "text").map((h) => {
      const v = rows.map((r) => A.toNumber(r[h])).filter((x) => !isNaN(x)).sort((a, b) => a - b);
      const mo = S.moments(v);
      return [h, v.length, fmt(mo.mean), fmt(mo.sd), fmt(v[0]), fmt(S.quantileSorted(v, 0.25)), fmt(S.quantileSorted(v, 0.5)), fmt(S.quantileSorted(v, 0.75)), fmt(v[v.length - 1])];
    });
    $("descTable").innerHTML = table(["欄位", "count", "mean", "std", "min", "25%", "50%", "75%", "max"], desc, { num: [1, 2, 3, 4, 5, 6, 7, 8] });
    renderHeatmap();

    // Data Preparation
    const steps = [
      ["排除欄位", prep.exclude.join(", ") || "無", "ID 等識別欄位不具預測意義"],
      ["目標缺失", `刪除 ${prep.nMissingTarget} 筆`, "監督式學習需要目標值"],
      ["重複資料", `刪除 ${prep.nDup} 筆`, "避免重複計算"],
      ["高基數文字欄位", prep.highCard.join(", ") || "無", "唯一值 > 30 的文字欄位不納入"],
      ["訓練／測試切分", `${Math.round((1 - prep.testSize) * 100)}% / ${Math.round(prep.testSize * 100)}%（random_state=${prep.seed}）→ ${prep.trainRows.length.toLocaleString()} / ${prep.testRows.length.toLocaleString()} 筆`, "保留獨立測試集"],
      ["數值缺失值", "以訓練集中位數填補", "只用訓練集計算，避免資料洩漏"],
      ["類別欄位", prep.catCols.length ? prep.catCols.join(", ") + "：眾數填補 + One-hot（drop_first）" : "無", "drop_first 避免虛擬變數陷阱"],
      ["標準化", "StandardScaler（每個 CV 折各自以訓練折計算）", "避免資料洩漏，並讓係數可比較"],
    ];
    if (prep.constCols.length) steps.push(["常數欄位", prep.constCols.join(", "), "訓練集中沒有變化，無法估計係數"]);
    $("prepTable").innerHTML = table(["步驟", "處理結果", "原因"], steps, { wrap: [1, 2] });
    $("encodedFeats").innerHTML = prep.names.map((n) => `<span class="pill">${esc(n)}</span>`).join("");

    // Modeling
    const { fs, rec, method, feats, trained } = res;
    const methods = A.METHODS;
    $("fsTable").innerHTML = table(["k"].concat(methods.map((m) => m + " R²")),
      fs.ks.map((k) => [k].concat(methods.map((m) => { const r = fs.results.find((o) => o.method === m && o.k === k); return `${r.mean.toFixed(6)} ± ${r.se.toFixed(6)}`; }))),
      { num: [0, 1, 2, 3], highlight: (i) => fs.ks[i] === rec.k });
    $("fsCallout").innerHTML = `<b>1-SE 規則推薦</b>：${esc(rec.method)}，k = ${rec.k}（在最佳平均 CV R² 一個標準誤內，選特徵最少的組合）。目前使用：<b>${esc(method)}，${feats.length} 個特徵</b>。<div class="pills" style="margin-top:8px">${feats.map((j) => `<span class="pill">${esc(prep.names[j])}</span>`).join("")}</div>`;
    $("modelTable").innerHTML = table(["Model", "MAE", "MSE", "RMSE", "R²", "Adjusted R²", "Alpha"],
      trained.table.map((r) => [r.model, fmt(r.MAE, 6), fmt(r.MSE, 6), fmt(r.RMSE, 6), r.R2.toFixed(6), r.AdjR2.toFixed(6), isNaN(r.alpha) ? "—" : fmt(r.alpha, 6)]),
      { num: [1, 2, 3, 4, 5, 6], highlight: (i) => trained.table[i].model === res.finalName });
    const o = trained.ols;
    const coefRows = feats.map((j, a) => ({ name: prep.names[j], coef: o.coef[a], se: o.se[a], p: o.pval[a], lo: o.lo[a], hi: o.hi[a], std: o.beta[a] }))
      .sort((x, y) => Math.abs(y.std) - Math.abs(x.std));
    res.coefRows = coefRows;
    $("olsTitle").textContent = `statsmodels 等價 OLS 係數（訓練 R² = ${o.r2.toFixed(4)}，Adjusted R² = ${o.adj.toFixed(4)}，截距 = ${fmt(o.intercept)}）`;
    $("coefTable").innerHTML = table(["Feature", "Coefficient", "Std Error", "P-value", "CI 2.5%", "CI 97.5%", "標準化係數"],
      coefRows.map((r) => [r.name, fmt(r.coef, 6), fmt(r.se, 6), fmtP(r.p), fmt(r.lo, 6), fmt(r.hi, 6), fmt(r.std, 6)]), { num: [1, 2, 3, 4, 5, 6] });
    const t = coefRows[0];
    $("coefNote").innerHTML = `<b>係數解讀</b>：影響最大的特徵是 <code>${esc(t.name)}</code>，在其他特徵固定下，每增加 1 單位，預測的 <code>${esc(prep.target)}</code> 平均${t.coef >= 0 ? "增加" : "減少"} ${fmt(Math.abs(t.coef), 6)}（p-value = ${fmtP(t.p)}）。`;

    // Evaluation tables
    const m = res.final.metrics;
    kpiRow($("evalKpis"), [["MAE", fmt(m.MAE, 6)], ["MSE", fmt(m.MSE, 6)], ["RMSE", fmt(m.RMSE, 6)], ["R²", m.R2.toFixed(4)], ["Adjusted R²", m.AdjR2.toFixed(4)]]);
    $("avpTitle").textContent = `實際值 vs 預測值（${res.finalName}）`;
    const d = res.diag;
    const diagRows = [
      ["線性", "殘差 vs 預測值圖", "殘差應隨機分布在 0 附近、無明顯曲線"],
      ["殘差平均為 0", fmt(d.mom.mean, 6), d.residMeanOk ? "接近 0 ✓" : "偏離 0（超過 2 個標準誤且大於殘差標準差的 5%）"],
      ["常態性", `偏態 ${d.mom.skew.toFixed(3)}、超額峰度 ${d.mom.kurt.toFixed(3)}`, Math.abs(d.mom.skew) < 1 && Math.abs(d.mom.kurt) < 3 ? "接近常態 ✓" : "偏離常態，需留意"],
      ["誤差獨立", `Durbin-Watson = ${d.dw.toFixed(3)}`, d.dw > 1.5 && d.dw < 2.5 ? "約 2 表示無自相關 ✓" : "可能有自相關"],
      ["變異數同質（等變異性）", `Breusch-Pagan p-value = ${fmtP(d.bp.p)}`, d.bp.p >= 0.05 ? "p ≥ 0.05，等變異 ✓" : "p < 0.05，有異質變異（樣本數很大時容易顯著，請搭配殘差圖判斷）"],
      ["無多重共線性", `最大 VIF = ${Math.max(...d.vif).toFixed(3)}`, Math.max(...d.vif) < 5 ? "VIF < 5 ✓" : "VIF ≥ 5，有共線性"],
    ];
    $("diagTable").innerHTML = table(["假設", "檢查結果", "判讀"], diagRows, { wrap: [2] });
    const vifRows = feats.map((j, a) => [prep.names[j], d.vif[a]]).sort((x, y) => y[1] - x[1]).map(([n, v]) => [n, v.toFixed(4)]);
    $("vifTable").innerHTML = table(["Feature", "VIF"], vifRows, { num: [1] });
    const nMax = Math.min(300, prep.testRows.length);
    const nShow = $("nShow");
    nShow.max = Math.max(10, nMax);
    if (parseInt(nShow.value, 10) > nMax) nShow.value = nMax;
    $("nShowVal").textContent = Math.min(parseInt(nShow.value, 10), nMax);
    $("piNote").textContent = `信賴區間（CI）表示「平均預測值」的不確定性；預測區間（PI）表示「單筆新資料」的範圍，因此較寬。完整測試集中有 ${(res.trained.coverage * 100).toFixed(1)}% 的實際值落在 95% 預測區間內。區間由 OLS 模型（選用特徵）計算。`;
  }

  function renderHeatmap() {
    const { prep } = state;
    const cols = prep.X.slice(0, 30), names = prep.names.slice(0, 30);
    const all = cols.concat([prep.y]);
    const labels = names.concat([prep.target]);
    const st = S.suffStats(all, prep.y, prep.trainRows);
    const sys = S.buildSystem(st);
    const k = all.length;
    const corr = (a, b) => sys.G[a * k + b] / Math.sqrt(sys.G[a * k + a] * sys.G[b * k + b]);
    state.corrTarget = names.map((n, j) => ({ n, r: corr(j, k - 1) }));
    const color = (r) => {
      const t = Math.min(1, Math.abs(r));
      const [cr, cg, cb] = r >= 0 ? [178, 34, 52] : [33, 102, 172];
      const mix = (c) => Math.round(255 + (c - 255) * t);
      return `rgb(${mix(cr)},${mix(cg)},${mix(cb)})`;
    };
    const showVal = k <= 22;
    let html = `<table class="heat"><tr><th class="rowh"></th>${labels.map((l) => `<th title="${esc(l)}">${esc(l.length > 10 ? l.slice(0, 9) + "…" : l)}</th>`).join("")}</tr>`;
    for (let a = 0; a < k; a++) {
      html += `<tr><th class="rowh">${esc(labels[a])}</th>`;
      for (let b = 0; b < k; b++) { const r = corr(a, b); html += `<td style="background:${color(r)}" title="${esc(labels[a])} × ${esc(labels[b])}: ${r.toFixed(3)}">${showVal ? r.toFixed(2) : ""}</td>`; }
      html += "</tr>";
    }
    $("heatmap").innerHTML = html + "</table>";
  }

  function chartsDU() {
    const { prep } = state;
    const yv = Array.from(prep.y);
    const lo = Math.min(...yv), hi = Math.max(...yv);
    const nb = 40, w = (hi - lo) / nb || 1;
    const counts = new Array(nb).fill(0);
    for (const v of yv) counts[Math.min(nb - 1, Math.floor((v - lo) / w))]++;
    $("histTitle").textContent = `目標變數分布：${prep.target}`;
    chart("histChart", {
      type: "bar",
      data: { labels: counts.map((_, i) => fmt(lo + (i + 0.5) * w, 3)), datasets: [{ label: "次數", data: counts, backgroundColor: BLUE, barPercentage: 1, categoryPercentage: 1 }] },
      options: { plugins: { legend: { display: false } }, scales: { x: axis(prep.target), y: axis("Frequency") } },
    });
    const ct = state.corrTarget.slice().sort((a, b) => b.r - a.r);
    $("corrTitle").textContent = `各特徵與 ${prep.target} 的相關係數`;
    $("corrBox").style.height = Math.max(380, 22 * ct.length + 80) + "px";
    chart("corrChart", {
      type: "bar",
      data: { labels: ct.map((o) => o.n), datasets: [{ label: "Pearson r", data: ct.map((o) => o.r), backgroundColor: ct.map((o) => (o.r >= 0 ? BLUE : ORANGE)) }] },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: axis("Pearson r"), y: axis("", { ticks: { autoSkip: false } }) } },
    });
  }

  function chartsModeling() {
    const { res } = state;
    const { fs } = res;
    chart("fsChart", {
      type: "line",
      data: {
        labels: fs.ks.map(String),
        datasets: A.METHODS.map((m) => ({
          label: m, data: fs.ks.map((k) => fs.results.find((o) => o.method === m && o.k === k).mean),
          borderColor: METHOD_COLORS[m], backgroundColor: METHOD_COLORS[m], pointRadius: 4, tension: 0.15,
        })),
      },
      options: { interaction: { mode: "index", intersect: false }, scales: { x: axis("選取特徵數 k"), y: axis("Mean CV R²") } },
    });
    const top = res.coefRows.slice(0, 15).slice().reverse();
    $("coefBox").style.height = Math.max(380, 26 * top.length + 80) + "px";
    chart("coefChart", {
      type: "bar",
      data: { labels: top.map((r) => r.name), datasets: [{ label: "標準化係數", data: top.map((r) => r.std), backgroundColor: top.map((r) => (r.std >= 0 ? BLUE : ORANGE)) }] },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: axis("標準化係數"), y: axis("", { ticks: { autoSkip: false } }) } },
    });
  }

  function chartsEval() {
    const { res, prep } = state;
    const { trained, diag } = res;
    const n = trained.yte.length;
    const nShow = Math.min(parseInt($("nShow").value, 10), n);
    $("nShowVal").textContent = nShow;
    const pick = sampleIdx(n, nShow, prep.seed).sort((a, b) => trained.ci.mean[a] - trained.ci.mean[b]);
    const xs = pick.map((_, i) => i);
    const ci = trained.ci;
    chart("piChart", {
      type: "line",
      data: {
        labels: xs,
        datasets: [
          { label: "95% 預測區間（PI）", data: pick.map((i) => ci.piHi[i]), borderWidth: 0, pointRadius: 0, backgroundColor: "rgba(232,131,58,0.22)", fill: "+1" },
          { label: "_pi_lo", data: pick.map((i) => ci.piLo[i]), borderWidth: 0, pointRadius: 0, fill: false },
          { label: "95% 信賴區間（CI）", data: pick.map((i) => ci.ciHi[i]), borderWidth: 0, pointRadius: 0, backgroundColor: "rgba(46,111,183,0.45)", fill: "+1" },
          { label: "_ci_lo", data: pick.map((i) => ci.ciLo[i]), borderWidth: 0, pointRadius: 0, fill: false },
          { label: "預測值", data: pick.map((i) => ci.mean[i]), borderColor: BLUE, borderWidth: 2.5, pointRadius: 0, fill: false },
          { label: "實際值", data: pick.map((i) => trained.yte[i]), showLine: false, pointRadius: 3.5, pointBackgroundColor: "#222", borderColor: "#222", fill: false },
        ],
      },
      options: {
        plugins: { legend: { position: "bottom", labels: { filter: (item) => !item.text.startsWith("_") } }, title: { display: true, text: "OLS 預測值與 95% 信賴區間／預測區間（依預測值排序）" } },
        scales: { x: axis("測試樣本（依預測值排序）", { ticks: { maxTicksLimit: 12 } }), y: axis(prep.target) },
      },
    });

    const sidx = sampleIdx(n, 3000, prep.seed + 7);
    const pred = res.final.pred;
    const pts = sidx.map((i) => ({ x: trained.yte[i], y: pred[i] }));
    let lo = Infinity, hi = -Infinity;
    for (const p of pts) { lo = Math.min(lo, p.x, p.y); hi = Math.max(hi, p.x, p.y); }
    chart("avpChart", {
      type: "scatter",
      data: { datasets: [
        { label: "樣本", data: pts, pointRadius: 2.5, backgroundColor: "rgba(46,111,183,0.4)" },
        { label: "y = x", data: [{ x: lo, y: lo }, { x: hi, y: hi }], showLine: true, borderColor: RED, borderDash: [6, 4], pointRadius: 0 },
      ] },
      options: { scales: { x: axis("實際值"), y: axis("預測值") } },
    });
    const rv = sidx.map((i) => ({ x: ci.mean[i], y: diag.resid[i] }));
    const xr = rv.map((p) => p.x);
    chart("rvfChart", {
      type: "scatter",
      data: { datasets: [
        { label: "殘差", data: rv, pointRadius: 2.5, backgroundColor: "rgba(46,111,183,0.4)" },
        { label: "0", data: [{ x: Math.min(...xr), y: 0 }, { x: Math.max(...xr), y: 0 }], showLine: true, borderColor: RED, borderDash: [6, 4], pointRadius: 0 },
      ] },
      options: { plugins: { legend: { display: false } }, scales: { x: axis("預測值"), y: axis("殘差") } },
    });
    // 殘差分布
    const r = Array.from(diag.resid);
    const rlo = Math.min(...r), rhi = Math.max(...r), nb = 40, w = (rhi - rlo) / nb || 1;
    const counts = new Array(nb).fill(0);
    for (const v of r) counts[Math.min(nb - 1, Math.floor((v - rlo) / w))]++;
    chart("residHist", {
      type: "bar",
      data: { labels: counts.map((_, i) => fmt(rlo + (i + 0.5) * w, 3)), datasets: [{ label: "次數", data: counts, backgroundColor: BLUE, barPercentage: 1, categoryPercentage: 1 }] },
      options: { plugins: { legend: { display: false }, title: { display: true, text: "殘差分布" } }, scales: { x: axis("殘差"), y: axis("次數") } },
    });
    // Q-Q plot
    const qs = sidx.map((i) => diag.resid[i]).sort((a, b) => a - b);
    const theo = S.probplotX(qs.length);
    const mx = S.mean(theo), my = S.mean(qs);
    let sxy = 0, sxx = 0;
    for (let i = 0; i < qs.length; i++) { sxy += (theo[i] - mx) * (qs[i] - my); sxx += (theo[i] - mx) ** 2; }
    const slope = sxy / sxx, icpt = my - slope * mx;
    chart("qqChart", {
      type: "scatter",
      data: { datasets: [
        { label: "殘差", data: qs.map((v, i) => ({ x: theo[i], y: v })), pointRadius: 2.5, backgroundColor: "rgba(46,111,183,0.5)" },
        { label: "常態參考線", data: [{ x: theo[0], y: icpt + slope * theo[0] }, { x: theo[theo.length - 1], y: icpt + slope * theo[theo.length - 1] }], showLine: true, borderColor: RED, pointRadius: 0 },
      ] },
      options: { plugins: { legend: { display: false }, title: { display: true, text: "Normal Q-Q Plot" } }, scales: { x: axis("理論分位數"), y: axis("樣本分位數") } },
    });
  }

  // ------------------------------------------------------------------
  // Deployment
  // ------------------------------------------------------------------
  function renderPredictForm() {
    const { prep } = state;
    const cols = prep.numCols.concat(prep.catCols);
    $("predictForm").innerHTML = cols.map((c, i) => {
      if (prep.catCols.includes(c)) {
        return `<div><label for="in_${i}">${esc(c)}</label><select id="in_${i}" data-col="${esc(c)}">${prep.categories[c].map((v) => `<option ${v === prep.modes[c] ? "selected" : ""}>${esc(v)}</option>`).join("")}</select></div>`;
      }
      const isInt = state.raw.types[c] === "int";
      const v = isInt ? Math.round(prep.medians[c]) : prep.medians[c];
      return `<div><label for="in_${i}">${esc(c)}</label><input id="in_${i}" type="number" step="${isInt ? 1 : "any"}" value="${v}" data-col="${esc(c)}"></div>`;
    }).join("");
    $("predictForm").querySelectorAll("input,select").forEach((el) => el.addEventListener("input", predictNow));
    predictNow();
  }

  function predictNow() {
    const { prep, res } = state;
    const input = {};
    $("predictForm").querySelectorAll("[data-col]").forEach((el) => { input[el.dataset.col] = el.tagName === "SELECT" ? el.value : el.value === "" ? null : Number(el.value); });
    const x = A.encodeInput(prep, input);
    const feats = res.feats;
    const pr = S.olsPredict(res.trained.ols, feats.map((j) => x[j]));
    const sys = res.trained.sys;
    let mp = sys.my;
    feats.forEach((j, a) => { mp += res.final.beta[a] * (x[j] - sys.m[j]) / sys.sd[j]; });
    kpiRow($("predictOut"), [
      [`${res.finalName} 預測值`, fmt(mp, 6)],
      ["OLS 預測值", fmt(pr.mean, 6)],
      ["95% 信賴區間", `${fmt(pr.ciLo, 4)} ~ ${fmt(pr.ciHi, 4)}`],
      ["95% 預測區間", `${fmt(pr.piLo, 4)} ~ ${fmt(pr.piHi, 4)}`],
    ]);
  }

  function downloadCsv() {
    const { prep, res } = state;
    const cols = prep.numCols.concat(prep.catCols);
    const t = prep.target;
    const header = cols.concat([`Actual_${t}`, `Predicted_${t}`, "PI_lower", "PI_upper"]);
    const q = (v) => { const s = String(v ?? ""); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    const lines = [header.map(q).join(",")];
    const tr = res.trained;
    prep.testRows.forEach((i, r) => {
      lines.push(cols.map((c) => q(prep.rows[i][c])).concat([tr.yte[r], res.final.pred[r], tr.ci.piLo[r], tr.ci.piHi[r]]).join(","));
    });
    const blob = new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `predictions_${state.cfg.key}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ------------------------------------------------------------------
  // 事件
  // ------------------------------------------------------------------
  function init() {
    $("problem").addEventListener("change", (e) => loadProblem(e.target.value));
    $("loadUrl").addEventListener("click", loadUrl);
    $("url").addEventListener("keydown", (e) => { if (e.key === "Enter") loadUrl(); });
    $("file").addEventListener("change", (e) => { if (e.target.files[0]) loadFile(e.target.files[0]); });
    $("target").addEventListener("change", () => {
      state.fsCache = {};
      renderExclude(A.guessIdColumns(state.raw.rows, state.raw.headers, state.raw.types, $("target").value));
      analyze();
    });
    $("exclude").addEventListener("change", () => { updateExcludeSummary(); analyze(); });
    $("testSize").addEventListener("input", () => { $("testSizeVal").textContent = parseFloat($("testSize").value).toFixed(2); analyze(); });
    $("seed").addEventListener("change", analyze);
    $("fsMethod").addEventListener("change", analyze);
    $("k").addEventListener("input", () => { $("kVal").textContent = `${$("k").value} / ${$("k").max}`; analyze(); });
    $("finalModel").addEventListener("change", analyze);
    $("nShow").addEventListener("input", () => { if (state.activeTab === "t5") chartsEval(); else state.dirty.add("t5"); });
    $("download").addEventListener("click", downloadCsv);
    document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
      document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === b.dataset.tab));
      state.activeTab = b.dataset.tab;
      renderTab(b.dataset.tab);
    }));
    Chart.defaults.font.family = '"Noto Sans TC","Microsoft JhengHei",system-ui,sans-serif';
    Chart.defaults.color = "#476071";
    loadProblem($("problem").value);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
