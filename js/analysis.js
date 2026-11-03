/*
 * analysis.js — CRISP-DM 回歸分析流程（資料前處理、特徵選擇比較、模型訓練、評估）
 * 純函式，瀏覽器與 Node.js 皆可使用（Node 用於與 Python 結果交叉驗證）
 */
(function (root) {
  "use strict";
  const S = typeof require === "function" && typeof module !== "undefined" ? require("./stats.js") : root.Stats;

  const METHODS = ["SelectKBest", "RFE", "LassoCV"];

  function kGrid(p) {
    if (p <= 10) return Array.from({ length: p }, (_, i) => i + 1);
    const set = new Set([p]);
    for (let i = 0; i < 8; i++) set.add(Math.round(1 + ((p - 1) * i) / 7));
    return Array.from(set).sort((a, b) => a - b);
  }

  // ------------------------------------------------------------------
  // 資料前處理
  // ------------------------------------------------------------------
  function isMissing(v) { return v === null || v === undefined || v === "" || (typeof v === "number" && isNaN(v)); }

  function toNumber(v) {
    if (isMissing(v)) return NaN;
    if (typeof v === "number") return v;
    if (typeof v === "boolean") return v ? 1 : 0;
    const s = String(v).trim().replace(/,/g, "");
    if (s === "") return NaN;
    const n = Number(s);
    return isFinite(n) ? n : NaN;
  }

  /** 判斷欄位型態：數值（可轉數字的比例 ≥ 95%）或類別 */
  function columnTypes(rows, headers) {
    const types = {};
    for (const h of headers) {
      let nonMissing = 0, numeric = 0, allInt = true;
      for (const r of rows) {
        const v = r[h];
        if (isMissing(v)) continue;
        nonMissing++;
        const n = toNumber(v);
        if (!isNaN(n)) { numeric++; if (!Number.isInteger(n)) allInt = false; }
      }
      types[h] = nonMissing > 0 && numeric / nonMissing >= 0.95 ? (allInt ? "int" : "float") : "text";
    }
    return types;
  }

  function guessIdColumns(rows, headers, types, target) {
    const out = [];
    for (const h of headers) {
      if (h === target) continue;
      const name = String(h).trim().toLowerCase();
      if (["id", "index", "unnamed: 0", "no", "no."].includes(name)) { out.push(h); continue; }
      if (types[h] === "int" && rows.length > 20) {
        let unique = true, mono = true, prev = -Infinity;
        const seen = new Set();
        for (const r of rows) {
          const v = toNumber(r[h]);
          if (isNaN(v) || seen.has(v)) { unique = false; break; }
          seen.add(v);
          if (v < prev) mono = false;
          prev = v;
        }
        if (unique && mono) out.push(h);
      }
    }
    return out;
  }

  /**
   * 建立分析資料集：清理 → 切分 → 以訓練集學習中位數／眾數／類別 → One-hot（drop_first）
   */
  function prepare(rawRows, headers, opts) {
    const { target, exclude = [], testSize = 0.2, seed = 42 } = opts;
    const types = columnTypes(rawRows, headers);
    const featCols = headers.filter((h) => h !== target && !exclude.includes(h));

    // 刪除目標缺失
    let rows = rawRows.filter((r) => !isNaN(toNumber(r[target])));
    const nMissingTarget = rawRows.length - rows.length;
    // 刪除重複列（只看保留欄位）
    const keyCols = featCols.concat([target]);
    const seen = new Set();
    const dedup = [];
    for (const r of rows) {
      const key = keyCols.map((c) => (isMissing(r[c]) ? "\u0000" : String(r[c]))).join("\u0001");
      if (seen.has(key)) continue;
      seen.add(key);
      dedup.push(r);
    }
    const nDup = rows.length - dedup.length;
    rows = dedup;

    const numCols = featCols.filter((c) => types[c] !== "text");
    const textCols = featCols.filter((c) => types[c] === "text");
    const catCols = [], highCard = [];
    for (const c of textCols) {
      const u = new Set();
      for (const r of rows) if (!isMissing(r[c])) u.add(String(r[c]));
      (u.size <= 30 ? catCols : highCard).push(c);
    }
    const n = rows.length;
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) y[i] = toNumber(rows[i][target]);

    // 訓練／測試切分
    const perm = S.permutation(n, seed);
    const nTest = Math.ceil(n * testSize);
    const testRows = Int32Array.from(perm.slice(0, nTest)).sort();
    const trainRows = Int32Array.from(perm.slice(nTest)).sort();

    // 只用訓練集學習填補值
    const medians = {}, modes = {}, categories = {};
    for (const c of numCols) {
      const vals = [];
      for (const i of trainRows) { const v = toNumber(rows[i][c]); if (!isNaN(v)) vals.push(v); }
      vals.sort((a, b) => a - b);
      medians[c] = vals.length ? S.quantileSorted(vals, 0.5) : 0;
    }
    for (const c of catCols) {
      const cnt = new Map();
      for (const i of trainRows) { const v = rows[i][c]; if (!isMissing(v)) cnt.set(String(v), (cnt.get(String(v)) || 0) + 1); }
      let best = null, bestN = -1;
      for (const [k, v] of Array.from(cnt.entries()).sort((a, b) => (a[0] < b[0] ? -1 : 1))) if (v > bestN) { best = k; bestN = v; }
      modes[c] = best ?? "missing";
      const cats = new Set(cnt.keys());
      cats.add(modes[c]);
      categories[c] = Array.from(cats).sort();
    }

    const names = [], X = [], origin = [];
    for (const c of numCols) {
      const col = new Float64Array(n);
      for (let i = 0; i < n; i++) { const v = toNumber(rows[i][c]); col[i] = isNaN(v) ? medians[c] : v; }
      names.push(String(c)); X.push(col); origin.push(c);
    }
    for (const c of catCols) {
      const cats = categories[c];
      for (const level of cats.slice(1)) {
        const col = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          const v = isMissing(rows[i][c]) ? modes[c] : String(rows[i][c]);
          col[i] = v === level ? 1 : 0;
        }
        names.push(`${c}_${level}`); X.push(col); origin.push(c);
      }
    }
    // 移除訓練集中為常數的欄位
    const constCols = [];
    const keep = [];
    for (let j = 0; j < X.length; j++) {
      let first = X[j][trainRows[0]], isConst = true;
      for (const i of trainRows) if (X[j][i] !== first) { isConst = false; break; }
      if (isConst) constCols.push(names[j]); else keep.push(j);
    }
    return {
      rows, types, target, featCols, numCols, catCols, highCard, nMissingTarget, nDup, exclude,
      X: keep.map((j) => X[j]), names: keep.map((j) => names[j]), origin: keep.map((j) => origin[j]),
      constCols, y, trainRows, testRows, medians, modes, categories, seed, testSize,
    };
  }

  /** 把一筆原始輸入轉成編碼後特徵（依 prep.names 順序） */
  function encodeInput(prep, input) {
    return prep.names.map((name, j) => {
      const c = prep.origin[j];
      if (prep.catCols.includes(c)) {
        const v = isMissing(input[c]) ? prep.modes[c] : String(input[c]);
        return name === `${c}_${v}` ? 1 : 0;
      }
      const v = toNumber(input[c]);
      return isNaN(v) ? prep.medians[c] : v;
    });
  }

  // ------------------------------------------------------------------
  // 特徵選擇比較（5-fold CV，只用訓練集）
  // ------------------------------------------------------------------
  function contiguousChunks(rows, k) { return S.kfold(rows, k, false); }

  function lassoRanking(X, y, trainRows, trainStats, sys, idx, k) {
    const chunks = contiguousChunks(trainRows, 3);
    const chunkStats = chunks.map((c) => S.suffStats(X, y, c));
    const res = S.lassoCV(trainStats, chunkStats, idx, sys.sd);
    return { order: S.topByAbs(idx, res.beta, idx.length), alpha: res.alpha, beta: res.beta };
  }

  function compareFeatureSelection(prep) {
    const { X, y, trainRows, seed } = prep;
    const p = X.length;
    const idx = Array.from({ length: p }, (_, j) => j);
    const ks = kGrid(p);
    const nSplits = trainRows.length >= 25 ? 5 : 3;
    const folds = S.kfold(trainRows, nSplits, true, seed);
    const total = S.suffStats(X, y, trainRows);
    const scores = {};
    for (const m of METHODS) for (const k of ks) scores[`${m}|${k}`] = [];

    for (const val of folds) {
      const valSet = new Set(val);
      const trRows = Int32Array.from(Array.from(trainRows).filter((i) => !valSet.has(i)));
      const vst = S.suffStats(X, y, val);
      const trS = S.subtractStats(total, vst);
      const sys = S.buildSystem(trS);
      const kb = S.rankKBest(sys, idx);
      const rfe = S.rfeSets(sys, idx, ks);
      const lasso = lassoRanking(X, y, trRows, trS, sys, idx).order;
      for (const k of ks) {
        const sets = { SelectKBest: kb.slice(0, k).sort((a, b) => a - b), RFE: rfe[k], LassoCV: lasso.slice(0, k).sort((a, b) => a - b) };
        for (const m of METHODS) {
          const f = sets[m];
          const beta = S.olsCoef(sys, f);
          scores[`${m}|${k}`].push(S.r2FromStats(vst, S.valSSE(vst, sys, f, beta)));
        }
      }
    }
    const results = [];
    for (const m of METHODS) for (const k of ks) {
      const s = scores[`${m}|${k}`];
      const mu = S.mean(s);
      const sd = Math.sqrt(s.reduce((a, v) => a + (v - mu) ** 2, 0) / Math.max(s.length - 1, 1));
      results.push({ method: m, k, mean: mu, std: sd, se: sd / Math.sqrt(s.length) });
    }
    return { results, ks, nSplits };
  }

  function recommend(results) {
    let best = results[0];
    for (const r of results) if (r.mean > best.mean) best = r;
    const cand = results.filter((r) => r.mean >= best.mean - best.se);
    cand.sort((a, b) => a.k - b.k || b.mean - a.mean);
    return { method: cand[0].method, k: cand[0].k };
  }

  /** 在完整訓練集上依方法選出 k 個特徵 */
  function selectFeatures(prep, method, k) {
    const { X, y, trainRows } = prep;
    const p = X.length;
    const idx = Array.from({ length: p }, (_, j) => j);
    if (method === "全部特徵" || k >= p) return idx;
    const total = S.suffStats(X, y, trainRows);
    const sys = S.buildSystem(total);
    if (method === "SelectKBest") return S.rankKBest(sys, idx).slice(0, k).sort((a, b) => a - b);
    if (method === "RFE") return S.rfeSets(sys, idx, [k])[k];
    return lassoRanking(X, y, trainRows, total, sys, idx).order.slice(0, k).sort((a, b) => a - b);
  }

  // ------------------------------------------------------------------
  // 模型訓練與評估
  // ------------------------------------------------------------------
  function metrics(yTrue, yPred, p) {
    const n = yTrue.length;
    let ae = 0, se = 0, my = 0;
    for (let i = 0; i < n; i++) my += yTrue[i];
    my /= n;
    let sst = 0;
    for (let i = 0; i < n; i++) { const e = yTrue[i] - yPred[i]; ae += Math.abs(e); se += e * e; sst += (yTrue[i] - my) ** 2; }
    const mse = se / n;
    const r2 = sst > 0 ? 1 - se / sst : 0;
    const adj = n - p - 1 > 0 ? 1 - ((1 - r2) * (n - 1)) / (n - p - 1) : NaN;
    return { MAE: ae / n, MSE: mse, RMSE: Math.sqrt(mse), R2: r2, AdjR2: adj };
  }

  function trainModels(prep, feats) {
    const { X, y, trainRows, testRows, seed } = prep;
    const total = S.suffStats(X, y, trainRows);
    const sys = S.buildSystem(total);
    const yte = Float64Array.from(testRows, (i) => y[i]);
    const p = feats.length;

    const ols = S.olsInference(sys, feats);
    const ridgeAlphas = Array.from({ length: 40 }, (_, i) => Math.pow(10, -4 + (7 * i) / 39));
    const ridge = S.ridgeGCV(X, y, trainRows, sys, feats, ridgeAlphas);
    const cvFolds = S.kfold(trainRows, trainRows.length >= 25 ? 5 : 3, true, seed + 1);
    const lasso = S.lassoCV(total, cvFolds.map((f) => S.suffStats(X, y, f)), feats, sys.sd);

    const models = {
      LinearRegression: { beta: ols.beta, alpha: NaN },
      Ridge: { beta: ridge.beta, alpha: ridge.alpha },
      Lasso: { beta: lasso.beta, alpha: lasso.alpha },
    };
    const table = [];
    for (const [name, m] of Object.entries(models)) {
      m.pred = S.predictRows(X, testRows, sys, feats, m.beta);
      m.metrics = metrics(yte, m.pred, p);
      table.push({ model: name, ...m.metrics, alpha: m.alpha });
    }
    // OLS 測試集 95% 區間
    const ci = { mean: new Float64Array(testRows.length), ciLo: new Float64Array(testRows.length), ciHi: new Float64Array(testRows.length), piLo: new Float64Array(testRows.length), piHi: new Float64Array(testRows.length) };
    let covered = 0;
    for (let r = 0; r < testRows.length; r++) {
      const i = testRows[r];
      const pr = S.olsPredict(ols, feats.map((j) => X[j][i]));
      ci.mean[r] = pr.mean; ci.ciLo[r] = pr.ciLo; ci.ciHi[r] = pr.ciHi; ci.piLo[r] = pr.piLo; ci.piHi[r] = pr.piHi;
      if (yte[r] >= pr.piLo && yte[r] <= pr.piHi) covered++;
    }
    return { sys, ols, models, table, yte, ci, coverage: covered / testRows.length };
  }

  function diagnostics(prep, trained, feats) {
    const { X, y, trainRows, testRows, seed } = prep;
    const { ols, sys, ci, yte } = trained;
    const resid = Float64Array.from(yte, (v, r) => v - ci.mean[r]);
    const mom = S.moments(resid);
    const trainPred = S.predictRows(X, trainRows, sys, feats, ols.beta);
    const trainResid = Float64Array.from(trainRows, (i, r) => y[i] - trainPred[r]);
    const dw = S.durbinWatson(trainResid);
    // Breusch-Pagan 與 VIF 最多抽 20,000 筆訓練資料
    const nSample = Math.min(20000, trainRows.length);
    const perm = S.permutation(trainRows.length, seed);
    const pick = Array.from(perm.slice(0, nSample)).sort((a, b) => a - b);
    const sampleRows = Int32Array.from(pick, (r) => trainRows[r]);
    const sampleResid = Float64Array.from(pick, (r) => trainResid[r]);
    const bp = S.breuschPagan(X, sampleRows, feats, sampleResid);
    const vsys = S.buildSystem(S.suffStats(X, y, sampleRows));
    const vifs = S.vif(vsys, feats);
    const residSE = mom.sd / Math.sqrt(resid.length);
    // 殘差平均：在 2 個標準誤內，或小於殘差標準差的 5%（實務上可忽略）即視為接近 0
    return { resid, mom, dw, bp, vif: vifs, residMeanOk: Math.abs(mom.mean) < 2 * residSE || Math.abs(mom.mean) < 0.05 * mom.sd };
  }

  const api = { METHODS, kGrid, isMissing, toNumber, columnTypes, guessIdColumns, prepare, encodeInput, compareFeatureSelection, recommend, selectFeatures, metrics, trainModels, diagnostics };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Analysis = api;
})(typeof self !== "undefined" ? self : this);
