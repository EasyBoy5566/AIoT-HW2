/*
 * stats.js — 多元線性回歸運算核心（純 JavaScript）
 * HW2｜學號 5115056015 鍾易憲
 *
 * 所有模型都由「充分統計量」(sum x, sum x·x', sum x·y …) 計算，
 * 因此交叉驗證、特徵選擇與正則化路徑都只需 O(p²) 的運算，5 萬筆資料也能在瀏覽器即時完成。
 *
 * 對應 Python 版本：
 *   LinearRegression / statsmodels OLS  → olsInference
 *   RidgeCV（留一 GCV 選 α）             → ridgeGCV
 *   LassoCV（座標下降 + K-fold 選 α）     → lassoCV
 *   SelectKBest(f_regression) / RFE / SelectFromModel(LassoCV) → rankKBest / rfeSets / lasso 排名
 */
(function (root) {
  "use strict";

  // ------------------------------------------------------------------
  // 亂數與資料切分（固定 seed，可重現）
  // ------------------------------------------------------------------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function permutation(n, seed) {
    const rng = mulberry32(seed);
    const idx = new Int32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    return idx;
  }

  /** 把 rows 切成 k 份；shuffle=false 時依原順序連續切（同 sklearn KFold 預設） */
  function kfold(rows, k, shuffle, seed) {
    const order = shuffle ? Array.from(permutation(rows.length, seed), (i) => rows[i]) : Array.from(rows);
    const folds = [];
    const n = order.length;
    let start = 0;
    for (let f = 0; f < k; f++) {
      const size = Math.floor(n / k) + (f < n % k ? 1 : 0);
      folds.push(Int32Array.from(order.slice(start, start + size)));
      start += size;
    }
    return folds;
  }

  // ------------------------------------------------------------------
  // 充分統計量
  // ------------------------------------------------------------------
  /** X: 長度 p 的欄位陣列（每欄 Float64Array，長度 N）；y: Float64Array；rows: 要納入的列 */
  function suffStats(X, y, rows) {
    const p = X.length, n = rows.length;
    const sx = new Float64Array(p), sxy = new Float64Array(p), sxx = new Float64Array(p * p);
    const xi = new Float64Array(p);
    let sy = 0, syy = 0;
    for (let r = 0; r < n; r++) {
      const i = rows[r];
      const yi = y[i];
      sy += yi; syy += yi * yi;
      for (let j = 0; j < p; j++) {
        const v = X[j][i];
        xi[j] = v; sx[j] += v; sxy[j] += v * yi;
      }
      for (let j = 0; j < p; j++) {
        const v = xi[j], off = j * p;
        for (let k = j; k < p; k++) sxx[off + k] += v * xi[k];
      }
    }
    for (let j = 0; j < p; j++) for (let k = j + 1; k < p; k++) sxx[k * p + j] = sxx[j * p + k];
    return { n, p, sx, sxy, sxx, sy, syy };
  }

  function subtractStats(a, b) {
    const p = a.p;
    const out = { n: a.n - b.n, p, sx: new Float64Array(p), sxy: new Float64Array(p), sxx: new Float64Array(p * p), sy: a.sy - b.sy, syy: a.syy - b.syy };
    for (let j = 0; j < p; j++) { out.sx[j] = a.sx[j] - b.sx[j]; out.sxy[j] = a.sxy[j] - b.sxy[j]; }
    for (let j = 0; j < p * p; j++) out.sxx[j] = a.sxx[j] - b.sxx[j];
    return out;
  }

  /**
   * 由統計量建立「標準化 + 置中」的系統（等同 StandardScaler 後再含截距回歸）
   * sdOverride：用外層訓練集的標準差（例如 Pipeline 內的 LassoCV）
   */
  function buildSystem(st, sdOverride) {
    const { n, p } = st;
    const m = new Float64Array(p), sd = new Float64Array(p), c = new Float64Array(p), G = new Float64Array(p * p);
    const my = st.sy / n;
    for (let j = 0; j < p; j++) m[j] = st.sx[j] / n;
    for (let j = 0; j < p; j++) {
      if (sdOverride) sd[j] = sdOverride[j];
      else {
        const v = st.sxx[j * p + j] / n - m[j] * m[j];
        sd[j] = v > 1e-12 * Math.max(1, m[j] * m[j]) ? Math.sqrt(v) : 1;
      }
    }
    for (let j = 0; j < p; j++) {
      c[j] = (st.sxy[j] - n * m[j] * my) / sd[j];
      for (let k = 0; k < p; k++) G[j * p + k] = (st.sxx[j * p + k] - n * m[j] * m[k]) / (sd[j] * sd[k]);
    }
    const syyc = st.syy - n * my * my;
    return { n, p, m, sd, my, c, G, syyc };
  }

  function subMatrix(sys, idx) {
    const k = idx.length, p = sys.p;
    const A = new Float64Array(k * k), b = new Float64Array(k);
    for (let a = 0; a < k; a++) {
      b[a] = sys.c[idx[a]];
      for (let d = 0; d < k; d++) A[a * k + d] = sys.G[idx[a] * p + idx[d]];
    }
    return { A, b, k };
  }

  // ------------------------------------------------------------------
  // 線性代數（小型稠密矩陣）
  // ------------------------------------------------------------------
  /** Gauss-Jordan 反矩陣（部分主元）；奇異時加極小 ridge */
  function invert(Ain, k) {
    let A = Float64Array.from(Ain);
    let ridge = 0;
    for (let attempt = 0; attempt < 4; attempt++) {
      const M = Float64Array.from(A), I = new Float64Array(k * k);
      for (let i = 0; i < k; i++) I[i * k + i] = 1;
      let ok = true;
      for (let col = 0; col < k; col++) {
        let piv = col, best = Math.abs(M[col * k + col]);
        for (let r = col + 1; r < k; r++) { const v = Math.abs(M[r * k + col]); if (v > best) { best = v; piv = r; } }
        if (best < 1e-12) { ok = false; break; }
        if (piv !== col) {
          for (let c = 0; c < k; c++) {
            let t = M[col * k + c]; M[col * k + c] = M[piv * k + c]; M[piv * k + c] = t;
            t = I[col * k + c]; I[col * k + c] = I[piv * k + c]; I[piv * k + c] = t;
          }
        }
        const d = M[col * k + col];
        for (let c = 0; c < k; c++) { M[col * k + c] /= d; I[col * k + c] /= d; }
        for (let r = 0; r < k; r++) {
          if (r === col) continue;
          const f = M[r * k + col];
          if (f === 0) continue;
          for (let c = 0; c < k; c++) { M[r * k + c] -= f * M[col * k + c]; I[r * k + c] -= f * I[col * k + c]; }
        }
      }
      if (ok) return I;
      let tr = 0;
      for (let i = 0; i < k; i++) tr += Math.abs(Ain[i * k + i]);
      ridge = (ridge || 1e-10) * 100 * Math.max(tr / k, 1);
      A = Float64Array.from(Ain);
      for (let i = 0; i < k; i++) A[i * k + i] += ridge;
    }
    throw new Error("矩陣無法反轉（特徵可能完全共線）");
  }

  function matVec(A, x, k) {
    const out = new Float64Array(k);
    for (let i = 0; i < k; i++) { let s = 0; for (let j = 0; j < k; j++) s += A[i * k + j] * x[j]; out[i] = s; }
    return out;
  }

  function solve(A, b, k) { return matVec(invert(A, k), b, k); }

  /** 對稱矩陣特徵分解（Jacobi） */
  function eigSym(Ain, k) {
    const A = Float64Array.from(Ain), V = new Float64Array(k * k);
    for (let i = 0; i < k; i++) V[i * k + i] = 1;
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) off += A[i * k + j] * A[i * k + j];
      if (off < 1e-22) break;
      for (let p = 0; p < k; p++) {
        for (let q = p + 1; q < k; q++) {
          const apq = A[p * k + q];
          if (Math.abs(apq) < 1e-300) continue;
          const theta = (A[q * k + q] - A[p * k + p]) / (2 * apq);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let r = 0; r < k; r++) {
            const arp = A[r * k + p], arq = A[r * k + q];
            A[r * k + p] = c * arp - s * arq; A[r * k + q] = s * arp + c * arq;
          }
          for (let r = 0; r < k; r++) {
            const apr = A[p * k + r], aqr = A[q * k + r];
            A[p * k + r] = c * apr - s * aqr; A[q * k + r] = s * apr + c * aqr;
          }
          for (let r = 0; r < k; r++) {
            const vrp = V[r * k + p], vrq = V[r * k + q];
            V[r * k + p] = c * vrp - s * vrq; V[r * k + q] = s * vrp + c * vrq;
          }
        }
      }
    }
    const vals = new Float64Array(k);
    for (let i = 0; i < k; i++) vals[i] = A[i * k + i];
    return { vals, V };
  }

  // ------------------------------------------------------------------
  // 模型
  // ------------------------------------------------------------------
  function olsCoef(sys, idx) {
    const { A, b, k } = subMatrix(sys, idx);
    return solve(A, b, k);
  }

  /** 驗證集 SSE：以訓練系統的平均與標準差轉換驗證集統計量（不需逐列計算） */
  function valSSE(vst, sys, idx, beta) {
    const n = vst.n, my = sys.my, p = vst.p, k = idx.length;
    let sse = vst.syy - 2 * my * vst.sy + n * my * my;
    for (let a = 0; a < k; a++) {
      const j = idx[a];
      const suz = (vst.sxy[j] - sys.m[j] * vst.sy - my * vst.sx[j] + n * sys.m[j] * my) / sys.sd[j];
      sse -= 2 * beta[a] * suz;
      for (let d = 0; d < k; d++) {
        const l = idx[d];
        const szz = (vst.sxx[j * p + l] - sys.m[j] * vst.sx[l] - sys.m[l] * vst.sx[j] + n * sys.m[j] * sys.m[l]) / (sys.sd[j] * sys.sd[l]);
        sse += beta[a] * beta[d] * szz;
      }
    }
    return Math.max(sse, 0);
  }

  function r2FromStats(vst, sse) {
    const sst = vst.syy - (vst.sy * vst.sy) / vst.n;
    return sst > 0 ? 1 - sse / sst : (sse < 1e-12 ? 1 : 0);
  }

  function ridgeCoef(sys, idx, alpha) {
    const { A, b, k } = subMatrix(sys, idx);
    for (let i = 0; i < k; i++) A[i * k + i] += alpha;
    return solve(A, b, k);
  }

  function softThreshold(v, t) { return v > t ? v - t : v < -t ? v + t : 0; }

  /** Lasso 座標下降：min (1/2n)||ỹ − Zb||² + α||b||₁（同 sklearn） */
  function lassoCoef(sys, idx, alpha, warm) {
    const { A, b: cvec, k } = subMatrix(sys, idx);
    const n = sys.n;
    const beta = warm ? Float64Array.from(warm) : new Float64Array(k);
    const grad = new Float64Array(k); // A·beta
    for (let i = 0; i < k; i++) { let s = 0; for (let j = 0; j < k; j++) s += A[i * k + j] * beta[j]; grad[i] = s; }
    for (let iter = 0; iter < 5000; iter++) {
      let maxDelta = 0, maxBeta = 0;
      for (let j = 0; j < k; j++) {
        const ajj = A[j * k + j];
        if (ajj <= 0) continue;
        const rho = (cvec[j] - grad[j] + ajj * beta[j]) / n;
        const nb = softThreshold(rho, alpha) / (ajj / n);
        const delta = nb - beta[j];
        if (delta !== 0) {
          for (let i = 0; i < k; i++) grad[i] += A[i * k + j] * delta;
          beta[j] = nb;
        }
        maxDelta = Math.max(maxDelta, Math.abs(delta));
        maxBeta = Math.max(maxBeta, Math.abs(nb));
      }
      if (maxDelta <= 1e-7 * Math.max(maxBeta, 1e-12)) break;
    }
    return beta;
  }

  function lassoAlphaGrid(sys, idx, nAlphas = 100, eps = 1e-3) {
    let amax = 0;
    for (const j of idx) amax = Math.max(amax, Math.abs(sys.c[j]) / sys.n);
    if (amax <= 0) return [1e-8];
    const out = [];
    const lo = Math.log10(amax * eps), hi = Math.log10(amax);
    for (let i = 0; i < nAlphas; i++) out.push(Math.pow(10, hi - ((hi - lo) * i) / (nAlphas - 1)));
    return out;
  }

  /**
   * LassoCV：在 trainStats（已用 sdScale 標準化）上做 K-fold 選 α
   * foldStats：各驗證折的統計量（相對於同一組訓練列）
   */
  function lassoCV(trainStats, foldStats, idx, sdScale) {
    const full = buildSystem(trainStats, sdScale);
    const alphas = lassoAlphaGrid(full, idx);
    const mse = new Float64Array(alphas.length);
    for (const vst of foldStats) {
      const sysF = buildSystem(subtractStats(trainStats, vst), sdScale);
      let warm = null;
      for (let a = 0; a < alphas.length; a++) {
        warm = lassoCoef(sysF, idx, alphas[a], warm);
        mse[a] += valSSE(vst, sysF, idx, warm) / vst.n / foldStats.length;
      }
    }
    let best = 0;
    for (let a = 1; a < alphas.length; a++) if (mse[a] < mse[best]) best = a;
    let warm = null;
    for (let a = 0; a <= best; a++) warm = lassoCoef(full, idx, alphas[a], warm);
    return { alpha: alphas[best], beta: warm, sys: full };
  }

  /**
   * RidgeCV（留一法 GCV，同 sklearn RidgeCV 預設）
   * Z: 置中標準化後的訓練列投影，用特徵分解計算 hat 矩陣對角線
   */
  function ridgeGCV(X, y, rows, sys, idx, alphas) {
    const k = idx.length, n = rows.length;
    const { A, b } = subMatrix(sys, idx);
    const { vals, V } = eigSym(A, k);
    const Vtc = new Float64Array(k);
    for (let q = 0; q < k; q++) { let s = 0; for (let j = 0; j < k; j++) s += V[j * k + q] * b[j]; Vtc[q] = s; }
    const P = new Float64Array(n * k), yc = new Float64Array(n), z = new Float64Array(k);
    for (let r = 0; r < n; r++) {
      const i = rows[r];
      yc[r] = y[i] - sys.my;
      for (let a = 0; a < k; a++) z[a] = (X[idx[a]][i] - sys.m[idx[a]]) / sys.sd[idx[a]];
      for (let q = 0; q < k; q++) { let s = 0; for (let a = 0; a < k; a++) s += z[a] * V[a * k + q]; P[r * k + q] = s; }
    }
    let bestAlpha = alphas[0], bestErr = Infinity;
    const w = new Float64Array(k), wc = new Float64Array(k);
    for (const alpha of alphas) {
      for (let q = 0; q < k; q++) { w[q] = 1 / (vals[q] + alpha); wc[q] = w[q] * Vtc[q]; }
      let err = 0;
      for (let r = 0; r < n; r++) {
        let fit = 0, h = 1 / n;
        const off = r * k;
        for (let q = 0; q < k; q++) { const pv = P[off + q]; fit += pv * wc[q]; h += pv * pv * w[q]; }
        const e = (yc[r] - fit) / (1 - h);
        err += e * e;
      }
      if (err < bestErr) { bestErr = err; bestAlpha = alpha; }
    }
    return { alpha: bestAlpha, beta: ridgeCoef(sys, idx, bestAlpha) };
  }

  function rankKBest(sys, idx) {
    // f_regression 的 F 值與相關係數平方單調相關
    const score = idx.map((j) => {
      const r2 = (sys.c[j] * sys.c[j]) / (sys.G[j * sys.p + j] * sys.syyc || 1);
      return { j, s: r2 >= 1 ? Infinity : r2 / (1 - r2) };
    });
    return score.sort((a, b) => b.s - a.s || b.j - a.j).map((o) => o.j);
  }

  /** RFE：每次刪除標準化係數絕對值最小的特徵，回傳各 k 的特徵集合 */
  function rfeSets(sys, idx, kList) {
    const want = new Set(kList), out = {};
    let remaining = idx.slice();
    while (remaining.length >= 1) {
      if (want.has(remaining.length)) out[remaining.length] = remaining.slice();
      if (remaining.length === 1) break;
      const beta = olsCoef(sys, remaining);
      let worst = 0;
      for (let a = 1; a < beta.length; a++) if (Math.abs(beta[a]) < Math.abs(beta[worst])) worst = a;
      remaining.splice(worst, 1);
    }
    return out;
  }

  function topByAbs(idx, beta, k) {
    return idx.map((j, a) => ({ j, v: Math.abs(beta[a]), a }))
      .sort((x, y) => y.v - x.v || x.a - y.a).slice(0, k).map((o) => o.j).sort((x, y) => x - y);
  }

  // ------------------------------------------------------------------
  // OLS 統計推論（等同 statsmodels OLS，含截距）
  // ------------------------------------------------------------------
  function olsInference(sys, idx) {
    const { A, b, k } = subMatrix(sys, idx);
    const Ainv = invert(A, k);
    const beta = matVec(Ainv, b, k);
    let bc = 0;
    for (let a = 0; a < k; a++) bc += beta[a] * b[a];
    const sse = Math.max(sys.syyc - bc, 0);
    const df = sys.n - k - 1;
    const sigma2 = sse / df;
    const tq = tInv(0.975, df);
    const coef = [], se = [], tval = [], pval = [], lo = [], hi = [];
    let intercept = sys.my;
    for (let a = 0; a < k; a++) {
      const j = idx[a];
      const cj = beta[a] / sys.sd[j];
      const sej = Math.sqrt(Math.max(sigma2 * Ainv[a * k + a], 0)) / sys.sd[j];
      intercept -= cj * sys.m[j];
      coef.push(cj); se.push(sej);
      const t = sej > 0 ? cj / sej : Infinity;
      tval.push(t); pval.push(tTwoSidedP(t, df));
      lo.push(cj - tq * sej); hi.push(cj + tq * sej);
    }
    const r2 = sys.syyc > 0 ? 1 - sse / sys.syyc : 1;
    const adj = 1 - ((1 - r2) * (sys.n - 1)) / df;
    return { idx, beta, Ainv, sys, k, sse, df, sigma2, tq, coef, se, tval, pval, lo, hi, intercept, r2, adj };
  }

  /** 對一列（原始尺度的特徵值陣列，依 idx 順序）計算預測值與 95% CI / PI */
  function olsPredict(inf, xs) {
    const { k, sys, idx, beta, Ainv, sigma2, tq } = inf;
    const z = new Float64Array(k);
    let mean = sys.my;
    for (let a = 0; a < k; a++) { z[a] = (xs[a] - sys.m[idx[a]]) / sys.sd[idx[a]]; mean += beta[a] * z[a]; }
    let q = 1 / sys.n;
    for (let a = 0; a < k; a++) { let s = 0; for (let d = 0; d < k; d++) s += Ainv[a * k + d] * z[d]; q += z[a] * s; }
    const seMean = Math.sqrt(Math.max(sigma2 * q, 0));
    const seObs = Math.sqrt(Math.max(sigma2 * (1 + q), 0));
    return { mean, ciLo: mean - tq * seMean, ciHi: mean + tq * seMean, piLo: mean - tq * seObs, piHi: mean + tq * seObs };
  }

  /** 線性模型（標準化係數 beta，系統 sys）對 rows 的預測 */
  function predictRows(X, rows, sys, idx, beta) {
    const out = new Float64Array(rows.length);
    for (let r = 0; r < rows.length; r++) {
      const i = rows[r];
      let v = sys.my;
      for (let a = 0; a < idx.length; a++) v += beta[a] * (X[idx[a]][i] - sys.m[idx[a]]) / sys.sd[idx[a]];
      out[r] = v;
    }
    return out;
  }

  // ------------------------------------------------------------------
  // 機率分布
  // ------------------------------------------------------------------
  function logGamma(x) {
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
      -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    x -= 1;
    let a = c[0];
    const t = x + g + 0.5;
    for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  function betacf(a, b, x) {
    const MAXIT = 300, EPS = 3e-16, FPMIN = 1e-300;
    let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }

  function ibeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    if (x < (a + 1) / (a + b + 2)) return (bt * betacf(a, b, x)) / a;
    return 1 - (bt * betacf(b, a, 1 - x)) / b;
  }

  function erfc(x) {
    // Numerical Recipes erfc (相對誤差 < 1.2e-7)
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
      t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }

  function normCdf(x) { return 0.5 * erfc(-x / Math.SQRT2); }

  function normInv(p) {
    // Acklam 演算法
    const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
    const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    const pl = 0.02425;
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    let q, r;
    if (p < pl) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - pl) {
      q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    q = p - 0.5; r = q * q;
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  function tCdf(t, df) {
    if (!isFinite(t)) return t > 0 ? 1 : 0;
    if (df > 1e5) return normCdf(t);
    const x = df / (df + t * t);
    const tail = 0.5 * ibeta(x, df / 2, 0.5);
    return t > 0 ? 1 - tail : tail;
  }

  function tTwoSidedP(t, df) {
    if (!isFinite(t)) return 0;
    if (df > 1e5) return erfc(Math.abs(t) / Math.SQRT2);
    return ibeta(df / (df + t * t), df / 2, 0.5);
  }

  function tPdf(t, df) {
    return Math.exp(logGamma((df + 1) / 2) - logGamma(df / 2) - 0.5 * Math.log(df * Math.PI) - ((df + 1) / 2) * Math.log(1 + (t * t) / df));
  }

  function tInv(p, df) {
    const z = normInv(p);
    if (df > 1e5) return z;
    // Cornish-Fisher 起點 + Newton
    let t = z + (z * z * z + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df * df);
    for (let i = 0; i < 50; i++) {
      const f = tCdf(t, df) - p;
      const d = tPdf(t, df);
      if (!(d > 0)) break;
      const step = f / d;
      t -= step;
      if (Math.abs(step) < 1e-12 * Math.max(1, Math.abs(t))) break;
    }
    return t;
  }

  function gammaQ(a, x) {
    // 正規化上不完全 Gamma
    if (x <= 0) return 1;
    const gln = logGamma(a);
    if (x < a + 1) {
      let sum = 1 / a, del = sum, ap = a;
      for (let n = 0; n < 1000; n++) { ap += 1; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; }
      return 1 - sum * Math.exp(-x + a * Math.log(x) - gln);
    }
    let b = x + 1 - a, c = 1 / 1e-300, d = 1 / b, h = d;
    for (let i = 1; i < 1000; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
      c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return Math.exp(-x + a * Math.log(x) - gln) * h;
  }

  function chi2Sf(x, k) { return gammaQ(k / 2, x / 2); }

  // ------------------------------------------------------------------
  // 描述統計與診斷
  // ------------------------------------------------------------------
  function mean(a) { let s = 0; for (const v of a) s += v; return s / a.length; }

  function moments(a) {
    const m = mean(a);
    let m2 = 0, m3 = 0, m4 = 0;
    for (const v of a) { const d = v - m; const d2 = d * d; m2 += d2; m3 += d2 * d; m4 += d2 * d2; }
    const n = a.length;
    m2 /= n; m3 /= n; m4 /= n;
    return { mean: m, sdPop: Math.sqrt(m2), sd: Math.sqrt((m2 * n) / Math.max(n - 1, 1)), skew: m3 / Math.pow(m2, 1.5), kurt: m4 / (m2 * m2) - 3 };
  }

  function quantileSorted(s, q) {
    // 同 pandas 線性內插
    if (!s.length) return NaN;
    const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return s[lo] + (s[hi] - s[lo]) * (pos - lo);
  }

  function durbinWatson(e) {
    let num = 0, den = 0;
    for (let i = 0; i < e.length; i++) { den += e[i] * e[i]; if (i) num += (e[i] - e[i - 1]) ** 2; }
    return num / den;
  }

  /** Breusch-Pagan（statsmodels het_breuschpagan 預設）：以 e² 對 X 回歸，LM = n·R² */
  function breuschPagan(X, rows, idx, e) {
    const k = idx.length;
    const cols = idx.map((j) => X[j]);
    const e2 = new Float64Array(rows.length);
    const local = new Int32Array(rows.length);
    const Xl = cols.map(() => new Float64Array(rows.length));
    for (let r = 0; r < rows.length; r++) {
      e2[r] = e[r] * e[r]; local[r] = r;
      for (let a = 0; a < k; a++) Xl[a][r] = cols[a][rows[r]];
    }
    const st = suffStats(Xl, e2, local);
    const sys = buildSystem(st);
    const all = Array.from({ length: k }, (_, a) => a);
    const beta = olsCoef(sys, all);
    let bc = 0;
    for (let a = 0; a < k; a++) bc += beta[a] * sys.c[a];
    const r2 = sys.syyc > 0 ? bc / sys.syyc : 0;
    const lm = rows.length * r2;
    return { lm, p: chi2Sf(lm, k) };
  }

  /** VIF：相關係數矩陣反矩陣的對角線（等同 statsmodels 含截距的 VIF） */
  function vif(sys, idx) {
    const k = idx.length;
    if (k < 2) return [1];
    const R = new Float64Array(k * k);
    for (let a = 0; a < k; a++) for (let d = 0; d < k; d++) {
      const ja = idx[a], jd = idx[d];
      R[a * k + d] = sys.G[ja * sys.p + jd] / Math.sqrt(sys.G[ja * sys.p + ja] * sys.G[jd * sys.p + jd]);
    }
    const Ri = invert(R, k);
    return Array.from({ length: k }, (_, a) => Ri[a * k + a]);
  }

  /** Q-Q plot 理論分位數（同 scipy.stats.probplot 的 Filliben 估計） */
  function probplotX(n) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let u;
      if (i === n - 1) u = Math.pow(0.5, 1 / n);
      else if (i === 0) u = 1 - Math.pow(0.5, 1 / n);
      else u = (i + 1 - 0.3175) / (n + 0.365);
      out[i] = normInv(i === n - 1 ? u : i === 0 ? u : u);
    }
    out[n - 1] = normInv(Math.pow(0.5, 1 / n));
    out[0] = normInv(1 - Math.pow(0.5, 1 / n));
    return out;
  }

  const api = {
    mulberry32, permutation, kfold, suffStats, subtractStats, buildSystem, invert, solve, eigSym,
    olsCoef, valSSE, r2FromStats, ridgeCoef, lassoCoef, lassoAlphaGrid, lassoCV, ridgeGCV, rankKBest, rfeSets, topByAbs,
    olsInference, olsPredict, predictRows, tInv, tCdf, tTwoSidedP, normInv, normCdf, chi2Sf,
    mean, moments, quantileSorted, durbinWatson, breuschPagan, vif, probplotX,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Stats = api;
})(typeof self !== "undefined" ? self : this);
