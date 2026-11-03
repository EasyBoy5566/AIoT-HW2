"""
HW2：多元線性回歸 × CRISP-DM 互動網頁
學號 5115056015　鍾易憲

功能
- 下拉選單切換兩種 problem：洪水機率預測（Kaggle S4E5）／新創公司利潤預測（50 Startups）
- 可輸入任意 CSV 網址（或上傳 CSV）作為 Dataset，自動處理缺失值與類別編碼
- 依 CRISP-DM 六階段呈現：特徵選擇比較、模型評估、95% 信賴區間與預測區間、殘差診斷、即時預測
"""

from pathlib import Path
import io
import re
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_regression
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

st.set_page_config(page_title="HW2 多元線性回歸｜CRISP-DM", page_icon="📈", layout="wide")
st.markdown(
    """<style>
    .kpi-row{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 14px 0;}
    .kpi{flex:1 1 150px;min-width:140px;background:#f5f8fc;border:1px solid #e1e8f0;border-radius:10px;padding:10px 14px;}
    .kpi .lab{font-size:0.82rem;color:#55606b;margin-bottom:2px;}
    .kpi .val{font-size:1.45rem;font-weight:600;color:#1f2d3d;white-space:nowrap;}
    </style>""",
    unsafe_allow_html=True,
)


def kpi_row(items):
    """以可自動換行的卡片顯示指標，避免窄螢幕時數字被截斷"""
    import html
    cards = "".join(
        f"<div class='kpi'><div class='lab'>{html.escape(str(k))}</div><div class='val'>{html.escape(str(v))}</div></div>"
        for k, v in items
    )
    st.markdown(f"<div class='kpi-row'>{cards}</div>", unsafe_allow_html=True)

BASE = Path(__file__).parent
IS_BROWSER = sys.platform == "emscripten"  # GitHub Pages 版本（stlite / Pyodide，在瀏覽器執行 Python）
BROWSER_FLOOD_ROWS = 20000                  # 瀏覽器運算較慢，洪水資料再抽樣 20,000 筆
PAGES_URL = "https://easyboy5566.github.io/AIoT-HW2/"
REPO = "https://github.com/EasyBoy5566/AIoT-HW2"
STARTUPS_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
ACCENT = "#2E6FB7"
ORANGE = "#E8833A"

PROBLEMS = {
    "Problem 1：洪水機率預測": {
        "key": "flood",
        "path": BASE / "data" / "flood_train_sample.csv.gz",
        "target": "FloodProbability",
        "name": "Regression with a Flood Prediction Dataset（Kaggle Playground Series S4E5）",
        "link": "https://www.kaggle.com/competitions/playground-series-s4e5",
        "note": "原始 train.csv 共 1,117,957 筆；網頁版使用 random_state=42 隨機抽樣的 50,000 筆，"
                "完整資料的分析結果見 Notebook 與報告。",
        "business": [
            "**問題**：根據 20 項環境、氣候與人為風險因子（季風強度、排水系統、都市化、森林砍伐等，"
            "每項為 0～18 的整數分數），預測一個地區的洪水發生機率 `FloodProbability`。",
            "**價值**：協助政府與防災單位找出高風險區域與關鍵風險因子，作為防災資源配置、"
            "排水與河川治理優先順序、基礎設施改善的分析參考。",
            "**成功標準**：Kaggle 以 R² 評分；本研究目標是建立可解釋的多元線性回歸模型，R² 達 0.84 以上。",
            "**限制**：資料為 Kaggle 合成資料，係數與風險排序僅供方法研究，不能直接作為真實洪水預警依據。",
        ],
    },
    "Problem 2：新創公司利潤預測": {
        "key": "startups",
        "path": BASE / "data" / "50_Startups.csv",
        "target": "Profit",
        "name": "50 Startups（100-Days-Of-ML-Code）",
        "link": STARTUPS_URL,
        "note": "共 50 家新創公司，特徵為研發支出、行政支出、行銷支出與所在州別（類別變數，自動做 One-hot 編碼）。",
        "business": [
            "**問題**：根據新創公司的研發支出（R&D Spend）、行政支出（Administration）、"
            "行銷支出（Marketing Spend）與所在州別（State），預測公司年度利潤 `Profit`。",
            "**價值**：協助創投評估投資標的的獲利潛力，也幫助經營者了解哪一類支出與利潤最相關，作為預算分配參考。",
            "**成功標準**：測試集 R² 越高越好，並能指出影響利潤的關鍵支出項目。",
            "**限制**：樣本只有 50 筆，交叉驗證與測試集結果的變異較大，結論需謹慎解讀。",
        ],
    },
    "自訂 Dataset（CSV 網址）": None,
}

FLOOD_EXTERNAL = pd.DataFrame([
    ["本研究：20 特徵多元線性回歸（Ridge / OLS）", "測試 R² 0.8449；Kaggle Private 0.84426", "訓練快、係數可解釋", "無法捕捉非線性與交互作用"],
    ["特徵加總 feature_sum + 線性回歸", "測試 R² 0.8449（只用 1 個特徵）", "證實目標主要由 20 特徵總和決定", "與原始特徵同時使用會完全共線"],
    ["列統計特徵（sum / mean / std / skew…）", "Kaggle 高分討論核心技巧", "提供分布形狀與非線性資訊", "特徵冗餘，需交叉驗證確認增益"],
    ["LightGBM / CatBoost + 統計特徵", "CV R² 約 0.869～0.870", "可學習非線性與局部交互作用", "需調參、可解釋性較低"],
    ["Ensemble / Blending（第 2 名解法）", "Public 0.86943 / Private 0.86902", "降低單一模型誤差", "模型繁多，訓練與維護成本高"],
], columns=["方法", "代表成績", "優點", "限制"])


# ------------------------------------------------------------------
# 資料讀取
# ------------------------------------------------------------------
def to_raw_url(url: str) -> str:
    """把 GitHub 網頁連結轉成 raw 檔案連結"""
    url = url.strip()
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return url


@st.cache_data(show_spinner=False)
def load_local(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def fetch_text(url: str) -> str:
    """下載 CSV 文字：本機／Streamlit Cloud 用 urllib，瀏覽器版用 pyodide.http.open_url"""
    if IS_BROWSER:
        from pyodide.http import open_url  # type: ignore
        return open_url(url).read()
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8-sig", errors="replace")


@st.cache_data(show_spinner="下載資料中…", ttl=3600)
def load_url(url: str) -> pd.DataFrame:
    text = fetch_text(url)
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] == 1:  # 可能是分號或 tab 分隔
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    return df


@st.cache_data(show_spinner=False)
def load_upload(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    if df.shape[1] == 1:
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
    return df


def guess_id_columns(df: pd.DataFrame) -> list:
    ids = []
    for c in df.columns:
        name = str(c).strip().lower()
        if name in {"id", "index", "unnamed: 0", "no", "no."}:
            ids.append(c)
        elif pd.api.types.is_integer_dtype(df[c]) and df[c].is_unique and df[c].is_monotonic_increasing and len(df) > 20:
            ids.append(c)
    return ids


# ------------------------------------------------------------------
# 前處理：只用訓練集學習中位數、眾數與類別
# ------------------------------------------------------------------
class Preprocessor:
    def __init__(self, num_cols, cat_cols):
        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)

    def fit(self, X: pd.DataFrame):
        self.medians = X[self.num_cols].median()
        self.modes = {c: X[c].mode(dropna=True).iloc[0] if X[c].notna().any() else "missing" for c in self.cat_cols}
        self.columns = self._encode(X).columns.tolist()
        return self

    def _encode(self, X):
        out = X[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(self.medians)
        if self.cat_cols:
            cats = X[self.cat_cols].copy()
            for c in self.cat_cols:
                cats[c] = cats[c].fillna(self.modes[c]).astype(str)
            dummies = pd.get_dummies(cats, drop_first=True, dtype=float)
            out = pd.concat([out, dummies], axis=1)
        return out.astype(float)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._encode(X).reindex(columns=self.columns, fill_value=0.0)


def k_grid(p: int) -> list:
    if p <= 10:
        return list(range(1, p + 1))
    grid = sorted(set(np.linspace(1, p, 8).round().astype(int).tolist()) | {p})
    return grid


def make_selector(method: str, k: int):
    if method == "SelectKBest":
        return SelectKBest(f_regression, k=k)
    if method == "RFE":
        return RFE(LinearRegression(), n_features_to_select=k)
    return SelectFromModel(LassoCV(cv=3, max_iter=5000, random_state=42),
                           max_features=k, threshold=-np.inf)


@st.cache_data(show_spinner="特徵選擇：5-fold 交叉驗證比較中…")
def compare_feature_selection(Xtr: pd.DataFrame, ytr: pd.Series, seed: int) -> pd.DataFrame:
    p = Xtr.shape[1]
    n_splits = 5 if len(Xtr) >= 25 else 3
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for method in ["SelectKBest", "RFE", "LassoCV"]:
        for k in k_grid(p):
            pipe = Pipeline([("scaler", StandardScaler()), ("select", make_selector(method, k)),
                             ("model", LinearRegression())])
            scores = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="r2")
            rows.append({"Method": method, "K": k, "Mean R2": scores.mean(),
                         "Std R2": scores.std(ddof=1), "SE R2": scores.std(ddof=1) / np.sqrt(n_splits)})
    return pd.DataFrame(rows)


def recommend(results: pd.DataFrame):
    best = results.loc[results["Mean R2"].idxmax()]
    cand = results[results["Mean R2"] >= best["Mean R2"] - best["SE R2"]]
    rec = cand.sort_values(["K", "Mean R2"], ascending=[True, False]).iloc[0]
    return rec["Method"], int(rec["K"])


def selected_features(Xtr, ytr, method, k):
    if method == "全部特徵" or k >= Xtr.shape[1]:
        return Xtr.columns.tolist()
    scaled = StandardScaler().fit_transform(Xtr)
    sel = make_selector(method, k).fit(scaled, ytr)
    return Xtr.columns[sel.get_support()].tolist()


def metrics(y, pred, p):
    n = len(y)
    r2 = r2_score(y, pred)
    mse = mean_squared_error(y, pred)
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan
    return {"MAE": mean_absolute_error(y, pred), "MSE": mse, "RMSE": np.sqrt(mse), "R²": r2, "Adjusted R²": adj}


@st.cache_data(show_spinner="訓練模型中…")
def train_models(Xtr, ytr, Xte, yte, feats, seed):
    p = len(feats)
    cv = KFold(n_splits=5 if len(Xtr) >= 25 else 3, shuffle=True, random_state=seed)
    alphas = np.logspace(-4, 3, 40)
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": RidgeCV(alphas=alphas),
        "Lasso": LassoCV(cv=cv, max_iter=10000, random_state=seed),
    }
    table, preds, pipes = [], {}, {}
    for name, est in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)]).fit(Xtr[feats], ytr)
        pr = pipe.predict(Xte[feats])
        preds[name] = pr
        pipes[name] = pipe
        alpha = getattr(pipe.named_steps["model"], "alpha_", np.nan)
        table.append({"Model": name, **metrics(yte, pr, p), "Alpha": alpha})
    ols = sm.OLS(ytr, sm.add_constant(Xtr[feats], has_constant="add")).fit()
    return pd.DataFrame(table), preds, pipes, ols


def fmt(x, digits=4):
    if pd.isna(x):
        return "—"
    ax = abs(x)
    if ax >= 1e4:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:,.2f}"
    return f"{x:.{digits}f}"


# ------------------------------------------------------------------
# 側邊欄
# ------------------------------------------------------------------
st.sidebar.title("📈 HW2 多元線性回歸")
st.sidebar.caption("CRISP-DM × Feature Selection × 預測區間")

choice = st.sidebar.selectbox("選擇 Problem / Dataset", list(PROBLEMS.keys()))
cfg = PROBLEMS[choice]

if cfg is None:
    st.sidebar.markdown("**自訂 Dataset**")
    url = st.sidebar.text_input("CSV 網址（支援 GitHub 網頁連結）", value=STARTUPS_URL)
    upload = st.sidebar.file_uploader("或上傳 CSV 檔", type=["csv"])
    try:
        if upload is not None:
            df = load_upload(upload.getvalue())
            src_name, src_link = upload.name, None
        else:
            raw = to_raw_url(url)
            df = load_url(raw)
            src_name, src_link = raw.rsplit("/", 1)[-1], raw
    except Exception as e:  # noqa: BLE001
        msg = f"無法讀取資料：{e}"
        if IS_BROWSER:
            msg += "\n\n網頁版在瀏覽器中執行，只能讀取允許跨網域存取（CORS）的網址，例如 raw.githubusercontent.com；也可以改用上方的「上傳 CSV」。"
        st.error(msg)
        st.stop()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.error("資料中沒有數值欄位，無法進行回歸。")
        st.stop()
    target = st.sidebar.selectbox("目標欄位（y）", numeric_cols, index=len(numeric_cols) - 1)
    cfg = {"key": "custom", "target": target, "name": src_name, "link": src_link,
           "note": "使用者自訂資料集：系統自動排除 ID 欄位、以中位數／眾數填補缺失值，類別欄位做 One-hot 編碼。",
           "business": [f"**問題**：以資料集中的其他欄位預測 `{target}`。",
                        "**做法**：依 CRISP-DM 流程完成資料理解、前處理、特徵選擇、多元線性回歸建模與評估。",
                        "**成功標準**：測試集 R² 越高、RMSE 越低越好，並找出對目標影響最大的特徵。"]}
else:
    df = load_local(str(cfg["path"]))
    if IS_BROWSER and cfg["key"] == "flood" and len(df) > BROWSER_FLOOD_ROWS:
        df = df.sample(n=BROWSER_FLOOD_ROWS, random_state=42).sort_index()
        cfg = {**cfg, "note": cfg["note"].replace("50,000 筆", f"50,000 筆，瀏覽器版再抽樣 {BROWSER_FLOOD_ROWS:,} 筆")}
    target = cfg["target"]

id_guess = [c for c in guess_id_columns(df) if c != target]
drop_cols = st.sidebar.multiselect("排除的欄位（ID 等）", [c for c in df.columns if c != target], default=id_guess)

st.sidebar.divider()
test_size = st.sidebar.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)
seed = int(st.sidebar.number_input("random_state", value=42, step=1))
fs_choice = st.sidebar.selectbox("特徵選擇方法", ["自動推薦（1-SE 規則）", "SelectKBest", "RFE", "LassoCV", "全部特徵"])
final_model_name = st.sidebar.selectbox("最終模型", ["LinearRegression", "Ridge", "Lasso"])
st.sidebar.divider()
st.sidebar.markdown(f"[GitHub repo]({REPO})　·　學號 5115056015")
if IS_BROWSER:
    st.sidebar.caption("GitHub Pages 版本：Python 在你的瀏覽器中執行（stlite），切換資料集時需要等待數秒～數十秒。")

# ------------------------------------------------------------------
# 資料準備
# ------------------------------------------------------------------
data = df.drop(columns=drop_cols).copy()
data[target] = pd.to_numeric(data[target], errors="coerce")
n_missing_target = int(data[target].isna().sum())
data = data.dropna(subset=[target])
n_dup = int(data.duplicated().sum())
data = data.drop_duplicates()

X_raw = data.drop(columns=[target])
y = data[target]
num_cols = [c for c in X_raw.columns if pd.api.types.is_numeric_dtype(X_raw[c])]
other = [c for c in X_raw.columns if c not in num_cols]
cat_cols = [c for c in other if X_raw[c].nunique() <= 30]
high_card = [c for c in other if c not in cat_cols]
X_raw = X_raw[num_cols + cat_cols]

if X_raw.shape[1] == 0 or len(data) < 10:
    st.error("可用的特徵或資料筆數不足（至少需要 1 個特徵與 10 筆資料）。")
    st.stop()

Xtr_raw, Xte_raw, ytr, yte = train_test_split(X_raw, y, test_size=test_size, random_state=seed)
prep = Preprocessor(num_cols, cat_cols).fit(Xtr_raw)
Xtr, Xte = prep.transform(Xtr_raw), prep.transform(Xte_raw)
# 移除訓練集中為常數的欄位
const_cols = [c for c in Xtr.columns if Xtr[c].nunique() <= 1]
Xtr, Xte = Xtr.drop(columns=const_cols), Xte.drop(columns=const_cols)

fs_results = compare_feature_selection(Xtr, ytr, seed)
rec_method, rec_k = recommend(fs_results)
if fs_choice == "自動推薦（1-SE 規則）":
    method, k = rec_method, rec_k
elif fs_choice == "全部特徵":
    method, k = "全部特徵", Xtr.shape[1]
else:
    method = fs_choice
    k = st.sidebar.slider("選取特徵數 k", 1, Xtr.shape[1], min(rec_k, Xtr.shape[1]))
feats = selected_features(Xtr, ytr, method, k)

model_table, model_preds, model_pipes, ols = train_models(Xtr, ytr, Xte, yte, feats, seed)
final_pred = model_preds[final_model_name]
final_metrics = model_table.set_index("Model").loc[final_model_name]

Xte_ols = sm.add_constant(Xte[feats], has_constant="add")
pred_frame = ols.get_prediction(Xte_ols).summary_frame(alpha=0.05)
coverage = float(((yte.values >= pred_frame["obs_ci_lower"].values) & (yte.values <= pred_frame["obs_ci_upper"].values)).mean())

# ------------------------------------------------------------------
# 頁首
# ------------------------------------------------------------------
st.title("多元線性回歸 × CRISP-DM")
link_md = f"[{cfg['name']}]({cfg['link']})" if cfg.get("link") else cfg["name"]
st.markdown(f"**Dataset**：{link_md}　｜　**目標**：`{target}`")
st.caption(cfg["note"])

kpi_row([
    ("資料筆數", f"{len(data):,}"),
    ("原始特徵數", f"{X_raw.shape[1]}"),
    ("選用特徵數", f"{len(feats)} / {Xtr.shape[1]}"),
    (f"測試 R²（{final_model_name}）", f"{final_metrics['R²']:.4f}"),
    ("95% PI 涵蓋率", f"{coverage:.1%}"),
])

tabs = st.tabs(["1 Business Understanding", "2 Data Understanding", "3 Data Preparation",
                "4 Modeling", "5 Evaluation", "6 Deployment", "7 網路解法與 AI 輔助"])

# ---------------- 1 Business Understanding ----------------
with tabs[0]:
    st.subheader("Business Understanding")
    for line in cfg["business"]:
        st.markdown(f"- {line}")
    st.markdown("**CRISP-DM 流程**")
    st.markdown(
        "1. Business Understanding → 2. Data Understanding → 3. Data Preparation → "
        "4. Modeling（特徵選擇 + 多元線性回歸）→ 5. Evaluation（指標、預測區間、殘差診斷）→ 6. Deployment（即時預測）"
    )

# ---------------- 2 Data Understanding ----------------
with tabs[1]:
    st.subheader("Data Understanding")
    a, b, c = st.columns(3)
    a.metric("列數 × 欄數（原始）", f"{df.shape[0]:,} × {df.shape[1]}")
    b.metric("缺失值總數", f"{int(df.isna().sum().sum()):,}")
    c.metric("重複列", f"{int(df.drop(columns=drop_cols).duplicated().sum()):,}")
    st.markdown("**前 10 筆資料**")
    st.dataframe(df.head(10), hide_index=True)
    info = pd.DataFrame({"型態": df.dtypes.astype(str), "缺失值": df.isna().sum(), "唯一值": df.nunique()})
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**欄位資訊**")
        st.dataframe(info)
    with right:
        st.markdown("**描述統計**")
        st.dataframe(df.describe().T.round(4))

    g1, g2 = st.columns(2)
    with g1:
        fig = px.histogram(data, x=target, nbins=50, color_discrete_sequence=[ACCENT],
                           title=f"目標變數分布：{target}")
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig)
    with g2:
        corr_df = pd.concat([Xtr, ytr.rename(target)], axis=1)
        tc = corr_df.corr()[target].drop(target).sort_values()
        fig = px.bar(x=tc.values, y=tc.index, orientation="h", title=f"各特徵與 {target} 的相關係數",
                     color=tc.values, color_continuous_scale="RdBu", range_color=[-1, 1],
                     labels={"x": "Pearson r", "y": ""})
        fig.update_layout(height=max(380, 22 * len(tc) + 120), coloraxis_showscale=False)
        st.plotly_chart(fig)

    cm = corr_df.iloc[:, -31:].corr() if corr_df.shape[1] > 31 else corr_df.corr()
    fig = px.imshow(cm, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto",
                    text_auto=".2f" if cm.shape[0] <= 22 else False, title="相關係數熱力圖（Pearson）")
    fig.update_layout(height=max(450, 28 * cm.shape[0] + 150))
    st.plotly_chart(fig)
    st.caption("相關係數以訓練集計算；類別欄位以 One-hot 編碼後的欄位呈現。")

# ---------------- 3 Data Preparation ----------------
with tabs[2]:
    st.subheader("Data Preparation")
    steps = pd.DataFrame([
        ["排除欄位", ", ".join(map(str, drop_cols)) or "無", "ID 等識別欄位不具預測意義"],
        ["目標缺失", f"刪除 {n_missing_target} 筆", "監督式學習需要目標值"],
        ["重複資料", f"刪除 {n_dup} 筆", "避免重複計算"],
        ["高基數文字欄位", ", ".join(map(str, high_card)) or "無", "唯一值 > 30 的文字欄位不納入"],
        ["訓練／測試切分", f"{1 - test_size:.0%} / {test_size:.0%}（random_state={seed}）→ {len(Xtr):,} / {len(Xte):,} 筆", "保留獨立測試集"],
        ["數值缺失值", "以訓練集中位數填補", "只用訓練集計算，避免資料洩漏"],
        ["類別欄位", (", ".join(map(str, cat_cols)) + "：眾數填補 + One-hot（drop_first）") if cat_cols else "無", "drop_first 避免虛擬變數陷阱"],
        ["標準化", "StandardScaler 放在每個模型 Pipeline 內", "CV 每一折各自 fit，避免資料洩漏"],
    ], columns=["步驟", "處理結果", "原因"])
    st.dataframe(steps, hide_index=True)
    st.markdown(f"**編碼後特徵（{Xtr.shape[1]} 個）**：" + "、".join(f"`{c}`" for c in Xtr.columns))
    if const_cols:
        st.caption("已移除訓練集中為常數的欄位：" + "、".join(const_cols))

# ---------------- 4 Modeling ----------------
with tabs[3]:
    st.subheader("4.1 Feature Selection：三種方法 × 不同特徵數的 CV R²")
    fig = px.line(fs_results, x="K", y="Mean R2", color="Method", markers=True, error_y="SE R2",
                  title="特徵數 vs 交叉驗證平均 R²", labels={"Mean R2": "Mean CV R²", "K": "選取特徵數 k"})
    st.plotly_chart(fig)
    pivot = fs_results.pivot(index="K", columns="Method", values="Mean R2").round(6)
    st.dataframe(pivot)
    st.info(f"**1-SE 規則推薦**：{rec_method}，k = {rec_k}（在最佳平均 R² 一個標準誤內，選特徵最少的組合）。"
            f"目前使用：**{method}，{len(feats)} 個特徵**。")
    st.markdown("**選用的特徵**：" + "、".join(f"`{f}`" for f in feats))

    st.subheader("4.2 模型建立：LinearRegression / Ridge / Lasso / OLS")
    st.dataframe(model_table.round(6), hide_index=True)

    coef = pd.DataFrame({"Feature": feats, "Coefficient": ols.params[feats].values,
                         "Std Error": ols.bse[feats].values, "P-value": ols.pvalues[feats].values,
                         "CI 2.5%": ols.conf_int().loc[feats, 0].values,
                         "CI 97.5%": ols.conf_int().loc[feats, 1].values})
    coef["標準化係數"] = coef["Coefficient"] * Xtr[feats].std().values
    coef = coef.reindex(coef["標準化係數"].abs().sort_values(ascending=False).index)
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**statsmodels OLS 係數**（訓練 R² = {ols.rsquared:.4f}，Adjusted R² = {ols.rsquared_adj:.4f}，"
                    f"截距 = {fmt(ols.params['const'])}）")
        st.dataframe(coef.round(6), hide_index=True)
    with right:
        top = coef.head(15).iloc[::-1]
        fig = px.bar(top, x="標準化係數", y="Feature", orientation="h", title="標準化係數（特徵增加 1 個標準差的影響）",
                     color=np.where(top["標準化係數"] >= 0, "正向", "負向"),
                     color_discrete_map={"正向": ACCENT, "負向": ORANGE})
        fig.update_layout(showlegend=False, height=max(380, 26 * len(top) + 120))
        st.plotly_chart(fig)
    t = coef.iloc[0]
    st.markdown(f"**係數解讀**：影響最大的特徵是 `{t['Feature']}`，在其他特徵固定下，每增加 1 單位，"
                f"預測的 `{target}` 平均{'增加' if t['Coefficient'] >= 0 else '減少'} {fmt(abs(t['Coefficient']), 6)}"
                f"（p-value = {t['P-value']:.3g}）。")
    with st.expander("OLS 完整統計摘要"):
        st.text(ols.summary().as_text())

# ---------------- 5 Evaluation ----------------
with tabs[4]:
    st.subheader("5.1 測試集評估指標")
    kpi_row([
        ("MAE", fmt(final_metrics["MAE"], 6)),
        ("MSE", fmt(final_metrics["MSE"], 6)),
        ("RMSE", fmt(final_metrics["RMSE"], 6)),
        ("R²", f"{final_metrics['R²']:.4f}"),
        ("Adjusted R²", f"{final_metrics['Adjusted R²']:.4f}"),
    ])

    rng = np.random.default_rng(seed)
    st.subheader("5.2 預測圖：95% 信賴區間與預測區間")
    n_max = int(min(300, len(yte)))
    n_show = st.slider("顯示的測試樣本數", 10, n_max, min(100, n_max)) if n_max > 10 else n_max
    idx = rng.choice(len(yte), size=n_show, replace=False)
    band = pred_frame.iloc[idx].copy()
    band["Actual"] = yte.values[idx]
    band = band.sort_values("mean").reset_index(drop=True)
    xs = np.arange(len(band))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[band["obs_ci_upper"], band["obs_ci_lower"][::-1]],
                             fill="toself", fillcolor="rgba(232,131,58,0.22)", line=dict(width=0),
                             name="95% 預測區間（PI）", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[band["mean_ci_upper"], band["mean_ci_lower"][::-1]],
                             fill="toself", fillcolor="rgba(46,111,183,0.45)", line=dict(width=0),
                             name="95% 信賴區間（CI）", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=band["mean"], mode="lines", line=dict(color=ACCENT, width=2.5), name="預測值"))
    fig.add_trace(go.Scatter(x=xs, y=band["Actual"], mode="markers", marker=dict(color="#222", size=6), name="實際值"))
    fig.update_layout(title="OLS 預測值與 95% 信賴區間／預測區間（依預測值排序）",
                      xaxis_title="測試樣本（依預測值排序）", yaxis_title=target, height=500,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig)
    st.caption(f"信賴區間（CI）表示「平均預測值」的不確定性；預測區間（PI）表示「單筆新資料」的範圍，因此較寬。"
               f"完整測試集中有 {coverage:.1%} 的實際值落在 95% 預測區間內。區間由 statsmodels OLS（相同特徵）計算。")

    g1, g2 = st.columns(2)
    sidx = rng.choice(len(yte), size=min(3000, len(yte)), replace=False)
    with g1:
        lo, hi = float(min(yte.min(), final_pred.min())), float(max(yte.max(), final_pred.max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yte.values[sidx], y=final_pred[sidx], mode="markers",
                                 marker=dict(color=ACCENT, size=5, opacity=0.4), name="樣本"))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="red", dash="dash"), name="y = x"))
        fig.update_layout(title=f"實際值 vs 預測值（{final_model_name}）", xaxis_title="實際值", yaxis_title="預測值", height=450)
        st.plotly_chart(fig)
    resid = yte.values - pred_frame["mean"].values
    with g2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_frame["mean"].values[sidx], y=resid[sidx], mode="markers",
                                 marker=dict(color=ACCENT, size=5, opacity=0.4), name="殘差"))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title="殘差 vs 預測值", xaxis_title="預測值", yaxis_title="殘差", height=450, showlegend=False)
        st.plotly_chart(fig)

    st.subheader("5.3 殘差診斷與五大回歸假設")
    g3, g4 = st.columns(2)
    with g3:
        fig = px.histogram(x=resid, nbins=50, color_discrete_sequence=[ACCENT], title="殘差分布",
                           labels={"x": "殘差"})
        st.plotly_chart(fig)
    with g4:
        (osm, osr), (slope, intercept, _) = stats.probplot(resid[sidx], dist="norm")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color=ACCENT, size=5, opacity=0.5), name="殘差"))
        fig.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode="lines", line=dict(color="red"), name="常態參考線"))
        fig.update_layout(title="Normal Q-Q Plot", xaxis_title="理論分位數", yaxis_title="樣本分位數", height=450, showlegend=False)
        st.plotly_chart(fig)

    vx = Xtr[feats].sample(n=min(20000, len(Xtr)), random_state=seed)
    vdesign = sm.add_constant(vx, has_constant="add").to_numpy()
    vif = pd.DataFrame({"Feature": feats,
                        "VIF": [variance_inflation_factor(vdesign, i + 1) for i in range(len(feats))]}) if len(feats) > 1 else pd.DataFrame({"Feature": feats, "VIF": [1.0]})
    vif = vif.sort_values("VIF", ascending=False)
    sk, ku = stats.skew(resid), stats.kurtosis(resid)
    dw = sm.stats.stattools.durbin_watson(ols.resid)
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_n = min(20000, len(ols.resid))
    bp_idx = np.random.default_rng(seed).choice(len(ols.resid), size=bp_n, replace=False)
    bp_p = het_breuschpagan(np.asarray(ols.resid)[bp_idx], ols.model.exog[bp_idx])[1]
    diag = pd.DataFrame([
        ["線性", "殘差 vs 預測值", "殘差應隨機分布在 0 附近、無明顯曲線"],
        ["殘差平均為 0", f"{resid.mean():.6g}", "接近 0 ✓" if (abs(resid.mean()) < 2 * resid.std(ddof=1) / np.sqrt(len(resid)) or abs(resid.mean()) < 0.05 * resid.std(ddof=1)) else "偏離 0（超過 2 個標準誤且大於殘差標準差的 5%）"],
        ["常態性", f"偏態 {sk:.3f}、超額峰度 {ku:.3f}", "接近常態 ✓" if abs(sk) < 1 and abs(ku) < 3 else "偏離常態，需留意"],
        ["誤差獨立", f"Durbin-Watson = {dw:.3f}", "約 2 表示無自相關 ✓" if 1.5 < dw < 2.5 else "可能有自相關"],
        ["變異數同質（等變異性）", f"Breusch-Pagan p-value = {bp_p:.3g}",
         "p ≥ 0.05，等變異 ✓" if bp_p >= 0.05 else "p < 0.05，有異質變異（樣本數很大時容易顯著，請搭配殘差圖判斷）"],
        ["無多重共線性", f"最大 VIF = {vif['VIF'].max():.3f}", "VIF < 5 ✓" if vif["VIF"].max() < 5 else "VIF ≥ 5，有共線性"],
    ], columns=["假設", "檢查結果", "判讀"])
    d1, d2 = st.columns([3, 2])
    d1.dataframe(diag, hide_index=True)
    d2.dataframe(vif.round(4), hide_index=True)

# ---------------- 6 Deployment ----------------
with tabs[5]:
    st.subheader("Deployment：輸入特徵值，即時預測")
    st.caption("以全部特徵欄位的訓練集中位數／眾數為預設值；預測值與 95% 區間由 OLS 模型（選用特徵）計算。")
    with st.form("predict_form"):
        cols = st.columns(4)
        inputs = {}
        for i, c in enumerate(X_raw.columns):
            box = cols[i % 4]
            if c in cat_cols:
                options = sorted(Xtr_raw[c].dropna().astype(str).unique().tolist())
                default = str(prep.modes[c])
                inputs[c] = box.selectbox(str(c), options, index=options.index(default) if default in options else 0)
            else:
                med = float(prep.medians[c])
                is_int = pd.api.types.is_integer_dtype(X_raw[c])
                inputs[c] = box.number_input(str(c), value=int(round(med)) if is_int else med,
                                             step=1 if is_int else None, format=None if is_int else "%.4f")
        submitted = st.form_submit_button("預測", type="primary")
    if submitted:
        one = prep.transform(pd.DataFrame([inputs]))
        row = sm.add_constant(one[feats], has_constant="add")
        pf = ols.get_prediction(row).summary_frame(alpha=0.05).iloc[0]
        model_pred = model_pipes[final_model_name].predict(one[feats])[0]
        kpi_row([
            (f"{final_model_name} 預測值", fmt(model_pred, 6)),
            ("OLS 預測值", fmt(pf["mean"], 6)),
            ("95% 信賴區間", f"{fmt(pf['mean_ci_lower'], 4)} ~ {fmt(pf['mean_ci_upper'], 4)}"),
            ("95% 預測區間", f"{fmt(pf['obs_ci_lower'], 4)} ~ {fmt(pf['obs_ci_upper'], 4)}"),
        ])

    out = Xte_raw.copy()
    out[f"Actual_{target}"] = yte.values
    out[f"Predicted_{target}"] = final_pred
    out["PI_lower"] = pred_frame["obs_ci_lower"].values
    out["PI_upper"] = pred_frame["obs_ci_upper"].values
    st.download_button("下載測試集預測結果（CSV）", out.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"predictions_{cfg['key']}.csv", mime="text/csv")
    st.markdown("**部署方式**：本網頁以 Streamlit 建置並部署於 Streamlit Community Cloud；模型也可用 joblib 儲存後包成 "
                "FastAPI／Flask API，或以批次排程對新資料定期預測。")

# ---------------- 7 網路解法與 AI 輔助 ----------------
with tabs[6]:
    st.subheader("網路主流解法比較（Kaggle S4E5）")
    st.dataframe(FLOOD_EXTERNAL, hide_index=True)
    st.markdown("Kaggle 公開分析指出目標值近似由 20 個特徵的總和決定，所以線性回歸已能抓到主要訊號；"
                "高分解法再加入列統計特徵，用梯度提升樹學習剩下的非線性部分，R² 由約 0.845 提升到約 0.869。")

    st.subheader("NotebookLM 摘要")
    st.markdown(
        "參賽者主要使用梯度提升樹模型（CatBoost、LightGBM、XGBoost），並結合線性模型（Linear Regression、Ridge）"
        "與模型融合（Blending／Ensemble）提升表現。特徵工程是拿高分的關鍵，最核心的做法是跨特徵的列統計量，"
        "例如每筆資料 20 個特徵的總和、平均值、標準差、中位數、偏度與熵。加入進階列統計特徵後，CatBoost 單模 "
        "CV R² 達 0.87003，LightGBM 約 0.86931。線性回歸的優點是訓練極快，只用特徵總和就能得到約 0.844～0.846 "
        "的基準分數；缺點是無法捕捉非線性與特徵交互作用，因此上限受限。"
    )
    st.caption("由 NotebookLM 根據 7 個網路來源（Kaggle 競賽頁、討論區與高分 Notebook）產生。")

    st.subheader("GPT 輔助內容")
    st.markdown(
        "本作業使用 ChatGPT 依 CRISP-DM 各階段協助撰寫 Colab 程式（EDA、前處理、特徵選擇、模型、預測區間、殘差診斷、部署）"
        "與整理網路解法；所有數值皆為實際執行的結果。網頁程式（本 Streamlit App）由 AI 協助撰寫。"
    )
    st.markdown(
        f"- [ChatGPT 對話紀錄（PDF）]({REPO}/blob/main/docs/GPT_conversation.pdf)\n"
        f"- [NotebookLM 研究摘要（PDF）]({REPO}/blob/main/docs/NotebookLM_summary.pdf)\n"
        f"- [完整報告（PDF）]({REPO}/blob/main/docs/5115056015_hw2_report.pdf)\n"
        f"- [Colab Notebook（完整 111 萬筆資料）]({REPO}/blob/main/5115056015_hw2.ipynb)\n"
        f"- [資訊圖表（PNG）]({REPO}/blob/main/images/infographic.png)"
    )
    st.subheader("資訊圖表")
    st.image("https://raw.githubusercontent.com/EasyBoy5566/AIoT-HW2/main/images/infographic.png")
