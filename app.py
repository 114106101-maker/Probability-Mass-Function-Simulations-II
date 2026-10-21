import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# 機率模擬實驗：X、Z、Y、K、J
# ============================================================
st.set_page_config(
    page_title="機率模擬實驗",
    page_icon="📊",
    layout="wide",
)

st.title("📊 機率模擬實驗")
st.caption("軟體缺陷抽樣檢查與網路封包遺失／重傳")

N_DEFAULT = 10_000

# ---------- 理論機率 ----------
def hypergeom_pmf(x):
    # X ~ Hypergeometric(N=40, K=8, n=10)
    if 0 <= x <= 8:
        return math.comb(8, x) * math.comb(32, 10 - x) / math.comb(40, 10)
    return 0.0


def binom_pmf(k, n, p):
    if 0 <= k <= n:
        return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    return 0.0


def geom_pmf(y, p=0.8):
    return p * ((1 - p) ** (y - 1)) if y >= 1 else 0.0


def negbin_total_pmf(j, r=3, p=0.8):
    # J = total trials needed to get r successes
    # P(J=j) = C(j-1,r-1) p^r (1-p)^(j-r), j >= r
    if j >= r:
        return math.comb(j - 1, r - 1) * (p ** r) * ((1 - p) ** (j - r))
    return 0.0


def make_distribution_df(values, support, pmf_func, variable_name):
    counts = pd.Series(values).value_counts()
    df = pd.DataFrame({"結果": support})
    df["模擬相對次數"] = df["結果"].map(
        lambda x: counts.get(x, 0) / len(values)
    )
    df["理論機率"] = df["結果"].map(pmf_func)
    long_df = df.melt(
        id_vars="結果",
        value_vars=["模擬相對次數", "理論機率"],
        var_name="機率類型",
        value_name="機率",
    )
    return df, long_df


def show_histogram(values, support, pmf_func, variable_name, title):
    summary, long_df = make_distribution_df(
        values, support, pmf_func, variable_name
    )
    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("結果:O", title=variable_name, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("機率類型:N"),
            y=alt.Y("機率:Q", title="相對次數／理論機率", scale=alt.Scale(domainMin=0)),
            color=alt.Color("機率類型:N", title="圖例"),
            tooltip=[
                alt.Tooltip("結果:O", title="結果"),
                alt.Tooltip("機率類型:N", title="類型"),
                alt.Tooltip("機率:Q", title="機率", format=".6f"),
            ],
        )
        .properties(title=title, height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    return summary


# ---------- 側邊欄 ----------
with st.sidebar:
    st.header("⚙️ 模擬設定")
    mode = st.selectbox(
        "選擇實驗",
        [
            "模擬一：不歸還抽樣 X（原 Part I）",
            "模擬一：歸還抽樣 Z（Part II d）",
            "模擬二：固定傳輸 10 次 K（Part II c）",
            "模擬二：重傳直到第一次成功 Y（比較用）",
            "模擬二：重傳直到第三次成功 J（Part II d）",
            "比較：Y、K、J 三種直方圖（Part II e）",
        ],
    )
    n_sim = st.number_input(
        "模擬次數",
        min_value=100,
        max_value=100_000,
        value=N_DEFAULT,
        step=100,
    )
    seed = st.number_input("隨機種子", min_value=0, max_value=999_999, value=42, step=1)
    run = st.button("🚀 開始模擬", type="primary", use_container_width=True)

# 以按鈕執行，避免每次更動控制項都重新抽樣
if "results" not in st.session_state:
    st.session_state.results = None
if run:
    rng = np.random.default_rng(int(seed))
    N = int(n_sim)
    population = np.array([1] * 8 + [0] * 32, dtype=np.int8)

    results = {}
    # X：不歸還抽樣，超幾何分布
    results["X"] = rng.choice(population, size=(N, 10), replace=True) if False else None
    x_samples = np.array([
        rng.choice(population, size=10, replace=False).sum()
        for _ in range(N)
    ])
    results["X"] = x_samples

    # Z：每次抽取後歸還；等同 Binomial(n=10,p=0.2)
    results["Z"] = rng.binomial(n=10, p=0.2, size=N)

    # K：固定傳輸 10 次，成功機率 0.8
    results["K"] = rng.binomial(n=10, p=0.8, size=N)

    # Y：直到第一次成功
    results["Y"] = rng.geometric(p=0.8, size=N)

    # J：直到第三次成功；NumPy 回傳的是成功前失敗次數，因此加 3
    results["J"] = rng.negative_binomial(n=3, p=0.8, size=N) + 3

    st.session_state.results = results
    st.session_state.n_sim = N
    st.session_state.seed = int(seed)

results = st.session_state.results
if results is None:
    st.info("請在左側選擇實驗，再按「開始模擬」。預設模擬次數為 10,000 次。")
    st.stop()

N = st.session_state.n_sim
st.caption(f"本次模擬次數：{N:,}｜隨機種子：{st.session_state.seed}")

# ---------- 統計資料 ----------
def event_probability(values, event):
    return float(np.mean(event(values)))


def display_metrics(variable, values, theoretical_mean, event_label=None, theory_prob=None, event_func=None):
    cols = st.columns(4)
    cols[0].metric("模擬平均值", f"{np.mean(values):.6f}")
    cols[1].metric("理論平均值", f"{theoretical_mean:.6f}")
    if event_label is not None:
        sim_p = event_probability(values, event_func)
        cols[2].metric(f"模擬 {event_label}", f"{sim_p:.6f}")
        cols[3].metric(f"理論 {event_label}", f"{theory_prob:.6f}")


# ---------- 顯示選定實驗 ----------
if mode.startswith("模擬一：不歸還"):
    st.header("模擬一：不歸還抽樣檢查（X）")
    st.write("40 個程式中有 8 個缺陷，每次不歸還地抽取 10 個。")
    x = results["X"]
    display_metrics(
        "X", x, 10 * 8 / 40,
        "P(X = 3)", hypergeom_pmf(3), lambda a: a == 3
    )
    support = list(range(0, 9))
    summary = show_histogram(
        x, support, hypergeom_pmf, "X（缺陷程式數）",
        "X：模擬相對次數與超幾何分布理論機率"
    )
    st.write(f"理論 P(X=3) = {hypergeom_pmf(3):.9f}")
    st.dataframe(summary, use_container_width=True, hide_index=True)

elif mode.startswith("模擬一：歸還"):
    st.header("模擬一：歸還抽樣檢查（Z）")
    st.write("每次抽取後歸還，因此每次抽取獨立，Z ~ Binomial(10, 0.2)。")
    z = results["Z"]
    p_z3 = binom_pmf(3, 10, 0.2)
    display_metrics("Z", z, 10 * 0.2, "P(Z = 3)", p_z3, lambda a: a == 3)
    support = list(range(0, 11))
    summary = show_histogram(
        z, support, lambda k: binom_pmf(k, 10, 0.2), "Z（缺陷程式數）",
        "Z：模擬相對次數與二項分布理論機率"
    )
    st.write(f"理論 P(Z=3) = {p_z3:.9f}")
    st.dataframe(summary, use_container_width=True, hide_index=True)

elif mode.startswith("模擬二：固定"):
    st.header("模擬二：固定傳輸 10 次（K）")
    st.write("每次傳輸成功機率為 0.8，K ~ Binomial(10, 0.8)。")
    k = results["K"]
    p_k_gt3 = 1 - sum(binom_pmf(i, 10, 0.8) for i in range(0, 4))
    display_metrics("K", k, 10 * 0.8, "P(K > 3)", p_k_gt3, lambda a: a > 3)
    support = list(range(0, 11))
    summary = show_histogram(
        k, support, lambda i: binom_pmf(i, 10, 0.8), "K（成功次數）",
        "K：模擬相對次數與二項分布理論機率"
    )
    st.write(f"理論 P(K>3) = {p_k_gt3:.9f}")
    st.dataframe(summary, use_container_width=True, hide_index=True)

elif mode.startswith("模擬二：重傳直到第一次"):
    st.header("模擬二：重傳直到第一次成功（Y）")
    st.write("Y ~ Geometric(p=0.8)，代表第一次成功所需的傳輸次數。")
    y = results["Y"]
    p_y_gt3 = 0.2 ** 3
    display_metrics("Y", y, 1 / 0.8, "P(Y > 3)", p_y_gt3, lambda a: a > 3)
    max_show = max(8, int(np.quantile(y, 0.999)))
    support = list(range(1, min(max_show, 30) + 1))
    summary = show_histogram(
        y, support, lambda i: geom_pmf(i, 0.8), "Y（首次成功所需次數）",
        "Y：模擬相對次數與幾何分布理論機率"
    )
    st.write(f"理論 P(Y>3) = {p_y_gt3:.9f}")
    st.dataframe(summary, use_container_width=True, hide_index=True)

elif mode.startswith("模擬二：重傳直到第三次"):
    st.header("模擬二：重傳直到第三次成功（J）")
    st.write("J 是累積得到 3 次成功所需的總傳輸次數，J ~ 負二項分布。")
    j = results["J"]
    display_metrics("J", j, 3 / 0.8)
    max_show = max(10, int(np.quantile(j, 0.999)))
    support = list(range(3, min(max_show, 40) + 1))
    summary = show_histogram(
        j, support, lambda i: negbin_total_pmf(i, 3, 0.8), "J（第三次成功所需總次數）",
        "J：模擬相對次數與負二項分布理論機率"
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    st.header("模擬二：Y、K、J 三種直方圖比較")
    st.write(
        "Y：直到第一次成功；K：固定傳輸 10 次的成功數；"
        "J：直到第三次成功。三者的橫軸意義不同，因此分別繪圖比較。"
    )
    y, k, j = results["Y"], results["K"], results["J"]
    st.subheader("Y：直到第一次成功")
    display_metrics("Y", y, 1 / 0.8, "P(Y > 3)", 0.2 ** 3, lambda a: a > 3)
    show_histogram(
        y, list(range(1, 11)), lambda i: geom_pmf(i, 0.8), "Y（傳輸次數）",
        "Y：幾何分布"
    )
    st.subheader("K：固定傳輸 10 次")
    p_k_gt3 = 1 - sum(binom_pmf(i, 10, 0.8) for i in range(0, 4))
    display_metrics("K", k, 8, "P(K > 3)", p_k_gt3, lambda a: a > 3)
    show_histogram(
        k, list(range(0, 11)), lambda i: binom_pmf(i, 10, 0.8), "K（成功次數）",
        "K：二項分布"
    )
    st.subheader("J：直到第三次成功")
    display_metrics("J", j, 3 / 0.8)
    show_histogram(
        j, list(range(3, 16)), lambda i: negbin_total_pmf(i, 3, 0.8), "J（傳輸次數）",
        "J：負二項分布"
    )

# ---------- 比較 X 與 Z ----------
st.divider()
st.header("補充比較：不歸還抽樣 X 與歸還抽樣 Z")
st.write("兩者的理論平均值皆為 2，但分布不同。")
comp = pd.DataFrame({
    "變數": ["X（不歸還）", "Z（歸還）"],
    "理論平均值": [2.0, 2.0],
    "理論變異數": [
        10 * (8 / 40) * (1 - 8 / 40) * ((40 - 10) / (40 - 1)),
        10 * 0.2 * 0.8,
    ],
    "理論 P(結果=3)": [hypergeom_pmf(3), binom_pmf(3, 10, 0.2)],
    "模擬平均值": [float(np.mean(results["X"])), float(np.mean(results["Z"]))],
    "模擬 P(結果=3)": [float(np.mean(results["X"] == 3)), float(np.mean(results["Z"] == 3))],
})
st.dataframe(comp, use_container_width=True, hide_index=True)

st.caption("提示：模擬值會因隨機種子而改變；10,000 次模擬的結果通常會接近理論值，但不一定完全相同。")
