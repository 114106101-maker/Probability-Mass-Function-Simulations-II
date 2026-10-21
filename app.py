import time
import textwrap
from math import comb

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# 1. 頁面配置與原有樣式
# ============================================================
st.set_page_config(
    page_title="📊 機率模擬實驗 | Probability Simulations",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

custom_css = textwrap.dedent("""
<style>
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #ffffff !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif !important;
    color: #1c1c1e !important;
}
.main .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1100px; }
.card { background:#fff; border-radius:20px; padding:20px 24px; margin-bottom:18px; box-shadow:0 4px 20px rgba(0,0,0,.03); border:1px solid rgba(0,0,0,.06); }
.ios-kpi-card { background:#fff; border-radius:18px; padding:14px 10px; text-align:center; box-shadow:0 2px 12px rgba(0,0,0,.03); border:1px solid rgba(0,0,0,.06); }
.ios-kpi-title { font-size:12px; font-weight:500; color:#8e8e93; margin-bottom:4px; line-height:1.3; }
.ios-kpi-value { font-size:22px; font-weight:700; color:#1c1c1e; }
div[data-testid="stSidebar"] { background:#f8f9fa !important; border-right:1px solid rgba(0,0,0,.06); }
div.stButton > button { border-radius:16px !important; background:#fff !important; border:1px solid rgba(0,0,0,.1) !important; padding:.6rem .25rem !important; box-shadow:0 2px 8px rgba(0,0,0,.04) !important; transition:transform .3s ease, box-shadow .3s ease, background-color .2s ease !important; }
div.stButton > button, div.stButton > button p, div.stButton > button span { color:#1c1c1e !important; font-weight:600 !important; font-size:13px !important; }
div.stButton > button:hover { background:#f2f2f7 !important; transform:translateY(-3px) !important; box-shadow:0 8px 20px rgba(0,0,0,.12) !important; }
div.stButton > button[kind="primary"] { background:linear-gradient(135deg,#007aff 0%,#0056b3 100%) !important; border:none !important; }
div.stButton > button[kind="primary"], div.stButton > button[kind="primary"] p, div.stButton > button[kind="primary"] span { color:#fff !important; }
.inspect-card { background:#f8f9fa; border-radius:18px; padding:16px; margin:8px 0 18px; border:1px solid rgba(0,0,0,.06); }
.inspect-title { font-size:16px; font-weight:750; margin-bottom:10px; }
.inspect-grid { display:grid; grid-template-columns:repeat(5,minmax(65px,1fr)); gap:8px; }
.inspect-item { background:#fff; border-radius:14px; padding:10px 6px; text-align:center; border:1px solid rgba(0,0,0,.06); box-shadow:0 2px 8px rgba(0,0,0,.03); }
.inspect-item .num { font-size:11px; color:#8e8e93; }
.inspect-item .state { font-size:24px; margin:2px 0; }
.inspect-item .label { font-size:11px; font-weight:650; }
.inspect-success { color:#34c759; } .inspect-fail { color:#ff3b30; } .inspect-defect { color:#ff9500; } .inspect-normal { color:#007aff; }
.packet-scene { background:#fff; border:1px solid rgba(0,0,0,.07); border-radius:14px; padding:14px; margin:12px 0; }
.packet-lane { display:grid; grid-template-columns:76px minmax(100px,1fr) 76px; align-items:center; gap:10px; margin:10px 0; }
.packet-node { text-align:center; font-size:12px; font-weight:700; color:#4b5563; }
.packet-node .node-icon { font-size:25px; display:block; }
.packet-track { height:34px; position:relative; border-top:2px dashed #cbd5e1; margin-top:12px; }
.packet-track:before { content:""; position:absolute; right:-1px; top:-7px; border-left:10px solid #94a3b8; border-top:6px solid transparent; border-bottom:6px solid transparent; }
.packet-dot { position:absolute; left:0; top:-17px; width:34px; height:30px; border-radius:8px; background:#dbeafe; border:1px solid #60a5fa; display:flex; align-items:center; justify-content:center; font-size:17px; box-shadow:0 2px 5px rgba(0,0,0,.08); animation:packetMove 1.8s ease-in-out infinite; }
.packet-dot.failed { --travel:calc((100% - 34px) * .62); animation-name:packetFail; }
.packet-dot.success { --travel:calc(100% - 34px); animation-name:packetMove; }
@keyframes packetMove { 0% { left:0; } 75%,100% { left:var(--travel,calc(100% - 34px)); } }
@keyframes packetFail { 0% { left:0; opacity:1; } 60% { left:var(--travel); opacity:1; } 75%,100% { left:var(--travel); opacity:.15; } }
.packet-status { font-size:12px; font-weight:700; margin-top:2px; }
@media(max-width:700px) { .inspect-grid { grid-template-columns:repeat(5,minmax(48px,1fr)); gap:5px; } .inspect-item { padding:7px 3px; } .packet-lane { grid-template-columns:58px minmax(70px,1fr) 58px; gap:6px; } .packet-node { font-size:10px; } }
</style>
""")
st.markdown(custom_css, unsafe_allow_html=True)


def render_kpi_html(title_cn, title_en, val, color="#1c1c1e"):
    return f"""<div class="ios-kpi-card"><div class="ios-kpi-title">{title_cn}<br><span style="font-size:10px;">{title_en}</span></div><div class="ios-kpi-value" style="color:{color};">{val}</div></div>"""


# ============================================================
# 2. 理論分布與模擬函式
# ============================================================
def hypergeom_pmf(x):
    return comb(8, x) * comb(32, 10 - x) / comb(40, 10) if 0 <= x <= 8 and 0 <= 10-x <= 32 else 0.0


def binom_pmf(k, n, p):
    return comb(n, k) * p**k * (1-p)**(n-k) if 0 <= k <= n else 0.0


def geom_pmf(y, p=0.8):
    return p * (1-p)**(y-1) if y >= 1 else 0.0


def negbin_total_pmf(j, r=3, p=0.8):
    return comb(j-1, r-1) * p**r * (1-p)**(j-r) if j >= r else 0.0


@st.cache_data
def simulate_all(total_n, seed):
    rng = np.random.default_rng(seed)
    population = np.array([1] * 8 + [0] * 32, dtype=np.int8)

    # X：不歸還抽樣；保留每次抽到的 10 個程式，供明細檢視
    sample_matrix = np.empty((total_n, 10), dtype=np.int8)
    for i in range(total_n):
        sample_matrix[i] = rng.choice(population, size=10, replace=False)
    x = sample_matrix.sum(axis=1)

    # Z：歸還抽樣，Binomial(10, 0.2)
    z = rng.binomial(n=10, p=0.2, size=total_n)

    # K：固定傳輸 10 次，Binomial(10, 0.8)
    k = rng.binomial(n=10, p=0.8, size=total_n)

    # Y：直到第一次成功
    y = rng.geometric(p=0.8, size=total_n)

    # J：直到第三次成功；NumPy 回傳第三次成功前的失敗數，因此 +3
    j = rng.negative_binomial(n=3, p=0.8, size=total_n) + 3

    return {"X": x, "Z": z, "K": k, "Y": y, "J": j, "sample_matrix": sample_matrix}


def make_histogram(values, support, pmf, x_title, title):
    counts = pd.Series(values).value_counts()
    df = pd.DataFrame({"結果": support})
    df["模擬相對次數"] = df["結果"].map(lambda v: counts.get(v, 0) / len(values))
    df["理論機率"] = df["結果"].map(pmf)
    long_df = df.melt(id_vars="結果", var_name="機率類型", value_name="機率")
    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("結果:O", title=x_title, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("機率類型:N"),
            y=alt.Y("機率:Q", title="機率", scale=alt.Scale(domainMin=0)),
            color=alt.Color("機率類型:N", title="圖例"),
            tooltip=[
                alt.Tooltip("結果:O", title="結果"),
                alt.Tooltip("機率類型:N", title="類型"),
                alt.Tooltip("機率:Q", title="機率", format=".6f"),
            ],
        ).properties(title=title, height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    return df


def event_spec(mode):
    specs = {
        "X：不歸還抽樣": {
            "key": "X", "label": "P(X = 3)", "event": lambda a: a == 3,
            "theory_event": hypergeom_pmf(3), "mean": 2.0, "mean_label": "E(X)",
            "support": list(range(9)), "pmf": hypergeom_pmf,
            "x_title": "X = 抽到的缺陷程式數", "title": "X 的相對次數直方圖",
            "description": "每次不歸還地隨機抽取 10 個程式，X 為抽到的缺陷程式數。",
        },
        "Z：歸還抽樣": {
            "key": "Z", "label": "P(Z = 3)", "event": lambda a: a == 3,
            "theory_event": binom_pmf(3, 10, 0.2), "mean": 2.0, "mean_label": "E(Z)",
            "support": list(range(11)), "pmf": lambda a: binom_pmf(a, 10, 0.2),
            "x_title": "Z = 抽到的缺陷程式數", "title": "Z 的相對次數直方圖",
            "description": "每次抽取後歸還，再獨立抽取 10 次；Z ~ Binomial(10, 0.2)。",
        },
        "K：固定傳輸 10 次": {
            "key": "K", "label": "P(K > 3)", "event": lambda a: a > 3,
            "theory_event": 1 - sum(binom_pmf(i, 10, 0.8) for i in range(4)),
            "mean": 8.0, "mean_label": "E(K)", "support": list(range(11)),
            "pmf": lambda a: binom_pmf(a, 10, 0.8), "x_title": "K = 10 次中的成功次數",
            "title": "K 的相對次數直方圖",
            "description": "固定傳輸 10 次，K 為成功次數；K ~ Binomial(10, 0.8)。",
        },
        "Y：重傳直到第一次成功": {
            "key": "Y", "label": "P(Y > 3)", "event": lambda a: a > 3,
            "theory_event": 0.2**3, "mean": 1.25, "mean_label": "E(Y)",
            "support": list(range(1, 11)), "pmf": geom_pmf,
            "x_title": "Y = 第一次成功所需傳輸次數", "title": "Y 的相對次數直方圖",
            "description": "持續重傳直到第一次成功，Y 為所需傳輸次數；Y ~ Geometric(0.8)。",
        },
        "J：重傳直到第三次成功": {
            "key": "J", "label": "平均傳輸次數", "event": None,
            "theory_event": None, "mean": 3.75, "mean_label": "E(J)",
            "support": list(range(3, 16)), "pmf": lambda a: negbin_total_pmf(a, 3, 0.8),
            "x_title": "J = 第三次成功所需總傳輸次數", "title": "J 的相對次數直方圖",
            "description": "持續重傳直到第三次成功，J 為總傳輸次數；J 服從負二項分布。",
        },
    }
    mode_key = str(mode).strip()

    if mode_key not in specs:
        st.error(f"找不到模擬模式：{mode_key!r}")
        st.error(f"目前支援的模式：{list(specs.keys())}")
        st.stop()

    return specs[mode_key]


# ============================================================
# 3. 側邊欄：原控制功能 + 新增實驗
# ============================================================
MODES = [
    "X：不歸還抽樣",
    "Z：歸還抽樣",
    "K：固定傳輸 10 次",
    "Y：重傳直到第一次成功",
    "J：重傳直到第三次成功",
    "比較：X 與 Z",
    "比較：Y、K、J",
]

if "anim_status" not in st.session_state:
    st.session_state.anim_status = "idle"
if "current_step_idx" not in st.session_state:
    st.session_state.current_step_idx = 0
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = MODES[0]
if "total_n" not in st.session_state:
    st.session_state.total_n = 10000

with st.sidebar:
    st.header("⚙️ 控制面板 (Control Panel)")
    mode = st.radio(
        "選擇模擬 (Select Simulation)",
        MODES,
        index=MODES.index(st.session_state.selected_mode),
        key="mode_radio",
    )
    if mode != st.session_state.selected_mode:
        st.session_state.selected_mode = mode
        st.session_state.anim_status = "idle"
        st.session_state.current_step_idx = 0

    total_n = st.number_input(
        "模擬總次數 (N) / Total Simulations",
        min_value=100, max_value=10000, value=int(st.session_state.total_n),
        step=100, key="total_n_input",
    )
    st.session_state.total_n = int(total_n)
    fps = st.slider("動畫速率 (FPS) / Speed", 2, 15, 6)
    sound_enabled = st.checkbox(
        "🔊 開啟動畫提示音", value=True,
        help="瀏覽器可能會阻擋自動播放；若無聲，請按動畫下方的啟用音效按鈕。",
    )
    seed = st.number_input("隨機種子 (Seed)", min_value=0, max_value=9999, value=42)
    st.divider()
    c1, c2, c3 = st.columns(3)
    start_click = c1.button("🚀 開始 Start", type="primary", use_container_width=True)
    pause_label = "▶️ 繼續 Resume" if st.session_state.anim_status == "paused" else "⏸️ 暫停 Pause"
    pause_click = c2.button(pause_label, use_container_width=True)
    finish_click = c3.button("⚡ 結算 Finish", use_container_width=True)

if start_click:
    st.session_state.anim_status = "running"
    st.session_state.current_step_idx = 0
if pause_click:
    if st.session_state.anim_status == "running":
        st.session_state.anim_status = "paused"
    elif st.session_state.anim_status == "paused":
        st.session_state.anim_status = "running"
if finish_click:
    st.session_state.anim_status = "finished"
    st.session_state.current_step_idx = 0

# Cache simulated results by N and seed. Recompute only when those values change.
all_data = simulate_all(int(total_n), int(seed))
if "last_run_signature" not in st.session_state:
    st.session_state.last_run_signature = (int(total_n), int(seed))
if st.session_state.last_run_signature != (int(total_n), int(seed)):
    st.session_state.last_run_signature = (int(total_n), int(seed))
    st.session_state.anim_status = "idle"
    st.session_state.current_step_idx = 0

# ============================================================
# 4. 直方圖比較模式
# ============================================================
if mode == "比較：X 與 Z":
    st.title("📊 比較：不歸還抽樣 X 與歸還抽樣 Z")
    st.markdown("兩者平均值相同，但抽樣方式不同，因而具有不同的變異數與分布。")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("X：不歸還抽樣")
        st.metric("模擬平均", f"{np.mean(all_data['X']):.5f}")
        st.metric("模擬 P(X=3)", f"{np.mean(all_data['X'] == 3):.5f}")
        dx = make_histogram(all_data["X"], list(range(9)), hypergeom_pmf, "缺陷程式數", "X：超幾何分布")
        st.dataframe(dx, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Z：歸還抽樣")
        st.metric("模擬平均", f"{np.mean(all_data['Z']):.5f}")
        st.metric("模擬 P(Z=3)", f"{np.mean(all_data['Z'] == 3):.5f}")
        dz = make_histogram(all_data["Z"], list(range(11)), lambda a: binom_pmf(a, 10, .2), "缺陷程式數", "Z：二項分布")
        st.dataframe(dz, use_container_width=True, hide_index=True)
    comparison = pd.DataFrame({
        "項目": ["分布", "理論平均", "理論變異數", "P(結果=3)"],
        "X：不歸還": [
            "超幾何分布", "2.000000",
            f"{10*.2*.8*(40-10)/(40-1):.6f}", f"{hypergeom_pmf(3):.6f}"
        ],
        "Z：歸還": [
            "二項分布", "2.000000", f"{10*.2*.8:.6f}", f"{binom_pmf(3,10,.2):.6f}"
        ],
    })
    st.subheader("理論比較")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.info("兩者理論平均皆為 2；歸還抽樣的理論變異數較大（1.6），不歸還抽樣的變異數較小（約 1.3846）。")

elif mode == "比較：Y、K、J":
    st.title("📡 比較：Y、K、J 三種網路封包實驗")
    st.markdown("Y 是第一次成功所需次數；K 是固定 10 次內的成功數；J 是第三次成功所需總次數。")
    specs = [
        ("Y", "Y：幾何分布", list(range(1, 11)), geom_pmf, "Y = 首次成功所需傳輸次數"),
        ("K", "K：二項分布", list(range(11)), lambda a: binom_pmf(a, 10, .8), "K = 10 次中的成功次數"),
        ("J", "J：負二項分布", list(range(3, 16)), lambda a: negbin_total_pmf(a, 3, .8), "J = 第三次成功所需總次數"),
    ]
    for key, title, support, pmf, x_title in specs:
        with st.container():
            st.subheader(title)
            vals = all_data[key]
            st.write(f"模擬平均值：**{np.mean(vals):.5f}**")
            make_histogram(vals, support, pmf, x_title, title)
    st.subheader("理論平均比較")
    st.dataframe(pd.DataFrame({
        "變數": ["Y", "K", "J"],
        "理論平均": [1/.8, 10*.8, 3/.8],
        "模擬平均": [np.mean(all_data["Y"]), np.mean(all_data["K"]), np.mean(all_data["J"])],
        "分布": ["幾何分布", "二項分布", "負二項分布"],
    }), use_container_width=True, hide_index=True)

else:
    # ========================================================
    # 5. 單一實驗模式：動畫、音效、收斂圖、KPI、明細、CSV
    # ========================================================
    spec = event_spec(mode)
    key = spec["key"]
    values = all_data[key]
    n = len(values)
    event_func = spec["event"]
    theory_event = spec["theory_event"]
    cumulative_mean = np.cumsum(values) / np.arange(1, n + 1)
    if event_func is not None:
        event_flags = event_func(values)
        cumulative_probability = np.cumsum(event_flags) / np.arange(1, n + 1)
        event_count_total = int(np.sum(event_flags))
    else:
        event_flags = None
        cumulative_probability = cumulative_mean.copy()
        event_count_total = None

    st.title("📊 機率模擬實驗")
    st.markdown(
        f"""<div class="card">
        <div style="font-size:22px;font-weight:800;margin-bottom:8px;">{mode}</div>
        <div style="font-size:14px;line-height:1.7;color:#3a3a3c;">
        📌 <b>實驗設定：</b>{spec['description']}<br>
        重複模擬 <b>{n:,}</b> 次。<br>
        🎯 <b>理論平均：</b>{spec['mean_label']} = {spec['mean']:.6f}<br>
        </div></div>""",
        unsafe_allow_html=True,
    )

    # animation frames
    frame_indices = np.unique(np.linspace(1, n, num=min(n, 24), dtype=int))
    inspection_spot = st.empty()
    kpi_cols = st.columns(4)
    kpi_spots = [col.empty() for col in kpi_cols]
    chart_spot = st.empty()
    conv_spot = st.empty()
    mean_spot = st.empty()

    def render_inspection(frame_n):
        # Show the actual sampled program list for X, or an illustrative transmission sequence for packet experiments.
        if key == "X":
            sample = all_data["sample_matrix"][frame_n - 1].tolist()
            items = []
            for idx, state in enumerate(sample, 1):
                icon = "⚠️" if state else "✓"
                cls = "inspect-defect" if state else "inspect-normal"
                label = "有缺陷" if state else "正常"
                items.append(
                    f'<div class="inspect-item"><div class="num">程式 {idx}</div>'
                    f'<div class="state {cls}">{icon}</div><div class="label {cls}">{label}</div></div>'
                )
            defects = sum(sample)
            inspection_spot.markdown(
                f'<div class="inspect-card"><div class="inspect-title">🔎 第 {frame_n:,} 次抽樣檢查：10 個程式</div>'
                f'<div style="font-size:13px;color:#6e6e73;margin-bottom:10px;">本次抽到 <b>{defects}</b> 個缺陷、'
                f'<b>{10-defects}</b> 個正常程式。</div><div class="inspect-grid">{"".join(items)}</div></div>',
                unsafe_allow_html=True,
            )
        elif key == "Z":
            # A single illustrative draw of 10 independent draws with replacement
            rng = np.random.default_rng(int(seed) + int(frame_n))
            sample = rng.binomial(1, .2, size=10).tolist()
            items = []
            for idx, state in enumerate(sample, 1):
                icon = "⚠️" if state else "✓"
                cls = "inspect-defect" if state else "inspect-normal"
                label = "有缺陷" if state else "正常"
                items.append(
                    f'<div class="inspect-item"><div class="num">抽取 {idx}</div>'
                    f'<div class="state {cls}">{icon}</div><div class="label {cls}">{label}</div></div>'
                )
            inspection_spot.markdown(
                f'<div class="inspect-card"><div class="inspect-title">🔎 第 {frame_n:,} 次歸還抽樣示意</div>'
                f'<div style="font-size:13px;color:#6e6e73;margin-bottom:10px;">每次抽取後都歸還，單次抽到缺陷的機率為 0.2。'
                f'此處顯示該次抽樣示意。</div><div class="inspect-grid">{"".join(items)}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            attempts = int(values[frame_n - 1])
            if key == "K":
                # Display the 10 fixed attempts for this trial
                rng = np.random.default_rng(int(seed) + int(frame_n) + 17)
                trial = rng.binomial(1, .8, size=10)
                # Ensure the illustration agrees with the simulated K count.
                success_count = int(values[frame_n - 1])
                trial[:] = 0
                trial[:success_count] = 1
                rng.shuffle(trial)
                attempts = 10
                stop_at_success = False
            elif key == "J":
                stop_at_success = True
                # Illustrative sequence ending on the third success; the final count agrees with J.
                rng = np.random.default_rng(int(seed) + int(frame_n) + 31)
                trial = []
                successes = 0
                for _ in range(max(0, attempts - 3)):
                    trial.append(0)
                trial.extend([1, 1, 1])
                rng.shuffle(trial)
                trial = np.array(trial)
                # Place the final success at the end.
                trial[-1] = 1
                if int(np.sum(trial)) != 3:
                    trial[:-1] = 0
                    trial[-2] = 1
                    trial[-3] = 1
                stop_at_success = True
            else:
                stop_at_success = True
                trial = np.array([0] * (attempts - 1) + [1])

            items, lanes = [], []
            for idx, state in enumerate(trial, 1):
                success = bool(state)
                cls = "inspect-success" if success else "inspect-fail"
                icon = "✓" if success else "✕"
                label = "成功" if success else "失敗"
                items.append(
                    f'<div class="inspect-item"><div class="num">第 {idx} 次</div>'
                    f'<div class="state {cls}">{icon}</div><div class="label {cls}">{label}</div></div>'
                )
                lanes.append(
                    f'<div class="packet-lane"><div class="packet-node"><span class="node-icon">💻</span>傳送端</div>'
                    f'<div><div class="packet-track"><div class="packet-dot {"success" if success else "failed"}" '
                    f'style="animation-delay:{(idx-1)*.08:.2f}s;">📦</div></div>'
                    f'<div class="packet-status {cls}">第 {idx} 次：{"封包成功抵達" if success else "封包遺失，準備重傳"}</div></div>'
                    f'<div class="packet-node"><span class="node-icon">🖥️</span>接收端</div></div>'
                )
            description = (
                f"本次固定傳輸 10 次，成功 {int(values[frame_n-1])} 次。"
                if key == "K" else
                f"本次總傳輸次數為 {attempts}，直到{'第三次' if key == 'J' else '第一次'}成功。"
            )
            inspection_spot.markdown(
                f'<div class="inspect-card"><div class="inspect-title">📡 第 {frame_n:,} 組封包傳輸示意</div>'
                f'<div style="font-size:13px;color:#6e6e73;margin-bottom:10px;">{description}</div>'
                f'<div class="packet-scene">{"".join(lanes)}</div><div class="inspect-grid">{"".join(items)}</div></div>',
                unsafe_allow_html=True,
            )

    def render_frame(frame_n, play_sound=False):
        frame_n = int(frame_n)
        current_mean = cumulative_mean[frame_n - 1]
        if event_func is not None:
            current_prob = cumulative_probability[frame_n - 1]
            current_count = int(np.sum(event_flags[:frame_n]))
            event_text = f"{current_prob:.4f}"
            event_count_text = str(current_count)
            theory_text = f"{theory_event:.4f}"
            conv_title = f"{spec['label']} 收斂軌跡"
            conv_values = cumulative_probability
        else:
            current_prob = None
            event_count_text = "—"
            event_text = "—"
            theory_text = "—"
            conv_title = "累積平均值收斂軌跡"
            conv_values = cumulative_mean

        render_inspection(frame_n)
        kpi_spots[0].markdown(render_kpi_html("模擬次數", "Simulations", f"{frame_n:,}"), unsafe_allow_html=True)
        kpi_spots[1].markdown(render_kpi_html("事件發生次數", "Event Count", event_count_text), unsafe_allow_html=True)
        kpi_spots[2].markdown(render_kpi_html("模擬事件機率", "Simulation", event_text, "#007AFF"), unsafe_allow_html=True)
        kpi_spots[3].markdown(render_kpi_html("理論事件機率", "Theory", theory_text, "#FF3B30"), unsafe_allow_html=True)

        # Fixed-width support prevents the x-axis changing while animation advances.
        if key in ("X",):
            support = list(range(9))
        elif key in ("Z", "K"):
            support = list(range(11))
        elif key == "Y":
            support = list(range(1, min(15, max(8, int(np.quantile(values, .99)))) + 1))
        else:
            support = list(range(3, min(25, max(10, int(np.quantile(values, .99))) ) + 1))
        with chart_spot.container():
            summary = pd.DataFrame({"結果": support})
            counts = pd.Series(values[:frame_n]).value_counts()
            summary["模擬相對次數"] = summary["結果"].map(lambda v: counts.get(v, 0) / frame_n)
            summary["理論機率"] = summary["結果"].map(spec["pmf"])
            long_df = summary.melt(id_vars="結果", var_name="機率類型", value_name="機率")
            chart = (
                alt.Chart(long_df).mark_bar().encode(
                    x=alt.X("結果:O", title=spec["x_title"], axis=alt.Axis(labelAngle=0)),
                    xOffset=alt.XOffset("機率類型:N"),
                    y=alt.Y("機率:Q", title="機率", scale=alt.Scale(domainMin=0)),
                    color=alt.Color("機率類型:N", title="圖例"),
                    tooltip=["結果", "機率類型", alt.Tooltip("機率:Q", format=".4f")],
                ).properties(title=spec["title"], height=320)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("模擬相對次數與理論機率並排顯示；模擬次數增加時，兩者通常會逐漸接近。")

        conv_n = min(frame_n, 400)
        conv_idx = np.linspace(0, frame_n - 1, conv_n, dtype=int)
        conv_df = pd.DataFrame({
            "模擬值": conv_values[conv_idx],
            "理論值": np.full(conv_n, theory_event if event_func is not None else spec["mean"]),
        }, index=conv_idx + 1)
        with conv_spot.container():
            st.markdown(f"### {conv_title}")
            st.line_chart(conv_df, use_container_width=True)

        mean_spot.caption(f"目前平均值 {spec['mean_label']} = {current_mean:.4f} ｜ 理論平均 = {spec['mean']:.4f}")

        if sound_enabled:
            is_packet = key in ("Y", "K", "J")
            components.html(
                f"""<div style="font-family:sans-serif;font-size:12px;color:#666">
                <button id="enableSound" style="padding:6px 10px;border:1px solid #ccc;border-radius:8px;background:#fff;cursor:pointer">🔊 啟用／測試音效</button>
                <span id="soundStatus" style="margin-left:8px">首次使用請點擊按鈕測試音效</span>
                <script>
                (() => {{
                  const b=document.getElementById('enableSound'), s=document.getElementById('soundStatus'); let ctx;
                  function play() {{
                    if(!ctx) ctx=new (window.AudioContext||window.webkitAudioContext)();
                    const notes={str([659.25,493.88] if is_packet else [523.25,659.25,783.99])};
                    notes.forEach((f,i)=>{{const o=ctx.createOscillator(),g=ctx.createGain(),t=ctx.currentTime+i*.06;
                    o.type='sine';o.frequency.setValueAtTime(f,t);g.gain.setValueAtTime(.0001,t);
                    g.gain.exponentialRampToValueAtTime(.035,t+.01);g.gain.exponentialRampToValueAtTime(.0001,t+.08);
                    o.connect(g);g.connect(ctx.destination);o.start(t);o.stop(t+.09);}});
                  }}
                  b.addEventListener('click',async()=>{{try{{if(!ctx)ctx=new(window.AudioContext||window.webkitAudioContext)();await ctx.resume();play();s.textContent='音效已啟用';}}catch(e){{s.textContent='無法播放，請檢查瀏覽器音效權限';}}}});
                  if({str(bool(play_sound)).lower()}) requestAnimationFrame(async()=>{{try{{if(!ctx)ctx=new(window.AudioContext||window.webkitAudioContext)();await ctx.resume();if(ctx.state==='running')play();}}catch(e){{}}}});
                }})();
                </script></div>""",
                height=42,
            )

    if st.session_state.anim_status == "idle":
        render_frame(1)
        st.info("👈 點擊左側「開始」播放動畫，或「結算」直接查看完整結果。")
    elif st.session_state.anim_status == "finished":
        render_frame(n)
        st.success(f"模擬完成！平均值 = {np.mean(values):.6f}，理論平均 = {spec['mean']:.6f}。")
    elif st.session_state.anim_status == "paused":
        idx = min(st.session_state.current_step_idx, len(frame_indices)-1)
        current_n = int(frame_indices[idx])
        render_frame(current_n)
        st.warning(f"動畫已暫停於第 {current_n:,} 次模擬；按「繼續」可繼續播放。")
    elif st.session_state.anim_status == "running":
        progress = st.progress(0)
        start_idx = min(st.session_state.current_step_idx, len(frame_indices)-1)
        for idx in range(start_idx, len(frame_indices)):
            st.session_state.current_step_idx = idx
            frame_n = int(frame_indices[idx])
            progress.progress(int((idx+1)/len(frame_indices)*100))
            render_frame(frame_n, play_sound=True)
            time.sleep(1/fps)
            # Streamlit reruns on button clicks; state retains the most recent frame.
            if st.session_state.anim_status == "paused":
                break
        progress.empty()
        if st.session_state.anim_status == "running":
            st.session_state.anim_status = "finished"
            st.rerun()

    # Summary table and detailed lookup, kept from the original application.
    if st.session_state.anim_status in ("paused", "finished"):
        st.subheader("📊 指定模擬次數統計結果 (Summary Table)")
        checkpoints = [v for v in [100, 500, 1000, 2500, 5000, 10000] if v <= n]
        rows = []
        for checkpoint in checkpoints:
            row = {
                "模擬次數 / Simulations": checkpoint,
                f"模擬 {spec['mean_label']}": f"{cumulative_mean[checkpoint-1]:.6f}",
                f"理論 {spec['mean_label']}": f"{spec['mean']:.6f}",
            }
            if event_func is not None:
                row[f"模擬 {spec['label']}"] = f"{cumulative_probability[checkpoint-1]:.6f}"
                row[f"理論 {spec['label']}"] = f"{theory_event:.6f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("📝 題目要求結果")
        c1, c2 = st.columns(2)
        c1.markdown(f"**理論平均 {spec['mean_label']}：** `{spec['mean']:.6f}`")
        c2.markdown(f"**模擬平均 {spec['mean_label']}：** `{np.mean(values):.6f}`")
        if event_func is not None:
            c1.markdown(f"**理論 {spec['label']}：** `{theory_event:.6f}`")
            c2.markdown(f"**模擬 {spec['label']}：** `{np.mean(event_flags):.6f}`")

        st.divider()
        st.subheader("🔍 每一次模擬結果查詢 (Trial-by-trial Details)")
        selected_trial = st.number_input(
            "要查看第幾次模擬？", min_value=1, max_value=n, value=1, step=1,
            key=f"selected_trial_{key}",
        )
        trial_idx = int(selected_trial) - 1
        if key == "X":
            sample = all_data["sample_matrix"][trial_idx]
            detail_df = pd.DataFrame({
                "抽樣位置": np.arange(1, 11),
                "程式編號": [f"程式 {i}" for i in range(1, 11)],
                "抽樣結果": ["有缺陷" if v else "正常" for v in sample],
            })
            st.markdown(f"第 {int(selected_trial)} 次抽到 {int(np.sum(sample))} 個缺陷程式；X = {int(values[trial_idx])}。")
        elif key == "Z":
            st.markdown(f"第 {int(selected_trial)} 次歸還抽樣，Z = {int(values[trial_idx])} 個缺陷程式。")
            detail_df = pd.DataFrame({
                "抽取次數": np.arange(1, 11),
                "結果": ["有缺陷" if v else "正常" for v in np.random.default_rng(int(seed)+trial_idx).binomial(1, .2, 10)],
                "說明": ["每次抽取後歸還，彼此獨立"] * 10,
            })
        elif key == "K":
            kval = int(values[trial_idx])
            outcomes = np.array([1]*kval + [0]*(10-kval))
            np.random.default_rng(int(seed)+trial_idx).shuffle(outcomes)
            detail_df = pd.DataFrame({
                "傳輸次數": np.arange(1, 11),
                "結果": ["成功" if v else "失敗" for v in outcomes],
            })
            st.markdown(f"第 {int(selected_trial)} 組固定傳輸 10 次，成功 {kval} 次；K = {kval}。")
        elif key == "Y":
            attempts = int(values[trial_idx])
            detail_df = pd.DataFrame({
                "傳輸次數": np.arange(1, attempts+1),
                "結果": ["成功" if i == attempts else "失敗" for i in range(1, attempts+1)],
            })
            st.markdown(f"第 {int(selected_trial)} 組共傳輸 {attempts} 次才第一次成功；Y = {attempts}。")
        else:
            attempts = int(values[trial_idx])
            outcomes = [0] * (attempts - 3) + [1, 1, 1]
            detail_df = pd.DataFrame({
                "傳輸次數": np.arange(1, attempts+1),
                "結果": ["成功" if v else "失敗" for v in outcomes],
                "累積成功次數": np.cumsum(outcomes),
            })
            st.markdown(f"第 {int(selected_trial)} 組共傳輸 {attempts} 次才取得第三次成功；J = {attempts}。")
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        all_trials = pd.DataFrame({
            "模擬次數": np.arange(1, n+1),
            "結果值": values,
            "累積平均值": cumulative_mean,
        })
        if event_func is not None:
            all_trials[f"{spec['label']} 是否發生"] = event_flags
            all_trials[f"累積 {spec['label']}"] = cumulative_probability
        st.markdown("**全部模擬摘要（可搜尋、排序及下載）**")
        st.dataframe(all_trials, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ 下載全部模擬結果 CSV",
            data=all_trials.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"simulation_{key}_trial_results.csv",
            mime="text/csv",
        )
