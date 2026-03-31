import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="安慰语图灵测试",
    page_icon="🧠",
    layout="wide",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

STIMULI_PATH = DATA_DIR / "stimuli.csv"
RESPONSES_PATH = DATA_DIR / "responses.csv"


# =========================
# Helpers
# =========================
def init_state() -> None:
    defaults = {
        "participant_id": "",
        "consented": False,
        "started": False,
        "finished": False,
        "admin_mode": False,
        "trial_index": 0,
        "trial_order": [],
        "trial_start_time": None,
        "responses": [],
        "last_feedback": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def ensure_responses_file() -> None:
    if not RESPONSES_PATH.exists():
        pd.DataFrame(
            columns=[
                "timestamp",
                "participant_id",
                "trial_no",
                "item_id",
                "scenario",
                "text",
                "true_source",
                "participant_choice",
                "correct",
                "confidence",
                "warmth_rating",
                "rt_ms",
            ]
        ).to_csv(RESPONSES_PATH, index=False, encoding="utf-8-sig")


def clear_responses_file() -> None:
    pd.DataFrame(
        columns=[
            "timestamp",
            "participant_id",
            "trial_no",
            "item_id",
            "scenario",
            "text",
            "true_source",
            "participant_choice",
            "correct",
            "confidence",
            "warmth_rating",
            "rt_ms",
        ]
    ).to_csv(RESPONSES_PATH, index=False, encoding="utf-8-sig")


def load_stimuli() -> pd.DataFrame:
    encodings_to_try = ["utf-8-sig", "utf-8", "gbk", "gb18030"]

    if not STIMULI_PATH.exists():
        demo_df = pd.DataFrame(
            [
                {
                    "item_id": 1,
                    "scenario": "朋友考研失败后很沮丧",
                    "text": "这次没上岸肯定特别难受，先别急着逼自己想开。缓两天也没关系，等你想聊了我们再慢慢看下一步。",
                    "true_source": "human",
                },
                {
                    "item_id": 2,
                    "scenario": "朋友第一次考编落榜后特别自责",
                    "text": "落榜以后第一反应就是怪自己，这很常见。但一次结果真的说明不了太多，你先歇一下，别急着给自己下结论。",
                    "true_source": "ai",
                },
            ]
        )
        demo_df.to_csv(STIMULI_PATH, index=False, encoding="utf-8-sig")

    last_error = None
    df = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(STIMULI_PATH, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_error = e

    if df is None:
        raise last_error

    required_cols = {"item_id", "scenario", "text", "true_source"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"stimuli.csv 缺少字段: {missing}")

    df["true_source"] = df["true_source"].astype(str).str.lower().str.strip()
    return df


@st.cache_data(show_spinner=False)
def summarize_data() -> dict:
    ensure_responses_file()
    df = pd.read_csv(RESPONSES_PATH, encoding="utf-8-sig")

    if df.empty:
        return {
            "n_rows": 0,
            "n_participants": 0,
            "accuracy": None,
            "by_source": pd.DataFrame(),
            "recent": pd.DataFrame(),
        }

    accuracy = float(df["correct"].mean()) if "correct" in df.columns else None

    by_source = (
        df.groupby("true_source", as_index=False)
        .agg(
            acc=("correct", "mean"),
            rt_ms=("rt_ms", "mean"),
            warmth=("warmth_rating", "mean"),
        )
    )

    recent = df.tail(20)

    return {
        "n_rows": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "accuracy": accuracy,
        "by_source": by_source,
        "recent": recent,
    }


def save_responses(rows: list[dict]) -> None:
    if not rows:
        return

    ensure_responses_file()
    df_new = pd.DataFrame(rows)
    df_old = pd.read_csv(RESPONSES_PATH, encoding="utf-8-sig")
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all.to_csv(RESPONSES_PATH, index=False, encoding="utf-8-sig")


def start_experiment(stimuli: pd.DataFrame) -> None:
    order = list(stimuli.index)
    random.shuffle(order)

    st.session_state.trial_order = order
    st.session_state.trial_index = 0
    st.session_state.responses = []
    st.session_state.started = True
    st.session_state.finished = False
    st.session_state.last_feedback = None
    st.session_state.trial_start_time = time.perf_counter()


def reset_experiment() -> None:
    st.session_state.started = False
    st.session_state.finished = False
    st.session_state.trial_index = 0
    st.session_state.trial_order = []
    st.session_state.trial_start_time = None
    st.session_state.responses = []
    st.session_state.last_feedback = None
    st.session_state.participant_id = ""
    st.session_state.consented = False


# =========================
# UI init
# =========================
init_state()
ensure_responses_file()
stimuli = load_stimuli()

st.title("🧠 安慰语图灵测试")
st.caption("判断一段安慰语是 AI 写的，还是人写的。")


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.subheader("控制面板")

    admin_mode = st.checkbox("讲台模式 / Admin", value=st.session_state.admin_mode)
    st.session_state.admin_mode = admin_mode

    if st.button("刷新汇总"):
        summarize_data.clear()
        st.rerun()

    if st.button("重新加载题库"):
        summarize_data.clear()
        reset_experiment()
        st.rerun()

    if st.button("重置我的作答进度"):
        reset_experiment()
        st.rerun()

    if admin_mode:
        if st.button("清空测试数据"):
            clear_responses_file()
            summarize_data.clear()
            reset_experiment()
            st.success("测试数据已清空。")
            st.rerun()

    st.markdown("---")
    st.write(f"刺激文件路径: {STIMULI_PATH}")
    st.write(f"数据文件路径: {RESPONSES_PATH}")

    if RESPONSES_PATH.exists():
        with open(RESPONSES_PATH, "rb") as f:
            st.download_button(
                "下载 responses.csv",
                data=f,
                file_name="responses.csv",
                mime="text/csv",
            )

    if STIMULI_PATH.exists():
        with open(STIMULI_PATH, "rb") as f:
            st.download_button(
                "下载 stimuli.csv",
                data=f,
                file_name="stimuli.csv",
                mime="text/csv",
            )


# =========================
# Admin panel
# =========================
if st.session_state.admin_mode:
    stats = summarize_data()

    st.markdown("## 讲台实时结果")
    c1, c2, c3 = st.columns(3)
    c1.metric("已收集作答", stats["n_rows"])
    c2.metric("参与人数", stats["n_participants"])
    c3.metric(
        "平均正确率",
        "-" if stats["accuracy"] is None else f"{stats['accuracy'] * 100:.1f}%",
    )

    if not stats["by_source"].empty:
        st.markdown("### 来源类型上的平均表现")
        st.dataframe(stats["by_source"], use_container_width=True)

    if not stats["recent"].empty:
        st.markdown("### 最近数据")
        st.dataframe(stats["recent"], use_container_width=True)

st.markdown("---")


# =========================
# Main experiment flow
# =========================
if not st.session_state.consented:
    st.markdown("## 参与说明")
    st.write(
        "你将看到若干条安慰语。请判断它更像是 AI 写的还是人写的，"
        "并给出自信评分与温暖度评分。全程大约 2–4 分钟。"
    )

    pid = st.text_input("请输入你的被试编号", value=st.session_state.participant_id, max_chars=30)
    consent = st.checkbox("我同意将本次匿名作答用于课堂演示与统计")

    if st.button("开始测试", type="primary"):
        if not pid.strip():
            st.error("请先输入被试编号。")
        elif not consent:
            st.error("请先勾选同意。")
        else:
            st.session_state.participant_id = pid.strip()
            st.session_state.consented = True
            start_experiment(stimuli)
            st.rerun()

elif st.session_state.finished:
    st.success("测试完成，感谢参与。")

    df_me = pd.DataFrame(st.session_state.responses)
    if not df_me.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("你的正确率", f"{df_me['correct'].mean() * 100:.1f}%")
        c2.metric("平均反应时", f"{df_me['rt_ms'].mean():.0f} ms")
        c3.metric("平均温暖评分", f"{df_me['warmth_rating'].mean():.2f}")

        st.dataframe(
            df_me[
                [
                    "trial_no",
                    "true_source",
                    "participant_choice",
                    "correct",
                    "confidence",
                    "warmth_rating",
                    "rt_ms",
                ]
            ],
            use_container_width=True,
        )

    if st.button("重新开始"):
        reset_experiment()
        st.rerun()

else:
    current_pos = st.session_state.trial_index
    total_n = len(st.session_state.trial_order)

    if current_pos >= total_n:
        save_responses(st.session_state.responses)
        summarize_data.clear()
        st.session_state.finished = True
        st.rerun()

    item_idx = st.session_state.trial_order[current_pos]
    row = stimuli.iloc[item_idx]

    st.progress((current_pos + 1) / total_n)
    st.markdown(f"## 第 {current_pos + 1} / {total_n} 题")
    st.markdown(f"**情境：** {row['scenario']}")
    st.info(row["text"])

    if st.session_state.last_feedback:
        fb = st.session_state.last_feedback
        msg = (
            f"上一题：你的判断 = {fb['participant_choice']}；"
            f"正确答案 = {fb['true_source']}；"
            f"结果 = {'正确' if fb['correct'] else '错误'}"
        )
        if fb["correct"]:
            st.success(msg)
        else:
            st.warning(msg)

    with st.form(key=f"trial_form_{current_pos}"):
        participant_choice = st.radio(
            "你觉得这段安慰语更像谁写的？",
            options=["ai", "human"],
            format_func=lambda x: "AI" if x == "ai" else "真人",
            horizontal=True,
        )

        confidence = st.slider("你有多自信？", min_value=1, max_value=5, value=3)
        warmth_rating = st.slider("这段话让你觉得多温暖？", min_value=1, max_value=5, value=3)

        submitted = st.form_submit_button("提交并进入下一题", type="primary")

        if submitted:
            rt_ms = int((time.perf_counter() - st.session_state.trial_start_time) * 1000)
            correct = int(participant_choice == row["true_source"])

            trial_record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "participant_id": st.session_state.participant_id,
                "trial_no": current_pos + 1,
                "item_id": int(row["item_id"]),
                "scenario": row["scenario"],
                "text": row["text"],
                "true_source": row["true_source"],
                "participant_choice": participant_choice,
                "correct": correct,
                "confidence": confidence,
                "warmth_rating": warmth_rating,
                "rt_ms": rt_ms,
            }

            st.session_state.responses.append(trial_record)
            st.session_state.last_feedback = {
                "participant_choice": participant_choice,
                "true_source": row["true_source"],
                "correct": bool(correct),
            }
            st.session_state.trial_index += 1
            st.session_state.trial_start_time = time.perf_counter()
            st.rerun()

st.markdown("---")
st.caption("提示：程序每次刷新都会重新读取 data/stimuli.csv。修改题库后，点击左侧“重新加载题库”或“重置我的作答进度”即可生效。")