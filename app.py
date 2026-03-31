import os
import json
import time
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# =========================
# Optional: generate AI stimuli via Kimi / Moonshot API
# Official docs indicate Kimi API is compatible with the OpenAI SDK.
# Set base_url to https://api.moonshot.ai/v1 and use model='kimi-k2.5'.
# =========================
def generate_ai_stimuli_via_kimi(scenarios: list[str], n_per_scenario: int = 2, model: str = "kimi-k2.5") -> pd.DataFrame:
    """
    Requires environment variable MOONSHOT_API_KEY.

    Windows PowerShell:
        $env:MOONSHOT_API_KEY="your_key_here"

    macOS / Linux:
        export MOONSHOT_API_KEY="your_key_here"
    """
    if OpenAI is None:
        raise ImportError("未安装 openai。请先运行: pip install openai>=1.0")

    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise ValueError("未检测到 MOONSHOT_API_KEY 环境变量。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.ai/v1",
    )

    rows = []
    next_item_id = 1000
    for scenario in scenarios:
        prompt = (
            "你要为一个心理学课堂实验生成安慰语刺激。"
            "请围绕给定情境，写出简短、自然、口语化的中文安慰语。"
            f"每条 35-70 字，生成 {n_per_scenario} 条。"
            "不要编号，不要解释，不要重复，不要使用过度书面化表达。"
            f"情境：{scenario}"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是实验材料生成助手。输出纯文本，每行一条刺激。"},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
        )

        content = (resp.choices[0].message.content or "").strip()
        lines = [x.strip("-• ") for x in content.split("") if x.strip()]

        for line in lines[:n_per_scenario]:
            rows.append(
                {
                    "item_id": next_item_id,
                    "scenario": scenario,
                    "text": line,
                    "true_source": "ai",
                }
            )
            next_item_id += 1

    return pd.DataFrame(rows)


def append_kimi_stimuli_to_csv(scenarios: list[str], n_per_scenario: int = 2, model: str = "kimi-k2.5") -> pd.DataFrame:
    """
    Convenience helper:
    1. generate AI stimuli from Kimi
    2. append them to data/stimuli.csv
    3. return the appended rows
    """
    new_df = generate_ai_stimuli_via_kimi(
        scenarios=scenarios,
        n_per_scenario=n_per_scenario,
        model=model,
    )

    if STIMULI_PATH.exists():
        old_df = pd.read_csv(STIMULI_PATH)
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df.copy()

    merged.to_csv(STIMULI_PATH, index=False, encoding="utf-8-sig")
    load_stimuli.clear()
    return new_df


# =========================
# UI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="安慰语图灵测试",
    page_icon="🧠",
    layout="centered",
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
def summarize_data() -> dict:
    ensure_responses_file()
    df = pd.read_csv(RESPONSES_PATH)
    if df.empty:
        return {
            "n_rows": 0,
            "n_participants": 0,
            "accuracy": None,
            "human_rate": None,
            "ai_rate": None,
        }

    accuracy = float(df["correct"].mean()) if "correct" in df.columns else None
    human_rate = float((df["true_source"] == "human").mean())
    ai_rate = float((df["true_source"] == "ai").mean())
    return {
        "n_rows": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "accuracy": accuracy,
        "human_rate": human_rate,
        "ai_rate": ai_rate,
    }


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
    for key in [
        "started",
        "finished",
        "trial_index",
        "trial_order",
        "trial_start_time",
        "responses",
        "last_feedback",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    init_state()


# =========================
# Optional: generate AI stimuli via API
# =========================
def generate_ai_stimuli_via_openai(scenarios: list[str], n_per_scenario: int = 2, model: str = "gpt-4.1-mini") -> pd.DataFrame:
    """
    Requires environment variable OPENAI_API_KEY.
    Example:
        export OPENAI_API_KEY=...
    """
    if OpenAI is None:
        raise ImportError("未安装 openai。请先运行: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未检测到 OPENAI_API_KEY 环境变量。")

    client = OpenAI(api_key=api_key)

    rows = []
    next_item_id = 1000
    for scenario in scenarios:
        prompt = (
            "你要为一个心理学课堂实验生成安慰语刺激。"
            "请围绕给定情境，写出简短、自然、口语化的中文安慰语。"
            f"每条 35-70 字，生成 {n_per_scenario} 条，不要编号，不要解释，不要重复。"
            f"情境：{scenario}"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是实验材料生成助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
        )
        content = resp.choices[0].message.content.strip()
        lines = [x.strip("-• \n\t") for x in content.split("\n") if x.strip()]
        for line in lines[:n_per_scenario]:
            rows.append(
                {
                    "item_id": next_item_id,
                    "scenario": scenario,
                    "text": line,
                    "true_source": "ai",
                }
            )
            next_item_id += 1

    return pd.DataFrame(rows)


# =========================
# UI
# =========================
init_state()
ensure_responses_file()
stimuli = load_stimuli()

st.title("🧠 安慰语图灵测试")
st.caption("判断一段安慰语是 AI 写的，还是人写的。")

with st.sidebar:
    st.subheader("控制面板")
    admin_mode = st.checkbox("讲台模式 / Admin", value=st.session_state.admin_mode)
    st.session_state.admin_mode = admin_mode

    if st.button("刷新汇总"):
        summarize_data.clear()
        st.rerun()

    if st.button("重新加载题库"):
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

    if admin_mode:
        st.markdown("---")
        st.write("刺激文件路径:", str(STIMULI_PATH))
        st.write("数据文件路径:", str(RESPONSES_PATH))
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

if st.session_state.admin_mode:
    st.markdown("## 讲台实时结果")
    stats = summarize_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("已收集作答", stats["n_rows"])
    c2.metric("参与人数", stats["n_participants"])
    c3.metric("平均正确率", "-" if stats["accuracy"] is None else f"{stats['accuracy']*100:.1f}%")

    if RESPONSES_PATH.exists() and RESPONSES_PATH.stat().st_size > 0:
        df_admin = pd.read_csv(RESPONSES_PATH)
        if not df_admin.empty:
            st.markdown("### 来源类型上的平均表现")
            by_source = (
                df_admin.groupby("true_source", as_index=False)
                .agg(acc=("correct", "mean"), rt_ms=("rt_ms", "mean"), warmth=("warmth_rating", "mean"))
            )
            st.dataframe(by_source, use_container_width=True)

            st.markdown("### 最近数据")
            st.dataframe(df_admin.tail(20), use_container_width=True)

st.markdown("---")

if not st.session_state.consented:
    st.markdown("### 参与说明")
    st.write(
        "你将看到若干条安慰语。请判断它更像是 AI 写的还是人写的，"
        "并给出自信评分与温暖度评分。全程大约 2-4 分钟。"
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
        acc = df_me["correct"].mean()
        mean_rt = df_me["rt_ms"].mean()
        mean_warmth = df_me["warmth_rating"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("你的正确率", f"{acc*100:.1f}%")
        c2.metric("平均反应时", f"{mean_rt:.0f} ms")
        c3.metric("平均温暖评分", f"{mean_warmth:.2f}")

        st.dataframe(df_me[["trial_no", "true_source", "participant_choice", "correct", "confidence", "warmth_rating", "rt_ms"]], use_container_width=True)

    if st.button("重新开始"):
        reset_experiment()
        st.rerun()

else:
    current_pos = st.session_state.trial_index
    total_n = len(st.session_state.trial_order)

    if current_pos >= total_n:
        save_responses(st.session_state.responses)
        st.session_state.finished = True
        summarize_data.clear()
        st.rerun()

    item_idx = st.session_state.trial_order[current_pos]
    row = stimuli.iloc[item_idx]

    st.progress((current_pos + 1) / total_n)
    st.markdown(f"### 第 {current_pos + 1} / {total_n} 题")
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
st.caption(
    "提示：程序每次刷新都会重新读取 data/stimuli.csv。修改题库后，点击左侧“重新加载题库”或“重置我的作答进度”即可生效。"
)

# =========================
# Example stimuli.csv format
# =========================
# item_id,scenario,text,true_source
# 1,朋友考研失败后很沮丧,这次没有上岸一定很难受...,human
# 2,朋友考研失败后很沮丧,你现在感到失落是完全可以理解的...,ai
