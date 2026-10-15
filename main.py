import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 마지막 연도
MIN_DAYS = 300        # 연평균 계산에 필요한 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")
st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온의 추세를 직선(선형 회귀)으로 맞추고, 고른 해의 기온을 예상합니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df = df[df["날짜"].dt.year <= LAST_YEAR]

    yearly = (
        df.groupby(df["날짜"].dt.year)["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
        .rename(columns={"날짜": "연도"})
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS].reset_index(drop=True)
    return yearly


try:
    yearly = load_yearly(DATA_URL)
except Exception as e:  # 네트워크/형식 오류
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 3:
    st.error("조건을 만족하는 연도가 너무 적어 회귀 직선을 만들 수 없습니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# ---- 직선 정보 ----
st.subheader("회귀 직선에 사용한 데이터")
c1, c2, c3 = st.columns(3)
c1.metric("사용한 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
st.caption(
    f"{LAST_YEAR}년 이후 데이터와 관측일이 {MIN_DAYS}일 미만인 해는 제외했습니다."
)

# ---- 슬라이더 & 예상 기온 ----
st.subheader("예상 기온")
year = st.slider("연도를 고르세요", min_value=1900, max_value=2100, value=2030, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
      <div style="font-size:1.1rem; opacity:0.7;">{year}년 예상 연평균기온</div>
      <div style="font-size:4.5rem; font-weight:800; line-height:1.2;">{pred:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if year < start_year or year > end_year:
    st.caption("⚠️ 선택한 연도가 직선을 만든 기간 밖이라, 추세를 연장한 값입니다.")

# ---- 산점도 + 회귀 직선 ----
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
x_lo, x_hi = min(start_year, year), max(end_year, year)
line_x = np.array([x_lo, x_hi], dtype=float)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=slope * line_x + intercept,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, symbol="star", line=dict(width=1, color="black")),
        hovertemplate=f"{year}년 예상<br>{pred:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=40, b=10),
    height=480,
)
st.plotly_chart(fig, use_container_width=True)

# ---- 상관계수 ----
st.subheader("상관 분석")
m1, m2, m3 = st.columns(3)
m1.metric("상관계수 (r)", f"{r:.3f}")
m2.metric("결정계수 (R²)", f"{r**2:.3f}")
m3.metric("10년당 변화", f"{slope * 10:+.2f} °C")
st.markdown(f"회귀식: **기온 = {slope:.4f} × 연도 {intercept:+.2f}**")
