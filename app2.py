# -*- coding: utf-8 -*-
"""
서울시 부동산 실거래가 분석 대시보드
- EDA, 시계열 분석, 머신러닝 가격 예측을 제공하는 Streamlit 앱
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------
# 기본 설정
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="서울시 부동산 실거래가 분석",
    page_icon="🏢",
    layout="wide",
)

DEFAULT_CSV_NAME = "서울시_부동산_실거래가_정보_202606.csv"
USE_COLS = [
    "계약일", "자치구명", "법정동명", "건물면적(㎡)", "토지면적(㎡)",
    "층", "건축년도", "건물용도", "건물명", "신고구분", "물건금액(만원)",
]


# -----------------------------------------------------------------------
# 데이터 로딩 & 전처리
# -----------------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data(file) -> pd.DataFrame:
    """CSV를 읽고 노트북과 동일한 전처리(결측치/중복/이상치 제거)를 적용."""
    try:
        df = pd.read_csv(file, encoding="cp949")
    except UnicodeDecodeError:
        file.seek(0) if hasattr(file, "seek") else None
        df = pd.read_csv(file, encoding="utf-8")

    # 필요한 컬럼만 사용 (없는 컬럼은 무시)
    cols = [c for c in USE_COLS if c in df.columns]
    df = df[cols].copy()

    # 결측치 제거
    df = df.dropna()

    # 중복 제거
    df = df.drop_duplicates()

    # 건물면적이 0 이하인 행 제거
    df = df[df["건물면적(㎡)"] > 0]

    # 건축년도가 비정상(0 등)인 행 제거
    df = df[df["건축년도"] > 1900]

    # 이상치(IQR) 제거 - 물건금액 기준
    q1 = df["물건금액(만원)"].quantile(0.25)
    q3 = df["물건금액(만원)"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df = df[(df["물건금액(만원)"] >= lower) & (df["물건금액(만원)"] <= upper)]

    # 파생 변수
    df["면적당금액"] = df["물건금액(만원)"] / df["건물면적(㎡)"]
    df["계약일"] = pd.to_datetime(df["계약일"].astype(int).astype(str), format="%Y%m%d")
    df["연도"] = df["계약일"].dt.year
    df["월"] = df["계약일"].dt.month

    df = df.reset_index(drop=True)
    return df


@st.cache_resource(show_spinner="모델을 학습하는 중입니다... (최초 1회만 수행)")
def train_models(df: pd.DataFrame):
    """범주형 컬럼을 인코딩하고 여러 회귀 모델을 학습 후 성능을 비교."""
    cat_cols = ["자치구명", "법정동명", "건물용도", "건물명", "신고구분"]
    encoders = {}
    enc_df = df.copy()
    for c in cat_cols:
        le = LabelEncoder()
        enc_df[c] = le.fit_transform(enc_df[c].astype(str))
        encoders[c] = le

    feature_cols = [
        "자치구명", "법정동명", "건물면적(㎡)", "토지면적(㎡)",
        "층", "건축년도", "건물용도", "건물명", "신고구분",
    ]
    X = enc_df[feature_cols]
    y = enc_df["면적당금액"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(random_state=42),
    }

    results = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        results.append([name, mae, rmse, r2])
        fitted_models[name] = model

    result_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
    result_df = result_df.sort_values(by="R2", ascending=False).reset_index(drop=True)
    best_name = result_df.iloc[0]["Model"]

    return {
        "encoders": encoders,
        "feature_cols": feature_cols,
        "models": fitted_models,
        "result_df": result_df,
        "best_name": best_name,
        "X_test": X_test,
        "y_test": y_test,
    }


# -----------------------------------------------------------------------
# 사이드바: 데이터 소스 & 필터
# -----------------------------------------------------------------------
st.sidebar.title("🏢 메뉴")

uploaded = st.sidebar.file_uploader("CSV 파일 업로드 (선택)", type=["csv"])

data_source = None
if uploaded is not None:
    data_source = uploaded
elif os.path.exists(DEFAULT_CSV_NAME):
    data_source = DEFAULT_CSV_NAME
elif os.path.exists(os.path.join("dataset", DEFAULT_CSV_NAME)):
    data_source = os.path.join("dataset", DEFAULT_CSV_NAME)

if data_source is None:
    st.warning(
        f"`{DEFAULT_CSV_NAME}` 파일을 앱과 같은 폴더에 두거나, "
        "왼쪽 사이드바에서 CSV 파일을 업로드해주세요."
    )
    st.stop()

df_all = load_data(data_source)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 필터")

gu_options = sorted(df_all["자치구명"].unique().tolist())
sel_gu = st.sidebar.multiselect("자치구", gu_options, default=gu_options)

dong_pool = df_all[df_all["자치구명"].isin(sel_gu)]["법정동명"].unique().tolist()
dong_options = sorted(dong_pool)
sel_dong = st.sidebar.multiselect("법정동", dong_options, default=dong_options)

usage_options = sorted(df_all["건물용도"].unique().tolist())
sel_usage = st.sidebar.multiselect("건물용도", usage_options, default=usage_options)

report_options = sorted(df_all["신고구분"].unique().tolist())
sel_report = st.sidebar.multiselect("신고구분", report_options, default=report_options)

year_min, year_max = int(df_all["연도"].min()), int(df_all["연도"].max())
if year_min == year_max:
    st.sidebar.caption(f"거래 연도: {year_min}년 (단일 연도 데이터)")
    sel_year = (year_min, year_max)
else:
    sel_year = st.sidebar.slider("거래 연도", year_min, year_max, (year_min, year_max))

price_min, price_max = float(df_all["물건금액(만원)"].min()), float(df_all["물건금액(만원)"].max())
sel_price = st.sidebar.slider(
    "물건금액 (만원)", price_min, price_max, (price_min, price_max)
)

# 필터 적용
df = df_all[
    df_all["자치구명"].isin(sel_gu)
    & df_all["법정동명"].isin(sel_dong)
    & df_all["건물용도"].isin(sel_usage)
    & df_all["신고구분"].isin(sel_report)
    & df_all["연도"].between(sel_year[0], sel_year[1])
    & df_all["물건금액(만원)"].between(sel_price[0], sel_price[1])
].copy()

st.sidebar.markdown("---")
st.sidebar.metric("필터링된 거래 건수", f"{len(df):,} 건")


# -----------------------------------------------------------------------
# 메인 타이틀
# -----------------------------------------------------------------------
st.title("🏢 서울시 부동산 실거래가 분석 대시보드")
st.caption("출처: 서울시 부동산 실거래가 정보 (전처리: 결측치/중복/이상치 제거)")

if df.empty:
    st.error("선택한 필터 조건에 해당하는 데이터가 없습니다. 필터를 조정해주세요.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 데이터 개요", "📊 탐색적 분석", "📈 시계열 분석", "🤖 모델 비교 & 예측"]
)

# -----------------------------------------------------------------------
# Tab 1. 데이터 개요
# -----------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 거래 건수", f"{len(df):,} 건")
    c2.metric("평균 거래금액", f"{df['물건금액(만원)'].mean():,.0f} 만원")
    c3.metric("평균 면적당금액", f"{df['면적당금액'].mean():,.1f} 만원/㎡")
    c4.metric("평균 건물면적", f"{df['건물면적(㎡)'].mean():,.1f} ㎡")

    st.markdown("### 데이터 미리보기")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("### 기초 통계량")
    st.dataframe(df.describe(), use_container_width=True)

    st.caption("※ 전처리(결측치/중복/이상치 제거) 후 데이터로, 현재 결측치는 없습니다.")

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 필터링된 데이터 CSV 다운로드",
        data=csv_bytes,
        file_name="filtered_real_estate.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------
# Tab 2. 탐색적 분석 (EDA)
# -----------------------------------------------------------------------
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 면적당금액 분포")
        fig = px.histogram(
            df, x="면적당금액", nbins=50, marginal="box",
            color_discrete_sequence=["#e74c3c"],
        )
        fig.update_layout(xaxis_title="면적당금액(만원/㎡)", yaxis_title="건수")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 건물용도별 거래금액 분포")
        fig = px.box(
            df, x="건물용도", y="물건금액(만원)", color="건물용도",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 건물면적 vs 거래금액")
        fig = px.scatter(
            df, x="건물면적(㎡)", y="물건금액(만원)", color="건물용도",
            opacity=0.6, hover_data=["법정동명", "건축년도"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### 자치구별 평균 거래금액")
        gu_avg = (
            df.groupby("자치구명")["물건금액(만원)"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = px.bar(
            gu_avg, x="자치구명", y="물건금액(만원)",
            color_discrete_sequence=["#e74c3c"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 수치형 변수 상관관계")
    numeric_cols = ["건물면적(㎡)", "토지면적(㎡)", "층", "건축년도", "물건금액(만원)", "면적당금액"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="Reds", aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 건물용도별 거래 건수")
    usage_count = df["건물용도"].value_counts().reset_index()
    usage_count.columns = ["건물용도", "건수"]
    fig = px.pie(usage_count, names="건물용도", values="건수", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Tab 3. 시계열 분석
# -----------------------------------------------------------------------
with tab3:
    st.markdown("#### 일별 평균 거래금액 추이 (30일 이동평균)")
    daily = df.groupby("계약일")["면적당금액"].mean().sort_index()

    if len(daily) >= 2:
        rolling = daily.rolling(min(30, max(2, len(daily) // 2)), min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily.index, y=daily.values, mode="lines",
                                  name="일별 평균", line=dict(color="lightgray")))
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode="lines",
                                  name="이동평균", line=dict(color="#e74c3c", width=3)))
        fig.update_layout(xaxis_title="계약일", yaxis_title="면적당금액(만원/㎡)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("일별 추이를 그리기에 데이터 포인트가 부족합니다.")

    st.markdown("#### 월별 평균 거래금액")
    monthly = (
        df.set_index("계약일").resample("M")["물건금액(만원)"].mean().dropna()
    )
    if len(monthly) >= 1:
        fig = px.line(
            x=monthly.index, y=monthly.values, markers=True,
            labels={"x": "월", "y": "평균 거래금액(만원)"},
        )
        fig.update_traces(line_color="#e74c3c")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("월별 데이터가 충분하지 않습니다.")

    st.markdown("#### 건축년도별 평균 면적당금액")
    by_year_built = (
        df.groupby("건축년도")["면적당금액"].mean().reset_index().sort_values("건축년도")
    )
    fig = px.bar(by_year_built, x="건축년도", y="면적당금액",
                 color_discrete_sequence=["#e74c3c"])
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Tab 4. 모델 비교 & 가격 예측
# -----------------------------------------------------------------------
with tab4:
    st.markdown("### 회귀 모델 성능 비교")
    st.caption("전체(필터 적용 전) 전처리 데이터를 기준으로 모델을 학습합니다.")

    if len(df_all) < 50:
        st.warning("모델 학습을 위한 데이터가 너무 적습니다 (최소 50건 이상 권장).")
    else:
        bundle = train_models(df_all)
        result_df = bundle["result_df"]
        best_name = bundle["best_name"]

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(
                result_df, x="Model", y="R2", color_discrete_sequence=["#e74c3c"],
                text=result_df["R2"].round(3),
            )
            fig.update_layout(yaxis_title="R² Score", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**모델별 성능 지표**")
            st.dataframe(
                result_df.style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}", "R2": "{:.3f}"}),
                use_container_width=True,
            )
            st.success(f"최적 모델: **{best_name}** (R² = {result_df.iloc[0]['R2']:.3f})")

        st.markdown("---")
        st.markdown("### 💰 거래금액 예측")
        st.caption(f"선택한 조건으로 '{best_name}' 모델을 사용해 면적당금액 및 예상 거래금액을 추정합니다.")

        encoders = bundle["encoders"]
        feature_cols = bundle["feature_cols"]
        best_model = bundle["models"][best_name]

        with st.form("predict_form"):
            f1, f2, f3 = st.columns(3)
            with f1:
                in_gu = st.selectbox("자치구명", sorted(encoders["자치구명"].classes_))
                in_dong = st.selectbox("법정동명", sorted(encoders["법정동명"].classes_))
                in_usage = st.selectbox("건물용도", sorted(encoders["건물용도"].classes_))
            with f2:
                in_building = st.selectbox("건물명", sorted(encoders["건물명"].classes_))
                in_report = st.selectbox("신고구분", sorted(encoders["신고구분"].classes_))
                in_floor = st.number_input("층", value=1, step=1)
            with f3:
                in_area = st.number_input("건물면적(㎡)", min_value=1.0, value=60.0, step=1.0)
                in_land = st.number_input("토지면적(㎡)", min_value=0.0, value=30.0, step=1.0)
                year_built_min = int(df_all["건축년도"].min())
                year_built_max = max(int(df_all["건축년도"].max()), 2026)
                default_year_built = int(df_all["건축년도"].median())
                in_year_built = st.number_input(
                    "건축년도",
                    min_value=year_built_min,
                    max_value=year_built_max,
                    value=min(max(default_year_built, year_built_min), year_built_max),
                    step=1,
                )

            submitted = st.form_submit_button("예측하기")

        if submitted:
            row = pd.DataFrame([{
                "자치구명": encoders["자치구명"].transform([in_gu])[0],
                "법정동명": encoders["법정동명"].transform([in_dong])[0],
                "건물면적(㎡)": in_area,
                "토지면적(㎡)": in_land,
                "층": in_floor,
                "건축년도": in_year_built,
                "건물용도": encoders["건물용도"].transform([in_usage])[0],
                "건물명": encoders["건물명"].transform([in_building])[0],
                "신고구분": encoders["신고구분"].transform([in_report])[0],
            }])[feature_cols]

            pred_unit_price = best_model.predict(row)[0]
            pred_total_price = pred_unit_price * in_area

            r1, r2 = st.columns(2)
            r1.metric("예상 면적당금액", f"{pred_unit_price:,.1f} 만원/㎡")
            r2.metric("예상 거래금액", f"{pred_total_price:,.0f} 만원")
