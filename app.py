import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, concatenate,
    Conv1D, MaxPooling1D, Flatten, LSTM, GRU
)
from tensorflow.keras.callbacks import EarlyStopping

# ────────────────────────────────────────────────────────
# 페이지 설정
# ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="서울시 부동산 실거래가 분석",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 서울시 부동산 실거래가 예측 분석")
st.markdown("머신러닝 · 딥러닝 · 시계열 모델을 비교 분석합니다.")

# ────────────────────────────────────────────────────────
# 데이터 로드 & 전처리
# ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding="cp949")

    cols = ["계약일", "자치구명", "법정동명", "건물면적(㎡)", "토지면적(㎡)",
            "층", "건축년도", "건물용도", "건물명", "신고구분", "물건금액(만원)"]
    df = df[cols].copy()

    # 결측치·중복 제거
    df = df.dropna().drop_duplicates()
    df = df[df["건물면적(㎡)"] > 0]

    # 이상치(IQR)
    Q1 = df["물건금액(만원)"].quantile(0.25)
    Q3 = df["물건금액(만원)"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["물건금액(만원)"] >= Q1 - 1.5*IQR) &
            (df["물건금액(만원)"] <= Q3 + 1.5*IQR)]

    # 파생변수
    df["면적당금액"] = df["물건금액(만원)"] / df["건물면적(㎡)"]
    df["계약일"] = pd.to_datetime(df["계약일"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["계약일"])
    df["연도"] = df["계약일"].dt.year
    df["월"] = df["계약일"].dt.month
    df["계약년월"] = df["계약일"].dt.to_period("M").astype(str)

    # 인코딩
    cats = ["자치구명", "법정동명", "건물용도", "건물명", "신고구분"]
    encoders = {}
    for c in cats:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    return df, encoders


@st.cache_data(show_spinner=False)
def split_data(df, test_size):
    X = df.drop(columns=["계약일", "물건금액(만원)", "계약년월", "면적당금액"])
    y = df["면적당금액"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_tr, X_te, y_tr, y_te, X.columns.tolist()


def evaluate_model(y_true, y_pred, name):
    return {
        "Model": name,
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R²":   round(r2_score(y_true, y_pred), 4),
    }


# ────────────────────────────────────────────────────────
# 사이드바
# ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    uploaded = st.file_uploader(
        "CSV 파일 업로드",
        type=["csv"],
        help="서울시 부동산 실거래가 CSV (cp949 인코딩)"
    )
    st.divider()
    test_size  = st.slider("테스트 비율", 0.1, 0.4, 0.2, 0.05)
    epochs_dl  = st.slider("딥러닝 에포크", 10, 100, 30, 10)
    batch_size = st.selectbox("배치 사이즈", [32, 64, 128], index=0)
    st.divider()
    use_ml  = st.checkbox("머신러닝 학습", value=True)
    use_dl  = st.checkbox("딥러닝 학습", value=True)
    use_ts  = st.checkbox("시계열 분석 (Prophet)", value=False)
    st.divider()
    run_btn = st.button("🚀 분석 시작", use_container_width=True, type="primary")

# ────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────
if not uploaded:
    st.info("👈 사이드바에서 CSV 파일을 업로드하고 **분석 시작** 버튼을 눌러주세요.")
    st.stop()

# 데이터 로드
with st.spinner("데이터 로드 중..."):
    df, encoders = load_and_preprocess(uploaded)

# 탭 구성
tab_eda, tab_ml, tab_dl, tab_cmp, tab_ts = st.tabs([
    "📊 데이터 탐색", "🤖 머신러닝", "🧠 딥러닝", "📈 모델 비교", "📅 시계열"
])

# ────────────────────────────────────────────────────────
# TAB 1: EDA
# ────────────────────────────────────────────────────────
with tab_eda:
    st.subheader("📊 데이터 탐색 (EDA)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 데이터 수", f"{len(df):,}")
    c2.metric("자치구 수", df["자치구명"].nunique())
    c3.metric("평균 면적당금액 (만원/㎡)", f"{df['면적당금액'].mean():,.1f}")
    c4.metric("연도 범위",
              f"{int(df['연도'].min())} ~ {int(df['연도'].max())}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**면적당금액 분포**")
        fig = px.histogram(df, x="면적당금액", nbins=60, color_discrete_sequence=["#EF4444"])
        fig.update_layout(height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**자치구별 평균 면적당금액**")
        gu_price = (df.groupby("자치구명")["면적당금액"]
                    .mean().sort_values(ascending=False).reset_index())
        fig = px.bar(gu_price, x="자치구명", y="면적당금액",
                     color="면적당금액", color_continuous_scale="Reds")
        fig.update_layout(height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**월별 평균 면적당금액 추이**")
        monthly = (df.groupby("계약년월")["면적당금액"]
                   .mean().reset_index().sort_values("계약년월"))
        fig = px.line(monthly, x="계약년월", y="면적당금액",
                      color_discrete_sequence=["#EF4444"], markers=True)
        fig.update_layout(height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("**건물용도별 거래 비율**")
        use_cnt = df["건물용도"].value_counts().reset_index()
        use_cnt.columns = ["건물용도", "건수"]
        fig = px.pie(use_cnt, names="건물용도", values="건수",
                     color_discrete_sequence=px.colors.sequential.Reds_r)
        fig.update_layout(height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**상관관계 히트맵**")
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    aspect="auto")
    fig.update_layout(height=500, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("원본 데이터 미리보기"):
        st.dataframe(df.head(200), use_container_width=True)

# ────────────────────────────────────────────────────────
# 분석 실행
# ────────────────────────────────────────────────────────
if not run_btn:
    with tab_ml:
        st.info("사이드바에서 **분석 시작**을 눌러주세요.")
    with tab_dl:
        st.info("사이드바에서 **분석 시작**을 눌러주세요.")
    with tab_cmp:
        st.info("사이드바에서 **분석 시작**을 눌러주세요.")
    with tab_ts:
        st.info("사이드바에서 **분석 시작**을 눌러주세요.")
    st.stop()

# 데이터 분리
X_tr, X_te, y_tr, y_te, feat_cols = split_data(df, test_size)

# 스케일링
scaler = MinMaxScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

all_results = []   # 전체 모델 결과 수집

# ────────────────────────────────────────────────────────
# TAB 2: 머신러닝
# ────────────────────────────────────────────────────────
with tab_ml:
    st.subheader("🤖 머신러닝 모델 학습 & 비교")

    if not use_ml:
        st.warning("사이드바에서 '머신러닝 학습'을 체크해주세요.")
        st.stop()

    sk_models = {
        "Linear Regression":    LinearRegression(),
        "Decision Tree":        DecisionTreeRegressor(random_state=42),
        "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingRegressor(random_state=42),
        "Extra Trees":          ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "AdaBoost":             AdaBoostRegressor(random_state=42),
    }

    sk_results = []
    prog = st.progress(0, "머신러닝 모델 학습 중...")
    for i, (name, model) in enumerate(sk_models.items()):
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        r = evaluate_model(y_te, pred, name)
        sk_results.append(r)
        all_results.append(r)
        prog.progress((i+1)/len(sk_models), f"{name} 완료")
    prog.empty()

    sk_df = pd.DataFrame(sk_results).sort_values("R²", ascending=False).reset_index(drop=True)
    st.session_state["sk_df"] = sk_df
    st.session_state["sk_models"] = sk_models

    best_sk = sk_df.iloc[0]
    st.success(f"🏆 최고 모델: **{best_sk['Model']}** (R² = {best_sk['R²']:.4f})")

    st.dataframe(
        sk_df.style.background_gradient(subset=["R²"], cmap="Reds")
                   .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(sk_df, x="Model", y="R²", color="R²",
                     color_continuous_scale="Reds", title="R² 비교")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(sk_df, x="Model", y=["MAE", "RMSE"],
                     barmode="group", title="MAE / RMSE 비교",
                     color_discrete_sequence=["#EF4444", "#F97316"])
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance (최고 모델이 트리 기반인 경우)
    best_model_obj = sk_models[best_sk["Model"]]
    if hasattr(best_model_obj, "feature_importances_"):
        st.markdown(f"**{best_sk['Model']} 피처 중요도**")
        imp = pd.DataFrame({
            "변수": feat_cols,
            "중요도": best_model_obj.feature_importances_
        }).sort_values("중요도", ascending=True)
        fig = px.bar(imp, x="중요도", y="변수", orientation="h",
                     color="중요도", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────
# TAB 3: 딥러닝
# ────────────────────────────────────────────────────────
with tab_dl:
    st.subheader("🧠 딥러닝 모델 학습 & 비교")

    if not use_dl:
        st.warning("사이드바에서 '딥러닝 학습'을 체크해주세요.")
        st.stop()

    n_feat = X_tr_scaled.shape[1]
    es = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)

    # CNN / LSTM / GRU 용 3D 변환
    X_tr_3d = X_tr_scaled.reshape(X_tr_scaled.shape[0], X_tr_scaled.shape[1], 1)
    X_te_3d = X_te_scaled.reshape(X_te_scaled.shape[0], X_te_scaled.shape[1], 1)

    # LSTM/GRU 용 (timestep=1, features=n_feat)
    X_tr_seq = X_tr_scaled.reshape(X_tr_scaled.shape[0], 1, X_tr_scaled.shape[1])
    X_te_seq = X_te_scaled.reshape(X_te_scaled.shape[0], 1, X_te_scaled.shape[1])

    dl_models_def = {
        "DNN": lambda: Sequential([
            Dense(64, activation="relu", input_shape=(n_feat,)),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1)
        ]),
        "Wide & Deep": None,   # 별도 빌드
        "1D-CNN": None,
        "LSTM": None,
        "GRU": None,
    }

    def build_wide_deep(n):
        wi = Input(shape=(n,))
        di = Input(shape=(n,))
        d  = Dense(128, activation="relu")(di)
        d  = Dense(64,  activation="relu")(d)
        m  = concatenate([wi, d])
        o  = Dense(1)(m)
        mdl = Model([wi, di], o)
        mdl.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return mdl

    def build_cnn(n):
        mdl = Sequential([
            Conv1D(64, 2, activation="relu", input_shape=(n, 1)),
            MaxPooling1D(),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(1)
        ])
        mdl.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return mdl

    def build_lstm(n):
        mdl = Sequential([
            LSTM(64, input_shape=(1, n)),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        mdl.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return mdl

    def build_gru(n):
        mdl = Sequential([
            GRU(64, input_shape=(1, n)),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        mdl.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return mdl

    dl_results = []
    histories  = {}

    dl_tasks = ["DNN", "Wide & Deep", "1D-CNN", "LSTM", "GRU"]
    prog_dl = st.progress(0, "딥러닝 모델 학습 중...")

    for i, name in enumerate(dl_tasks):
        prog_dl.progress(i/len(dl_tasks), f"{name} 학습 중...")

        if name == "DNN":
            mdl = dl_models_def["DNN"]()
            mdl.compile(optimizer="adam", loss="mse", metrics=["mae"])
            h = mdl.fit(X_tr_scaled, y_tr,
                        validation_split=0.2, epochs=epochs_dl,
                        batch_size=batch_size, callbacks=[es], verbose=0)
            pred = mdl.predict(X_te_scaled, verbose=0).flatten()

        elif name == "Wide & Deep":
            mdl = build_wide_deep(n_feat)
            h = mdl.fit([X_tr_scaled, X_tr_scaled], y_tr,
                        validation_split=0.2, epochs=epochs_dl,
                        batch_size=batch_size, callbacks=[es], verbose=0)
            pred = mdl.predict([X_te_scaled, X_te_scaled], verbose=0).flatten()

        elif name == "1D-CNN":
            mdl = build_cnn(n_feat)
            h = mdl.fit(X_tr_3d, y_tr,
                        validation_split=0.2, epochs=epochs_dl,
                        batch_size=batch_size, callbacks=[es], verbose=0)
            pred = mdl.predict(X_te_3d, verbose=0).flatten()

        elif name == "LSTM":
            mdl = build_lstm(n_feat)
            h = mdl.fit(X_tr_seq, y_tr,
                        validation_split=0.2, epochs=epochs_dl,
                        batch_size=batch_size, callbacks=[es], verbose=0)
            pred = mdl.predict(X_te_seq, verbose=0).flatten()

        elif name == "GRU":
            mdl = build_gru(n_feat)
            h = mdl.fit(X_tr_seq, y_tr,
                        validation_split=0.2, epochs=epochs_dl,
                        batch_size=batch_size, callbacks=[es], verbose=0)
            pred = mdl.predict(X_te_seq, verbose=0).flatten()

        r = evaluate_model(y_te, pred, name)
        dl_results.append(r)
        all_results.append(r)
        histories[name] = h.history

    prog_dl.empty()

    dl_df = pd.DataFrame(dl_results).sort_values("R²", ascending=False).reset_index(drop=True)
    st.session_state["dl_df"] = dl_df

    best_dl = dl_df.iloc[0]
    st.success(f"🏆 최고 딥러닝 모델: **{best_dl['Model']}** (R² = {best_dl['R²']:.4f})")

    st.dataframe(
        dl_df.style.background_gradient(subset=["R²"], cmap="Reds")
                   .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(dl_df, x="Model", y="R²", color="R²",
                     color_continuous_scale="Reds", title="딥러닝 R² 비교")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(dl_df, x="Model", y=["MAE", "RMSE"], barmode="group",
                     title="딥러닝 MAE / RMSE 비교",
                     color_discrete_sequence=["#EF4444", "#F97316"])
        st.plotly_chart(fig, use_container_width=True)

    # 학습 곡선
    st.markdown("**모델별 학습 손실 곡선**")
    cols_loss = st.columns(len(dl_tasks))
    for idx, name in enumerate(dl_tasks):
        h = histories[name]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=h["loss"], name="Train", line=dict(color="#EF4444")))
        fig.add_trace(go.Scatter(y=h["val_loss"], name="Val", line=dict(color="#3B82F6")))
        fig.update_layout(title=name, height=220,
                          margin=dict(t=40, b=10, l=10, r=10),
                          legend=dict(font=dict(size=9)))
        cols_loss[idx].plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────
# TAB 4: 전체 비교
# ────────────────────────────────────────────────────────
with tab_cmp:
    st.subheader("📈 ML vs DL 통합 비교")

    if "sk_df" not in st.session_state or "dl_df" not in st.session_state:
        st.info("머신러닝과 딥러닝 탭을 먼저 실행해주세요.")
        st.stop()

    sk_df_ = st.session_state["sk_df"].copy()
    dl_df_ = st.session_state["dl_df"].copy()
    sk_df_["유형"] = "머신러닝"
    dl_df_["유형"] = "딥러닝"
    all_df = pd.concat([sk_df_, dl_df_], ignore_index=True).sort_values("R²", ascending=False)

    overall_best = all_df.iloc[0]
    st.success(f"🥇 전체 최고 모델: **{overall_best['Model']}** "
               f"({overall_best['유형']}) — R² = {overall_best['R²']:.4f}")

    st.dataframe(
        all_df.style.background_gradient(subset=["R²"], cmap="RdYlGn")
                    .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
        use_container_width=True
    )

    # 통합 바 차트
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(all_df.sort_values("R²"), x="R²", y="Model",
                     color="유형", orientation="h",
                     color_discrete_map={"머신러닝": "#EF4444", "딥러닝": "#3B82F6"},
                     title="전체 모델 R² 비교")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 레이더 차트
        r2_vals = all_df["R²"].clip(lower=0).tolist()
        labels  = all_df["Model"].tolist()
        r2_vals += [r2_vals[0]]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += [angles[0]]

        fig = go.Figure(go.Scatterpolar(
            r=r2_vals, theta=labels + [labels[0]],
            fill="toself", line_color="#EF4444", fillcolor="rgba(239,68,68,0.2)"
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="R² 레이더 차트", height=420
        )
        st.plotly_chart(fig, use_container_width=True)

    # 히트맵
    st.markdown("**모델 성능 히트맵**")
    hm = all_df.set_index("Model")[["MAE", "RMSE", "R²"]]
    fig = px.imshow(hm.T, text_auto=".4f", color_continuous_scale="RdBu_r",
                    aspect="auto")
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────
# TAB 5: 시계열 (Prophet)
# ────────────────────────────────────────────────────────
with tab_ts:
    st.subheader("📅 시계열 분석 (Prophet)")

    if not use_ts:
        st.info("사이드바에서 '시계열 분석 (Prophet)'을 체크하고 다시 실행해주세요.")
        st.stop()

    try:
        from prophet import Prophet

        ts = (df.groupby("계약일")["물건금액(만원)"]
                .mean().reset_index())
        ts.columns = ["ds", "y"]

        with st.spinner("Prophet 모델 학습 중..."):
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False)
            m.fit(ts)

        periods = st.slider("예측 기간 (일)", 30, 365, 180, 30)
        future   = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)

        # 예측 시각화
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["ds"], y=ts["y"],
                                 name="실제", mode="markers",
                                 marker=dict(color="#EF4444", size=4)))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                                 name="예측", line=dict(color="#3B82F6")))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(59,130,246,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="신뢰구간"
        ))
        fig.update_layout(title="물건금액(만원) 시계열 예측", height=450,
                          xaxis_title="날짜", yaxis_title="평균 물건금액(만원)")
        st.plotly_chart(fig, use_container_width=True)

        # 성분 분해
        st.markdown("**트렌드 & 계절성 분해**")
        col1, col2 = st.columns(2)
        with col1:
            fig_t = px.line(forecast, x="ds", y="trend",
                            title="트렌드", color_discrete_sequence=["#EF4444"])
            st.plotly_chart(fig_t, use_container_width=True)
        with col2:
            fig_y = px.line(forecast, x="ds", y="yearly",
                            title="연간 계절성", color_discrete_sequence=["#F97316"])
            st.plotly_chart(fig_y, use_container_width=True)

        with st.expander("예측 결과 데이터"):
            st.dataframe(
                forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(50),
                use_container_width=True
            )

    except ImportError:
        st.error("Prophet이 설치되어 있지 않습니다. `pip install prophet`을 실행해주세요.")
    except Exception as e:
        st.error(f"Prophet 오류: {e}")
