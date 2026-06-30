# -*- coding: utf-8 -*-
"""
서울시 부동산 실거래가 분석·예측 대시보드 (Streamlit)
----------------------------------------------------
친구가 만든 노트북(project03_1.ipynb)의 전처리/모델 파이프라인을 그대로 옮겨
실시간 시세 예측 + 모델 비교 + EDA + 변수 영향도를 보여주는 웹 앱 
실행: pip install streamlit pandas numpy scikit-learn tensorflow matplotlib
    streamlit run app.py

데이터:   
    기본 경로 = ./dataset/서울시 부동산 실거래가 정보_202606.csv  (cp949)
    경로가 없으면 사이드바에서 CSV 업로드 가능.

참고:
    노트북 원본은 X 에 '물건금액(만원)'·'면적당금액'(타깃)이 함께 들어가 데이터 누수가 있었음.
    이 앱은 입력 가능한 실제 변수만 사용하도록 누수 컬럼을 제외하고 학습함.
    (전처리·인코딩·모델 구조는 노트북과 동일)
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TF 경고 최소화
import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------------
# 상수 (노트북과 동일)
# ----------------------------------------------------------------------------
DEFAULT_CSV = "dataset/서울시 부동산 실거래가 정보_202606.csv"

USE_COLS = ["계약일", "자치구명", "법정동명", "건물면적(㎡)", "토지면적(㎡)",
            "층", "건축년도", "건물용도", "건물명", "신고구분", "물건금액(만원)"]
CAT_COLS = ["자치구명", "법정동명", "건물용도", "건물명", "신고구분"]   # LabelEncoding 대상
EMBED_CATS = ["자치구명", "법정동명", "건물용도"]                       # 임베딩 모델 입력
TARGET = "면적당금액"
# 학습 입력에서 제외 (누수/식별자)
LEAK_COLS = ["물건금액(만원)", "면적당금액"]
DROP_COLS = ["계약일"]

GREEN = "#2E8B57"
DGREEN = "#14613F"
GOLD = "#D99A2B"

# ----------------------------------------------------------------------------
# 순수 로직 함수 (Streamlit 비의존 → 단독 테스트 가능)
# ----------------------------------------------------------------------------
def read_dataframe(path_or_buffer):
    """CSV 로드 (cp949 우선, 실패 시 utf-8) + 필요한 컬럼만 선택."""
    try:
        df = pd.read_csv(path_or_buffer, encoding="cp949")
    except (UnicodeDecodeError, Exception):
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        df = pd.read_csv(path_or_buffer, encoding="utf-8")
    cols = [c for c in USE_COLS if c in df.columns]
    missing = [c for c in USE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")
    return df[cols].copy()


def preprocess(df_in):
    """노트북과 동일한 전처리 + 인코딩. 원본/인코딩본을 함께 반환."""
    df = df_in.copy()

    # 결측·중복 제거
    df = df.dropna().drop_duplicates()
    # 면적 0 제거
    df = df[df["건물면적(㎡)"] > 0]
    # 물건금액 IQR 이상치 제거
    q1, q3 = df["물건금액(만원)"].quantile(0.25), df["물건금액(만원)"].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df[(df["물건금액(만원)"] >= lo) & (df["물건금액(만원)"] <= hi)]

    # 파생 변수
    df["면적당금액"] = df["물건금액(만원)"] / df["건물면적(㎡)"]
    df["계약일"] = pd.to_datetime(df["계약일"], errors="coerce")
    df = df.dropna(subset=["계약일"])
    df["연도"] = df["계약일"].dt.year
    df["계약년월"] = df["계약일"].dt.year + df["계약일"].dt.month  # 노트북 동일 정의

    df = df.reset_index(drop=True)

    # 원본(문자열 범주 유지) 보관 → UI 옵션/EDA 용
    df_raw = df.copy()

    # 범주형 LabelEncoding
    encoders = {}
    df_enc = df.copy()
    for c in CAT_COLS:
        le = LabelEncoder()
        df_enc[c] = le.fit_transform(df_enc[c].astype(str))
        encoders[c] = le

    feature_cols = [c for c in df_enc.columns
                    if c not in LEAK_COLS + DROP_COLS]
    num_cols = [c for c in feature_cols if c not in EMBED_CATS]

    return {
        "df_raw": df_raw,
        "df_enc": df_enc,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "target": TARGET,
    }


def split_xy(bundle, test_size=0.2, random_state=42):
    df_enc = bundle["df_enc"]
    X = df_enc[bundle["feature_cols"]].copy()
    y = df_enc[bundle["target"]].copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _metrics(y_true, pred):
    return (mean_absolute_error(y_true, pred),
            float(np.sqrt(mean_squared_error(y_true, pred))),
            r2_score(y_true, pred))


def train_sklearn(X_tr, y_tr, X_te, y_te):
    """노트북의 6개 sklearn 회귀 모델 학습 + 성능."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(random_state=42),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        mae, rmse, r2 = _metrics(y_te, m.predict(X_te))
        rows.append([name, mae, rmse, r2])
        fitted[name] = m
    res = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"]).sort_values("R2", ascending=False)
    return fitted, res.reset_index(drop=True)


def train_dl(bundle, X_tr, y_tr, X_te, y_te, epochs=30, batch_size=64, sample_cap=20000):
    """노트북의 Keras 모델들(DNN·Wide&Deep·1D-CNN·LSTM·GRU·Embedding) 학습 + 성능.
    속도를 위해 epochs/표본 수 축소 가능. scaler·구성요소를 함께 반환."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (Input, Dense, Dropout, concatenate,
                                          Embedding, Flatten, Conv1D, MaxPooling1D, LSTM, GRU)
    from tensorflow.keras.callbacks import EarlyStopping
    tf.random.set_seed(42)

    feature_cols = bundle["feature_cols"]
    num_cols = bundle["num_cols"]
    df_enc = bundle["df_enc"]

    # 표본 축소 (대용량일 때 학습 시간 단축)
    if sample_cap and len(X_tr) > sample_cap:
        idx = np.random.RandomState(42).choice(len(X_tr), sample_cap, replace=False)
        X_tr = X_tr.iloc[idx]; y_tr = y_tr.iloc[idx]

    scaler = MinMaxScaler()
    Xtr_s = scaler.fit_transform(X_tr)
    Xte_s = scaler.transform(X_te)
    f = Xtr_s.shape[1]
    es = EarlyStopping(patience=5, restore_best_weights=True)
    fit_kw = dict(epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0, callbacks=[es])

    models, hist = {}, {}

    # DNN
    dnn = Sequential([Input((f,)), Dense(64, activation="relu"),
                      Dense(32, activation="relu"), Dense(1)])
    dnn.compile(optimizer="adam", loss="mse", metrics=["mae"])
    hist["DNN"] = dnn.fit(Xtr_s, y_tr, **fit_kw)
    models["DNN"] = dnn

    # Wide & Deep
    wide_in = Input((f,)); deep_in = Input((f,))
    deep = Dense(128, activation="relu")(deep_in)
    deep = Dense(64, activation="relu")(deep)
    out = Dense(1)(concatenate([wide_in, deep]))
    wd = Model([wide_in, deep_in], out)
    wd.compile(optimizer="adam", loss="mse", metrics=["mae"])
    hist["Wide&Deep"] = wd.fit([Xtr_s, Xtr_s], y_tr, **fit_kw)
    models["Wide&Deep"] = wd

    # 1D-CNN
    Xtr_cnn = Xtr_s.reshape(Xtr_s.shape[0], f, 1)
    cnn = Sequential([Input((f, 1)), Conv1D(64, 2, activation="relu"),
                      MaxPooling1D(), Flatten(),
                      Dense(64, activation="relu"), Dense(1)])
    cnn.compile(optimizer="adam", loss="mse", metrics=["mae"])
    hist["1D-CNN"] = cnn.fit(Xtr_cnn, y_tr, **fit_kw)
    models["1D-CNN"] = cnn

    # LSTM / GRU  (timestep=1)
    Xtr_seq = Xtr_s.reshape(Xtr_s.shape[0], 1, f)
    lstm = Sequential([Input((1, f)), LSTM(64),
                       Dense(32, activation="relu"), Dense(1)])
    lstm.compile(optimizer="adam", loss="mse", metrics=["mae"])
    hist["LSTM"] = lstm.fit(Xtr_seq, y_tr, **fit_kw)
    models["LSTM"] = lstm

    gru = Sequential([Input((1, f)), GRU(64),
                      Dense(32, activation="relu"), Dense(1)])
    gru.compile(optimizer="adam", loss="mse", metrics=["mae"])
    hist["GRU"] = gru.fit(Xtr_seq, y_tr, **fit_kw)
    models["GRU"] = gru

    # Embedding + DNN (범주형 임베딩, 입력은 인코딩된 원값 = 비스케일)
    gu_in, dong_in, use_in = Input((1,)), Input((1,)), Input((1,))
    num_in = Input((len(num_cols),))
    gu_e = Flatten()(Embedding(df_enc["자치구명"].nunique() + 1, 8)(gu_in))
    dong_e = Flatten()(Embedding(df_enc["법정동명"].nunique() + 1, 8)(dong_in))
    use_e = Flatten()(Embedding(df_enc["건물용도"].nunique() + 1, 4)(use_in))
    deep = concatenate([gu_e, dong_e, use_e, num_in])
    deep = Dense(64, activation="relu")(deep)
    deep = Dense(32, activation="relu")(deep)
    emb = Model([gu_in, dong_in, use_in, num_in], Dense(1)(deep))
    emb.compile(optimizer="adam", loss="mse", metrics=["mae"])

    def emb_inputs(Xdf):
        return [Xdf["자치구명"].values, Xdf["법정동명"].values,
                Xdf["건물용도"].values, Xdf[num_cols].values]
    hist["Embedding"] = emb.fit(emb_inputs(X_tr), y_tr, **fit_kw)
    models["Embedding"] = emb

    # 성능 평가
    rows = []
    rows.append(["DNN", *_metrics(y_te, dnn.predict(Xte_s, verbose=0).flatten())])
    rows.append(["Wide&Deep", *_metrics(y_te, wd.predict([Xte_s, Xte_s], verbose=0).flatten())])
    rows.append(["1D-CNN", *_metrics(y_te, cnn.predict(Xte_s.reshape(Xte_s.shape[0], f, 1), verbose=0).flatten())])
    rows.append(["LSTM", *_metrics(y_te, lstm.predict(Xte_s.reshape(Xte_s.shape[0], 1, f), verbose=0).flatten())])
    rows.append(["GRU", *_metrics(y_te, gru.predict(Xte_s.reshape(Xte_s.shape[0], 1, f), verbose=0).flatten())])
    rows.append(["Embedding", *_metrics(y_te, emb.predict(emb_inputs(X_te), verbose=0).flatten())])
    res = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"]).sort_values("R2", ascending=False)

    return {"models": models, "scaler": scaler, "num_cols": num_cols,
            "n_features": f, "history": hist}, res.reset_index(drop=True)


def predict_one(model_key, sk_models, dl_artifacts, row_df, num_cols):
    """단일 입력행(인코딩 완료 DataFrame, feature 순서 일치)에 대한 예측."""
    if sk_models and model_key in sk_models:
        return float(sk_models[model_key].predict(row_df)[0])

    art = dl_artifacts
    f = art["n_features"]
    if model_key == "Embedding":
        x = [row_df["자치구명"].values, row_df["법정동명"].values,
             row_df["건물용도"].values, row_df[num_cols].values]
        return float(art["models"]["Embedding"].predict(x, verbose=0).flatten()[0])

    xs = art["scaler"].transform(row_df)   # DataFrame → 컬럼명 유지
    m = art["models"][model_key]
    if model_key == "Wide&Deep":
        return float(m.predict([xs, xs], verbose=0).flatten()[0])
    if model_key == "1D-CNN":
        return float(m.predict(xs.reshape(1, f, 1), verbose=0).flatten()[0])
    if model_key in ("LSTM", "GRU"):
        return float(m.predict(xs.reshape(1, 1, f), verbose=0).flatten()[0])
    # DNN
    return float(m.predict(xs, verbose=0).flatten()[0])


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def _inject_css():
    st.markdown(f"""
    <style>
      .stApp {{ background: linear-gradient(180deg,#f1f9f4 0%, #ffffff 38%); }}
      h1,h2,h3 {{ color:{DGREEN}; }}
      .stTabs [aria-selected="true"] {{ color:{DGREEN}; }}
      div[data-testid="stMetricValue"] {{ color:{DGREEN}; }}
      .stButton>button {{ background:{GREEN}; color:#fff; border:0; border-radius:8px; font-weight:600; }}
      .stButton>button:hover {{ background:{DGREEN}; color:#fff; }}
      .hero {{ background:linear-gradient(120deg,{DGREEN},{GREEN}); color:#fff;
               padding:18px 22px; border-radius:14px; margin-bottom:8px; }}
      .hero h1 {{ color:#fff; margin:0; font-size:1.5rem; }}
      .hero p {{ color:#e6f3ec; margin:4px 0 0; font-size:0.92rem; }}
    </style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _cached_bundle(raw_bytes: bytes):
    df = read_dataframe(io.BytesIO(raw_bytes))
    return preprocess(df)


@st.cache_resource(show_spinner=False)
def _cached_sklearn(data_key: str, test_size: float):
    bundle = st.session_state["_bundle"]
    X_tr, X_te, y_tr, y_te = split_xy(bundle, test_size=test_size)
    st.session_state["_splits"] = (X_tr, X_te, y_tr, y_te)
    return train_sklearn(X_tr, y_tr, X_te, y_te)


@st.cache_resource(show_spinner=False)
def _cached_dl(data_key: str, epochs: int, sample_cap: int, test_size: float):
    bundle = st.session_state["_bundle"]
    X_tr, X_te, y_tr, y_te = split_xy(bundle, test_size=test_size)
    return train_dl(bundle, X_tr, y_tr, X_te, y_te, epochs=epochs, sample_cap=sample_cap)


def _load_raw_bytes(sidebar):
    """사이드바에서 데이터 소스 결정 → 원본 bytes 반환 (+키)."""
    sidebar.subheader("📁 데이터")
    up = sidebar.file_uploader("CSV 업로드 (선택)", type=["csv"])
    path = sidebar.text_input("또는 파일 경로", value=DEFAULT_CSV)
    if up is not None:
        b = up.getvalue()
        return b, "upload:" + hashlib.md5(b).hexdigest()[:10]
    if path and os.path.exists(path):
        with open(path, "rb") as fp:
            b = fp.read()
        return b, "path:" + path + ":" + str(os.path.getsize(path))
    return None, None


def main():
    st.set_page_config(page_title="서울 부동산 실거래가 예측", page_icon="🏠", layout="wide")
    _inject_css()
    st.markdown('<div class="hero"><h1>🏠 서울시 부동산 실거래가 분석·예측 대시보드</h1>'
                '<p>TensorFlow/Keras 다중 딥러닝 + 머신러닝 모델 · 면적당 시세 예측 · EDA · 변수 영향도</p></div>',
                unsafe_allow_html=True)

    sb = st.sidebar
    raw_bytes, data_key = _load_raw_bytes(sb)
    if raw_bytes is None:
        st.info("좌측 사이드바에서 CSV를 업로드하거나 올바른 파일 경로를 입력하세요.\n\n"
                f"기본 경로: `{DEFAULT_CSV}`")
        st.stop()

    # 전처리
    try:
        with st.spinner("데이터 전처리 중..."):
            bundle = _cached_bundle(raw_bytes)
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류: {e}")
        st.stop()
    st.session_state["_bundle"] = bundle

    df_raw, df_enc = bundle["df_raw"], bundle["df_enc"]
    encoders, feature_cols, num_cols = bundle["encoders"], bundle["feature_cols"], bundle["num_cols"]

    sb.subheader("⚙️ 학습 옵션")
    test_size = sb.slider("테스트 비율", 0.1, 0.4, 0.2, 0.05)
    sb.caption("머신러닝(sklearn) 모델은 자동 학습됩니다.")
    use_dl = sb.checkbox("딥러닝(Keras) 모델도 학습", value=False,
                         help="DNN·Wide&Deep·1D-CNN·LSTM·GRU·Embedding. 시간이 다소 걸립니다.")
    epochs = sb.slider("DL epochs", 10, 100, 30, 5, disabled=not use_dl)
    sample_cap = sb.select_slider("DL 학습 표본 상한", [5000, 10000, 20000, 50000, 0],
                                  value=20000, disabled=not use_dl,
                                  help="0 = 전체 사용")

    sb.divider()
    sb.metric("전처리 후 데이터", f"{len(df_enc):,} 건")
    sb.metric("자치구 수", f"{df_raw['자치구명'].nunique()} 개")
    sb.caption(f"입력 변수 {len(feature_cols)}개 · 타깃: 면적당금액(만원/㎡)")

    # 머신러닝 학습
    with st.spinner("머신러닝 모델 학습 중..."):
        sk_models, sk_res = _cached_sklearn(data_key, test_size)
    X_tr, X_te, y_tr, y_te = st.session_state["_splits"]

    # 딥러닝 학습(옵션)
    dl_art, dl_res = None, None
    if use_dl:
        with st.spinner(f"딥러닝 6종 학습 중... (epochs={epochs})"):
            dl_art, dl_res = _cached_dl(data_key, epochs, sample_cap, test_size)

    # 통합 성능표
    all_res = sk_res.copy()
    if dl_res is not None:
        all_res = pd.concat([sk_res, dl_res], ignore_index=True).sort_values("R2", ascending=False).reset_index(drop=True)
    best_name = all_res.iloc[0]["Model"]

    tab1, tab2, tab3, tab4 = st.tabs(["🏠 시세 예측", "📊 모델 성능 비교", "🔍 데이터 탐색", "🧠 변수 영향도"])

    # ---------------- Tab 1: 시세 예측 ----------------
    with tab1:
        st.subheader("조건을 입력하면 예측 시세(면적당금액)를 계산합니다")
        c1, c2, c3 = st.columns(3)

        gu_opts = sorted(df_raw["자치구명"].unique().tolist())
        gu = c1.selectbox("자치구", gu_opts)
        dong_opts = sorted(df_raw.loc[df_raw["자치구명"] == gu, "법정동명"].unique().tolist())
        dong = c1.selectbox("법정동", dong_opts)
        use_opts = sorted(df_raw["건물용도"].unique().tolist())
        usage = c1.selectbox("건물용도", use_opts)

        area = c2.number_input("건물면적(㎡)", min_value=1.0,
                               value=float(round(df_raw["건물면적(㎡)"].median(), 1)), step=1.0)
        land = c2.number_input("토지면적(㎡)", min_value=0.0,
                               value=float(round(df_raw["토지면적(㎡)"].median(), 1)), step=1.0)
        floor = c2.number_input("층", value=int(df_raw["층"].median()), step=1)

        yr_built = c3.number_input("건축년도", min_value=1950, max_value=2030,
                                   value=int(df_raw["건축년도"].median()), step=1)
        report_opts = sorted(df_raw["신고구분"].unique().tolist())
        report = c3.selectbox("신고구분", report_opts)
        year = c3.number_input("거래 연도", min_value=2000, max_value=2035,
                               value=int(df_raw["연도"].median()), step=1)
        month = c3.number_input("거래 월", min_value=1, max_value=12, value=6, step=1)

        model_choices = list(sk_models.keys())
        if dl_art is not None:
            model_choices += list(dl_art["models"].keys())
        pick = st.selectbox("예측 모델", model_choices,
                            index=model_choices.index(best_name) if best_name in model_choices else 0)

        if st.button("💡 예측 실행", type="primary"):
            # 입력행 구성 (인코딩)
            def enc(col, val):
                le = encoders[col]
                # 보지 못한 값 방어
                val = str(val)
                if val in le.classes_:
                    return int(le.transform([val])[0])
                return int(pd.Series(le.transform(le.classes_)).mode().iloc[0])

            # 건물명: 선택한 동에서 가장 흔한 건물 → 대표값으로 자동 지정
            dong_mask = df_raw["법정동명"] == dong
            if dong_mask.any():
                rep_bldg = df_raw.loc[dong_mask, "건물명"].mode().iloc[0]
            else:
                rep_bldg = df_raw["건물명"].mode().iloc[0]

            row = {
                "자치구명": enc("자치구명", gu),
                "법정동명": enc("법정동명", dong),
                "건물면적(㎡)": area,
                "토지면적(㎡)": land,
                "층": floor,
                "건축년도": yr_built,
                "건물용도": enc("건물용도", usage),
                "건물명": enc("건물명", rep_bldg),
                "신고구분": enc("신고구분", report),
                "연도": year,
                "계약년월": year + month,
            }
            row_df = pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)

            try:
                per_area = predict_one(pick, sk_models, dl_art, row_df, num_cols)
            except Exception as e:
                st.error(f"예측 중 오류: {e}")
                per_area = None

            if per_area is not None:
                per_area = max(per_area, 0.0)
                total = per_area * area
                m1, m2, m3 = st.columns(3)
                m1.metric("예상 면적당 시세", f"{per_area:,.0f} 만원/㎡")
                m2.metric("예상 총 거래가", f"{total:,.0f} 만원")
                m3.metric("적용 모델", pick)

                # 동일 법정동 실제 분포와 비교
                comp = df_raw.loc[df_raw["법정동명"] == dong, "면적당금액"]
                if len(comp) >= 5:
                    lo, med, hi = comp.quantile([0.1, 0.5, 0.9])
                    st.caption(f"📍 **{gu} {dong}** 실거래 면적당금액 분포 — "
                               f"하위10% {lo:,.0f} · 중앙값 {med:,.0f} · 상위10% {hi:,.0f} (만원/㎡)")
                    pos = "평균 수준" if lo <= per_area <= hi else ("높은 편" if per_area > hi else "낮은 편")
                    st.progress(min(max((per_area - lo) / (hi - lo + 1e-9), 0), 1.0),
                                text=f"이 동네 분포 대비: {pos}")
        st.caption("※ 예측 타깃은 ‘면적당금액(만원/㎡)’이며, 총액 = 면적당금액 × 건물면적으로 환산합니다.")

    # ---------------- Tab 2: 모델 성능 비교 ----------------
    with tab2:
        st.subheader("모델별 성능 (테스트셋)")
        show = all_res.copy()
        show[["MAE", "RMSE", "R2"]] = show[["MAE", "RMSE", "R2"]].round(3)
        st.dataframe(
            show.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"})
                .background_gradient(subset=["R2"], cmap="Greens"),
            use_container_width=True, hide_index=True)

        st.markdown(f"**🏆 최적 모델: `{best_name}`**  (R² {all_res.iloc[0]['R2']:.3f})")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("R² (높을수록 우수)")
            st.bar_chart(all_res.set_index("Model")[["R2"]], color=GREEN, height=320)
        with cc2:
            st.caption("MAE · RMSE (낮을수록 우수)")
            st.bar_chart(all_res.set_index("Model")[["MAE", "RMSE"]], height=320)

        if not use_dl:
            st.info("딥러닝(Keras) 모델 결과를 추가하려면 사이드바에서 **‘딥러닝 모델도 학습’** 을 켜세요.")

        # 학습 곡선 (DL)
        if dl_art is not None:
            st.divider()
            st.caption("딥러닝 학습 곡선 (Loss)")
            mk = st.selectbox("모델 선택", list(dl_art["history"].keys()), key="hist_pick")
            h = dl_art["history"][mk].history
            hist_df = pd.DataFrame({"train_loss": h.get("loss", []),
                                    "val_loss": h.get("val_loss", [])})
            st.line_chart(hist_df, height=300)

    # ---------------- Tab 3: 데이터 탐색 ----------------
    with tab3:
        st.subheader("탐색적 데이터 분석 (EDA)")
        g1, g2 = st.columns(2)
        with g1:
            st.caption("자치구별 평균 면적당금액 (상위 15)")
            by_gu = (df_raw.groupby("자치구명")["면적당금액"].mean()
                     .sort_values(ascending=False).head(15))
            st.bar_chart(by_gu, color=GREEN, height=360)
        with g2:
            st.caption("건물면적 vs 물건금액 (표본 3,000)")
            samp = df_raw[["건물면적(㎡)", "물건금액(만원)"]].sample(
                min(3000, len(df_raw)), random_state=0)
            st.scatter_chart(samp, x="건물면적(㎡)", y="물건금액(만원)",
                             color=GREEN, height=360)

        st.caption("월별 평균 면적당금액 추이")
        ts = (df_raw.set_index("계약일")["면적당금액"]
              .resample("MS").mean().dropna())
        if len(ts) >= 2:
            st.line_chart(ts, color=GOLD, height=300)
        else:
            st.info("월별 추이를 그릴 만큼 기간이 충분하지 않습니다.")

        with st.expander("기초 통계 보기"):
            st.dataframe(df_raw[["건물면적(㎡)", "토지면적(㎡)", "층", "건축년도",
                                 "물건금액(만원)", "면적당금액"]].describe().round(1),
                         use_container_width=True)

    # ---------------- Tab 4: 변수 영향도 ----------------
    with tab4:
        st.subheader("변수 영향도 (Feature Importance)")
        tree_pref = ["Random Forest", "Extra Trees", "Gradient Boosting", "Decision Tree"]
        tree_name = next((n for n in tree_pref if n in sk_models), None)
        if tree_name:
            imp = pd.DataFrame({
                "변수": feature_cols,
                "중요도": sk_models[tree_name].feature_importances_
            }).sort_values("중요도", ascending=False)
            st.caption(f"`{tree_name}` 기준 변수 중요도")
            st.bar_chart(imp.set_index("변수"), color=GREEN, height=420)
            st.dataframe(imp.assign(중요도=imp["중요도"].round(4)),
                         use_container_width=True, hide_index=True)
            st.caption("트리 기반 모델의 분기 기여도 기준입니다. "
                       "딥러닝 모델의 SHAP 분석은 노트북에서 별도 수행할 수 있습니다.")
        else:
            st.info("트리 기반 모델이 없어 중요도를 계산할 수 없습니다.")


if os.environ.get("RE_APP_SMOKE") != "1":
    main()
