import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# ---------------------------------------------------
# 한글
# ---------------------------------------------------
font_path = "./font/NanumGothic.ttf"

font_prop = fm.FontProperties(
    fname=font_path
)

plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------
# 제목
# ---------------------------------------------------

st.set_page_config(
    page_title="서울시 부동산 실거래가 분석",
    layout="wide"
)

st.title("🏢 AI 기반 부동산 실거래가 예측")

# ---------------------------------------------------
# 파일 업로드
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "CSV 파일 선택",
    type=["csv"]
)

if uploaded_file:

    try:
        df = pd.read_csv(
            uploaded_file,
            encoding="cp949"
        )
    except:
        df = pd.read_csv(
            uploaded_file,
            encoding="utf-8"
        )

    st.success("데이터 로드 완료")

    # ---------------------------------------------------
    # 원본 데이터
    # ---------------------------------------------------

    st.subheader("원본 데이터")

    st.write(df.shape)
    st.dataframe(df.head())

    # ---------------------------------------------------
    # 컬럼 제거
    # ---------------------------------------------------

    drop_cols = [
        '자치구코드',
        '법정동코드',
        '지번구분',
        '본번',
        '부번',
        '권리구분',
        '취소일',
        '신고시군구명'
    ]

    df = df.drop(columns=drop_cols)

    # ---------------------------------------------------
    # 결측치
    # ---------------------------------------------------

    st.subheader("결측치")

    missing_df = pd.DataFrame({
        "결측치수": df.isnull().sum(),
        "결측비율(%)":
        round(df.isnull().mean()*100,2)
    })

    st.dataframe(missing_df)

    # ---------------------------------------------------
    # 중복
    # ---------------------------------------------------

    st.subheader("중복 데이터")

    st.metric(
        "중복 건수",
        df.duplicated().sum()
    )

    # ---------------------------------------------------
    # 사용 컬럼
    # ---------------------------------------------------

    use_cols = [
        "계약일",
        "자치구명",
        "법정동명",
        "건물면적(㎡)",
        "토지면적(㎡)",
        "층",
        "건축년도",
        "건물용도",
        "건물명",
        "신고구분",
        "물건금액(만원)"
    ]

    df = df[use_cols]

    # ---------------------------------------------------
    # 전처리
    # ---------------------------------------------------

    before_rows = len(df)

    df = df.dropna()
    df = df.drop_duplicates()

    after_rows = len(df)

    st.subheader("전처리 결과")

    st.write(
        pd.DataFrame({
            "항목":["원본","전처리후"],
            "건수":[before_rows,after_rows]
        })
    )

    # ---------------------------------------------------
    # 날짜
    # ---------------------------------------------------

    df["계약일"] = pd.to_datetime(df["계약일"])

    df["연도"] = df["계약일"].dt.year
    df["월"] = df["계약일"].dt.month

    # ---------------------------------------------------
    # 히스토그램
    # ---------------------------------------------------

    st.subheader("변수 분포")

    num_cols = df.select_dtypes(
        include=np.number
    ).columns

    selected_col = st.selectbox(
        "변수 선택",
        num_cols
    )

    fig, ax = plt.subplots()

    ax.hist(
        df[selected_col],
        bins=30,
        color="red"
    )

    ax.set_title(selected_col)

    st.pyplot(fig)

    # ---------------------------------------------------
    # 물건금액 기준 상관관계
    # ---------------------------------------------------

    st.subheader("물건금액 기준 상관관계")

    corr_target = df.select_dtypes(
        include=np.number
    ).corr()

    fig, ax = plt.subplots(
        figsize=(8,6)
    )

    sns.heatmap(
        corr_target.sort_values(
            by="물건금액(만원)"
        ),
        annot=True,
        fmt=".2f",
        cmap="Reds",
        ax=ax,
        ascending=True
    )

    st.pyplot(fig)

    # ---------------------------------------------------
    # 인코딩
    # ---------------------------------------------------

    cat_cols = [
        "자치구명",
        "법정동명",
        "건물용도",
        "신고구분",
        "건물명"
    ]

    encoders = {}

    for col in cat_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col]
        )

        encoders[col] = le

    # ---------------------------------------------------
    # Feature / Target
    # ---------------------------------------------------

    X = df[
        [
            "자치구명",
            "법정동명",
            "건물면적(㎡)",
            "토지면적(㎡)",
            "층",
            "건축년도",
            "건물용도",
            "건물명",
            "신고구분",
            "월"
        ]
    ]

    y = df["물건금액(만원)"]

    # ---------------------------------------------------
    # 모델 학습
    # ---------------------------------------------------

    if st.button("모델 학습 시작"):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        # ---------------------------------------------
        # XGBoost
        # ---------------------------------------------

        xgb = XGBRegressor(
            objective="reg:squarederror",
            random_state=42
        )

        xgb.fit(
            X_train_scaled,
            y_train
        )

        pred_xgb = xgb.predict(
            X_test_scaled
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred_xgb
            )
        )

        r2 = r2_score(
            y_test,
            pred_xgb
        )

        st.subheader("XGBoost 성능")

        c1,c2 = st.columns(2)

        c1.metric(
            "RMSE",
            f"{rmse:,.0f}"
        )

        c2.metric(
            "R²",
            f"{r2:.4f}"
        )

        # ---------------------------------------------
        # RandomForest
        # ---------------------------------------------

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(
            X_train,
            y_train
        )

        pred_rf = rf.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            pred_rf
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred_rf
            )
        )

        r2 = r2_score(
            y_test,
            pred_rf
        )

        st.subheader("RandomForest 성능")

        c1,c2,c3 = st.columns(3)

        c1.metric(
            "MAE",
            f"{mae:,.0f}"
        )

        c2.metric(
            "RMSE",
            f"{rmse:,.0f}"
        )

        c3.metric(
            "R²",
            f"{r2:.4f}"
        )

        # ---------------------------------------------
        # 변수 중요도
        # ---------------------------------------------

        st.subheader(
            "변수 중요도"
        )

        importance_df = pd.DataFrame({
            "변수":X.columns,
            "중요도":rf.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="중요도",
            ascending=False
        )

        fig, ax = plt.subplots(
            figsize=(10,6)
        )

        sns.barplot(
            data=importance_df,
            x="중요도",
            y="변수",
            color="red",
            ax=ax
        )

        st.pyplot(fig)

        st.dataframe(
            importance_df
        )