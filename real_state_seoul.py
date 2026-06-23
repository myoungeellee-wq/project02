import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="서울시 부동산 실거래가 분석",
    layout="wide"
)

st.title("🏢 AI 기반 부동산 실거래가 예측")

uploaded_file = st.file_uploader(
    "CSV 업로드",
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

    # ---------------------------
    # 컬럼 제거
    # ---------------------------

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

    drop_cols = [
        c for c in drop_cols
        if c in df.columns
    ]

    df = df.drop(columns=drop_cols)

    st.subheader("원본 데이터")

    st.write(df.shape)
    st.dataframe(df.head())

    # ---------------------------
    # 결측치
    # ---------------------------

    st.subheader("결측치 현황")

    missing_df = pd.DataFrame({
        "결측치수": df.isnull().sum(),
        "결측비율(%)":
            round(df.isnull().mean()*100,2)
    })

    st.dataframe(missing_df)

    # ---------------------------
    # 중복
    # ---------------------------

    st.subheader("중복 데이터")

    st.metric(
        "중복 건수",
        int(df.duplicated().sum())
    )

    # ---------------------------
    # 컬럼 선택
    # ---------------------------

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

    use_cols = [
        c for c in use_cols
        if c in df.columns
    ]

    df = df[use_cols]

    df = df.dropna()
    df = df.drop_duplicates()

    # ---------------------------
    # 날짜
    # ---------------------------

    df["계약일"] = pd.to_datetime(
        df["계약일"],
        errors="coerce"
    )

    df["연도"] = df["계약일"].dt.year
    df["월"] = df["계약일"].dt.month
    df["분기"] = df["계약일"].dt.quarter

    # ---------------------------
    # 히스토그램
    # ---------------------------

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

    # ---------------------------
    # 상관관계
    # ---------------------------

    st.subheader("상관관계 Heatmap")

    corr = df.select_dtypes(
        include=np.number
    ).corr()

    fig, ax = plt.subplots(
        figsize=(10,8)
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="RdBu_r",
        center=0,
        ax=ax
    )

    st.pyplot(fig)

    # ---------------------------
    # 인코딩
    # ---------------------------

    cat_cols = [
        "자치구명",
        "법정동명",
        "건물용도",
        "건물명",
        "신고구분"
    ]

    for col in cat_cols:

        if col in df.columns:

            le = LabelEncoder()

            df[col] = le.fit_transform(
                df[col].astype(str)
            )

    # ---------------------------
    # 모델링
    # ---------------------------

    target = "물건금액(만원)"

    X = df.drop(
        columns=[
            target,
            "계약일"
        ]
    )

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    st.subheader("모델 성능")

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{mean_absolute_error(y_test,y_pred):,.0f}"
    )

    c2.metric(
        "RMSE",
        f"{np.sqrt(mean_squared_error(y_test,y_pred)):,.0f}"
    )

    c3.metric(
        "R²",
        f"{r2_score(y_test,y_pred):.4f}"
    )

    # ---------------------------
    # 변수 중요도
    # ---------------------------

    st.subheader("변수 중요도")

    importance_df = pd.DataFrame({
        "변수":X.columns,
        "중요도":model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "중요도",
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

    st.dataframe(importance_df)

    # ---------------------------
    # 자치구 × 분기 Heatmap
    # ---------------------------

    st.subheader(
        "자치구 × 분기 평균 거래금액"
    )

    heatmap_df = df.pivot_table(
        index="자치구명",
        columns="분기",
        values="물건금액(만원)",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(
        figsize=(12,8)
    )

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        ax=ax
    )

    st.pyplot(fig)