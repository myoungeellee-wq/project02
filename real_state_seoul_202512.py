import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

font_path = "./font/NanumGothic.ttf"

font_prop = fm.FontProperties(
    fname=font_path
)

plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# fonts = [f.name for f in fm.fontManager.ttflist]
# print("NanumGothic" in fonts)
# st.write("NanumGothic 존재:", "NanumGothic" in fonts)
# st.write("폰트 속성 :",font_prop.get_name())
#print("font names:", if fonts = True)

# --------------------------------------------
# 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="부동산 실거래가 예측",
    layout="wide"
)

st.title("🏢 AI 기반 부동산 실거래가 예측")

# --------------------------------------------------
# 파일 업로드
# --------------------------------------------------

""" uploaded_file = st.file_uploader(
    "CSV 파일 업로드",
    type=["csv"]
)
 """
uploaded_files = st.file_uploader(
    "CSV 파일 업로드",
    type=["csv"],
    accept_multiple_files=True
)

for file in uploaded_files:
    df = pd.read_csv(file)
    st.write(file.name)
    st.dataframe(df.head())

if uploaded_files:

    try:
        df = pd.read_csv(
            uploaded_files,
            encoding="cp949"
        )

    except:
        df = pd.read_csv(
            uploaded_files,
            encoding="utf-8"
        )

    st.success("데이터 로드 완료")

    # --------------------------------------------------
    # 데이터 확인
    # --------------------------------------------------

    st.subheader("데이터 미리보기")
    st.dataframe(df.head())

    # --------------------------------------------------
    # 결측치
    # --------------------------------------------------

    st.subheader("결측치 통계")

    missing_df = pd.DataFrame({
        "결측치수": df.isnull().sum(),
        "결측비율(%)":
            round(df.isnull().mean()*100,2)
    })

    st.dataframe(missing_df)

    # --------------------------------------------------
    # 중복값
    # --------------------------------------------------

    st.subheader("중복값 통계")

    duplicate_count = df.duplicated().sum()

    st.metric(
        "중복 데이터 수",
        duplicate_count
    )

    # --------------------------------------------------
    # 전처리
    # --------------------------------------------------

    df = df.dropna()
    df = df.drop_duplicates()

    st.subheader("전처리 후 데이터")

    st.write(df.shape)

    # --------------------------------------------------
    # 날짜 처리
    # --------------------------------------------------

    if "계약일" in df.columns:

        df["계약일"] = pd.to_datetime(
            df["계약일"],
            errors="coerce"
        )

        df["연도"] = df["계약일"].dt.year
        df["월"] = df["계약일"].dt.month
        df["분기"] = df["계약일"].dt.quarter

    # --------------------------------------------------
    # Histogram
    # --------------------------------------------------

    st.subheader("주요 변수 분포")

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

    # --------------------------------------------------
    # Heatmap
    # --------------------------------------------------

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
        cmap="Reds",
        ax=ax
    )

    st.pyplot(fig)
        
    # --------------------------------------------------
    # 모델링
    # --------------------------------------------------

    target_col = st.selectbox(
        "예측 대상",
        df.columns
    )

    model_df = df.copy()

    # datetime -> 숫자 변환
    for col in model_df.columns:
        if pd.api.types.is_datetime64_any_dtype(model_df[col]):
            model_df[col] = (
                model_df[col].astype("int64")
                // 10**9
            )

    # 문자열 -> Label Encoding
    for col in model_df.columns:
        if model_df[col].dtype == "object":
            le = LabelEncoder()
            model_df[col] = le.fit_transform(
                model_df[col].astype(str)
            )

    # 숫자 컬럼만 사용
    model_df = model_df.select_dtypes(
        include=[np.number]
    )

    X = model_df.drop(
        columns=[target_col]
    )

    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )    

    # --------------------------------------------------
    # 성능평가
    # --------------------------------------------------

    st.subheader("모델 성능")
    y_pred = model.predict(
        X_test
    )
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    c1,c2,c3 = st.columns(3)

    c1.metric("MAE", f"{mae:,.2f}")
    c2.metric("RMSE", f"{rmse:,.2f}")
    c3.metric("R²", f"{r2:.4f}")

    # --------------------------------------------------
    # 실제값 vs 예측값
    # --------------------------------------------------

    st.subheader("실제값 vs 예측값")

    fig, ax = plt.subplots()

    ax.scatter(
        y_test,
        y_pred,
        color="red",
        alpha=0.5
    )

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    st.pyplot(fig)

    # --------------------------------------------------
    # 변수 중요도
    # --------------------------------------------------

    st.subheader("변수 중요도")

    importance_df = pd.DataFrame({

        "변수":
            X.columns,

        "중요도":
            model.feature_importances_
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

    # --------------------------------------------------
    # 구청별 분석
    # --------------------------------------------------

    if (
        "자치구명" in df.columns and
        "물건금액(만원)" in df.columns
    ):

        st.subheader(
            "구청별 평균 거래금액"
        )

        district_price = df.groupby(
            "자치구명"
        )["물건금액(만원)"].mean()

        fig, ax = plt.subplots(
            figsize=(12,6)
        )

        district_price.sort_values().plot(
            kind="bar",
            color="red",
            ax=ax
        )

        st.pyplot(fig)

    # --------------------------------------------------
    # 분기별 분석
    # --------------------------------------------------

    if (
        "분기" in df.columns and
        "물건금액(만원)" in df.columns
    ):

        st.subheader(
            "분기별 거래금액"
        )

        quarter_price = df.groupby(
            "분기"
        )["물건금액(만원)"].mean()

        fig, ax = plt.subplots()

        quarter_price.plot(
            marker="o",
            color="red",
            ax=ax
        )

        st.pyplot(fig)

        st.subheader(
            "구청별 / 분기별 Heatmap"
        )

        if "자치구명" in df.columns:

            pivot_df = df.pivot_table(
                index="자치구명",
                columns="분기",
                values="물건금액(만원)",
                aggfunc="mean"
            )

            fig, ax = plt.subplots(
                figsize=(12,8)
            )

            sns.heatmap(
                pivot_df,
                annot=True,
                cmap="Reds",
                ax=ax
            )

            st.pyplot(fig)