import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="주식시세 대시보드", layout="wide")

BASE_URL = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService"

st.title("📈 금융위원회 주식시세 대시보드")

service_key = st.sidebar.text_input(
    "yMi5jnM5F6Rg4YPiN9VDQDRgmq0qU85YMw0umYUOA4XNn4NQv8inxYcqcNICvib6YlFyOZGC8WTwiojKHZ/Dng==",
        type="password"
)


endpoint_map = {
    "주식시세": "getStockPriceInfo",
    "수익증권시세": "getSecuritiesPriceInfo",
    "신주인수권증권시세": "getPreemptiveRightSecuritiesPriceInfo",
    "신주인수권증서시세": "getPreemptiveRightCertificatePriceInfo"
}

api_name = st.sidebar.selectbox(
    "조회 종류",
    list(endpoint_map.keys())
)

stock_name = st.sidebar.text_input("종목명", "삼성전자")
base_date = st.sidebar.text_input("기준일(YYYYMMDD)", "")
rows = st.sidebar.number_input("조회건수", 10, 1000, 100)

search_btn = st.sidebar.button("조회")

if search_btn:

    if not service_key:
        st.error("서비스키를 입력하세요.")
        st.stop()

    params = {
        "serviceKey": service_key,
        "resultType": "json",
        "numOfRows": rows,
        "pageNo": 1
    }

    if stock_name:
        params["likeItmsNm"] = stock_name

    if base_date:
        params["basDt"] = base_date

    url = f"{BASE_URL}/{endpoint_map[api_name]}"

    try:
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()

        data = res.json()

        items = data["response"]["body"]["items"]["item"]

        df = pd.DataFrame(items)

        st.success(f"{len(df)}건 조회 완료")

        st.dataframe(df, use_container_width=True)

        if "itmsNm" in df.columns:
            st.subheader("종목별 현황")

            display_cols = [
                c for c in [
                    "itmsNm",
                    "clpr",
                    "vs",
                    "fltRt",
                    "mkp",
                    "hipr",
                    "lopr",
                    "trqu"
                ]
                if c in df.columns
            ]

            st.dataframe(
                df[display_cols],
                use_container_width=True
            )

        if "clpr" in df.columns and "itmsNm" in df.columns:

            chart_df = df.copy()

            chart_df["clpr"] = pd.to_numeric(
                chart_df["clpr"],
                errors="coerce"
            )

            chart_df = chart_df.dropna(subset=["clpr"])

            if len(chart_df) > 0:
                st.subheader("종가 비교")
                st.bar_chart(
                    chart_df.set_index("itmsNm")["clpr"]
                )

        csv = df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "CSV 다운로드",
            csv,
            file_name="stock_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.caption("공공데이터포털 금융위원회 주식시세 OpenAPI")
