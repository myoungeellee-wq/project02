"""
금융위원회 주식시세 OpenAPI 통합 예제
사용법:
    pip install requests pandas
    python stock_api.py
"""

import requests
import pandas as pd


#SERVICE_KEY = "yMi5jnM5F6Rg4YPiN9VDQDRgmq0qU85YMw0umYUOA4XNn4NQv8inxYcqcNICvib6YlFyOZGC8WTwiojKHZ%2FDng%3D%3D"
SERVICE_KEY = "yMi5jnM5F6Rg4YPiN9VDQDRgmq0qU85YMw0umYUOA4XNn4NQv8inxYcqcNICvib6YlFyOZGC8WTwiojKHZ/Dng=="

BASE_URL = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService"


class StockAPI:

    def __init__(self, service_key):
        self.service_key = service_key

    def request(self, endpoint, **kwargs):

        params = {
            "serviceKey": self.service_key,
            "resultType": "json",
            **kwargs
        }

        url = f"{BASE_URL}/{endpoint}"

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        try:
            items = data["response"]["body"]["items"]["item"]
            return pd.DataFrame(items)
        except Exception:
            print(data)
            return pd.DataFrame()

    def stock_price(self, **kwargs):
        return self.request("getStockPriceInfo", **kwargs)

    def securities_price(self, **kwargs):
        return self.request("getSecuritiesPriceInfo", **kwargs)

    def preemptive_right_security(self, **kwargs):
        return self.request(
            "getPreemptiveRightSecuritiesPriceInfo",
            **kwargs
        )

    def preemptive_right_certificate(self, **kwargs):
        return self.request(
            "getPreemptiveRightCertificatePriceInfo",
            **kwargs
        )


if __name__ == "__main__":

    api = StockAPI(SERVICE_KEY)

    print("=" * 60)
    print("1. 삼성전자 조회")
    print("=" * 60)

    samsung = api.stock_price(
        likeItmsNm="삼성전자",
        numOfRows=100,
        pageNo=1
    )

    print(samsung.head())

    samsung.to_csv(
        "samsung_stock.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 60)
    print("2. 특정일 전체 종목 조회")
    print("=" * 60)

    stocks = api.stock_price(
        basDt="20250616",
        numOfRows=500,
        pageNo=1
    )

    print(stocks.head())

    stocks.to_csv(
        "daily_stock_price.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 60)
    print("3. 수익증권 조회")
    print("=" * 60)

    fund_df = api.securities_price(
        numOfRows=100,
        pageNo=1
    )

    print(fund_df.head())

    fund_df.to_csv(
        "fund_price.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("완료")
