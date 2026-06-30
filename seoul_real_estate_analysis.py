
# 서울시 부동산 실거래가 분석
# 실행:
# pip install pandas matplotlib seaborn ydata-profiling openpyxl
# python seoul_real_estate_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from ydata_profiling import ProfileReport
    YDATA_AVAILABLE = True
except ImportError:
    YDATA_AVAILABLE = False

FILE_NAME = "dataset/서울시 부동산 실거래가 정보_2025.csv"

# -----------------------------
# 데이터 로드
# -----------------------------
df = pd.read_csv(FILE_NAME, encoding="cp949")

print("=" * 60)
print("데이터 크기:", df.shape)
print("=" * 60)
print(df.info())

# -----------------------------
# 전처리
# -----------------------------
df["물건금액(만원)"] = pd.to_numeric(df["물건금액(만원)"], errors="coerce")
df["건물면적(㎡)"] = pd.to_numeric(df["건물면적(㎡)"], errors="coerce")

df["계약일"] = pd.to_datetime(
    df["계약일"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

df["평수"] = df["건물면적(㎡)"] / 3.3058
df["평당가격"] = df["물건금액(만원)"] / df["평수"]

print("\n결측치 현황")
print(df.isnull().sum())

# -----------------------------
# 자치구별 평균 거래금액
# -----------------------------
district_price = (
    df.groupby("자치구명")["물건금액(만원)"]
    .mean()
    .sort_values(ascending=False)
)

print("\n자치구별 평균 거래금액 TOP10")
print(district_price.head(10))

plt.figure(figsize=(12, 6))
district_price.plot(kind="bar")
plt.title("자치구별 평균 거래금액")
plt.tight_layout()
plt.savefig("01_자치구별_평균거래금액.png", encoding="utf-8-sig")
plt.close()

# -----------------------------
# 자치구별 거래건수
# -----------------------------
district_count = df["자치구명"].value_counts()

plt.figure(figsize=(12, 6))
district_count.plot(kind="bar", encoding="utf-8-sig")
plt.title("자치구별 거래건수")
plt.tight_layout()
plt.savefig("02_자치구별_거래건수.png", encoding="utf-8-sig")
plt.close()

# -----------------------------
# 자치구별 평당가격
# -----------------------------
district_pyung = (
    df.groupby("자치구명")["평당가격"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))
district_pyung.plot(kind="bar")
plt.title("자치구별 평균 평당가격")
plt.tight_layout()
plt.savefig("03_자치구별_평당가격.png", encoding="utf-8-sig")
plt.close()

# -----------------------------
# 월별 거래량
# -----------------------------
df["계약월"] = df["계약일"].dt.to_period("M")

monthly_count = (
    df.groupby("계약월")
    .size()
)

plt.figure(figsize=(12, 5))
monthly_count.plot(marker="o")
plt.title("월별 거래량")
plt.tight_layout()
plt.savefig("04_월별_거래량.png", encoding="utf-8-sig")
plt.close()

# -----------------------------
# 아파트 TOP20
# -----------------------------
apt_top20 = (
    df.groupby("건물명")["물건금액(만원)"]
    .mean()
    .sort_values(ascending=False)
    .head(20)
)

apt_top20.to_csv("05_아파트_TOP20.csv", encoding="utf-8-sig")

# -----------------------------
# YData Profiling
# -----------------------------
if YDATA_AVAILABLE:
    profile = ProfileReport(
        df,
        title="서울시 부동산 실거래가 분석",
        explorative=True
    )
    profile.to_file("서울시_부동산_분석리포트.html")
    print("YData Profiling 리포트 생성 완료")
else:
    print("ydata-profiling 미설치 상태")

print("\n분석 완료")
print("생성 파일:")
print("- 01_자치구별_평균거래금액.png")
print("- 02_자치구별_거래건수.png")
print("- 03_자치구별_평당가격.png")
print("- 04_월별_거래량.png")
print("- 05_아파트_TOP20.csv")
print("- 서울시_부동산_분석리포트.html")
