# 📈 Predict Future Sales: Time Series Forecasting

Kaggle의 [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales) 컴피티션 솔루션입니다.
러시아의 소프트웨어 회사인 1C Company의 일별 판매 데이터를 시계열 데이터로 변환하여, **다음 달(Month 34)의 상점-상품별 월간 판매량을 예측**하는 회귀 문제입니다.

## 🏆 Result
- **Private Leaderboard Score (RMSE):** `0.8570` (LightGBM Single Model)
- **Rank:** Top 7%

---

## 🚀 Key Approach & Workflow

시계열 데이터의 특성을 반영한 **Grid 데이터 구축**과 **Lag Feature 생성**에 집중하였으며, 과적합을 방지하기 위한 **Feature Selection**이 성능 향상의 핵심이었습니다.

### 1. Data Preprocessing (`04_00_PREP.ipynb`)
- **Outlier Removal:** 판매량(item_cnt_day) 및 가격(item_price)의 극단적인 이상치 제거.
- **Data Cleaning:** 중복된 Shop Name 통합, 잘못된 Shop ID 수정.
- **Grid Construction:**
  - 월(Month)별로 존재하는 모든 `Shop`과 `Item`의 조합(Cartesian Product)을 생성.
  - 판매 실적이 없는(0) 데이터까지 포함하여 모델이 '판매량 0'의 패턴을 학습하도록 유도.
- **Text Processing:** `Item Name`과 `Shop Name`에서 도시(City), 상품 분류(Category/Type) 등의 텍스트 정보를 추출하여 라벨 인코딩.

### 2. Feature Engineering
모델이 시계열 패턴을 학습할 수 있도록 다양한 파생 변수를 생성했습니다.
- **Lag Features (시차 변수):**
  - 핵심 피처. 1, 2, 3개월 전의 판매량, 평균 가격 등을 피처로 추가.
  - `Shop`, `Item`, `Category`, `City` 등 다양한 그룹별 평균 판매량의 Lag 값 생성.
- **Trend Features:**
  - 최근 가격 변동 추이(`delta_price_lag`).
- **Time Features:**
  - `Month` (계절성 반영)
  - 첫 판매 이후 경과 기간(`Item Age`), 마지막 판매 이후 경과 기간(`Last Sale`).

### 3. Modeling (`05_00_lgbm.ipynb`)
- **Model:** LightGBM (Gradient Boosting Decision Tree)
- **Objective:** Tweedie (0이 많은 데이터셋 특성 고려)
- **Validation Strategy:**
  - Train: Month 0 ~ 32
  - Validation: Month 33 (가장 최근 과거 데이터)
  - Test: Month 34
- **Hyperparameter Tuning:** `Optuna`를 사용하여 최적의 파라미터 탐색.

---

## 📉 Problem Solving & Improvements

초기 베이스라인 모델 수립 이후 성능을 끌어올리기 위해 다양한 가설을 검증했습니다.

### 💡 Issue 1: "More Features ≠ Better Performance"
- **문제:** Feature Engineering을 고도화하며 피처 수를 늘렸으나, 오히려 Validation 및 LB 점수가 하락하거나 정체되는 현상 발생
- **해결:** **Feature Selection (피처 선택)** 적용.
  - `Feature Importance (Gain)` 분석을 통해 기여도가 낮은 변수 식별.
  - **Adversarial Validation** 등을 참고하여 Train/Test 분포가 다른 변수 제거.
  - 불필요한 변수를 제거함으로써 모델의 일반화 성능을 높이고 `0.8570`의 최고 점수 달성.

### 💡 Issue 2: Ensemble Failed
- **시도:** 성능 향상을 위해 Seed Averaging 및 이종 모델(XGBoost, CatBoost) 간의 Blending 시도.
- **결과:** 단일 모델보다 성능이 하락함.
- **원인 분석:**
  1.  **높은 Lag_1 의존도:** 생성된 모든 모델이 `target_lag_1`(직전 달 판매량) 변수에 압도적으로 높은 의존도를 보임.
  2.  **낮은 다양성(Diversity):** 모델들이 거의 동일한 패턴으로 오답을 내고 있어, 앙상블을 통한 분산 감소 효과가 미미함.
  3.  **Local Minima:** LightGBM 모델이 특정 패턴(Local Minima)에 과도하게 수렴했을 가능성 존재.

---

## 📂 File Structure

```
.
├── experiments/            # notebook files
├── 04_00_PREP.ipynb        # Data Preprocessing & Feature Engineering
├── 05_00_lgbm.ipynb        # Modeling, Tuning, Analysis
└── README.md               # Project Report
```

## 🛠 Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** LightGBM, Scikit-learn
- **Optimization:** Optuna
- **Visualization:** Matplotlib, Seaborn

---

## 📝 Conclusion
시계열 데이터 예측에서 **'직전 달의 판매량(Lag_1)'**이 가장 강력한 피처임을 확인했습니다. 하지만 이에 대한 과도한 의존은 모델의 다양성을 해치고 앙상블 효과를 저해할 수 있음을 배웠습니다. 향후에는 Lag 변수 의존도를 낮추기 위해 **Rolling Window Statistics**를 더 길게 잡거나, 비선형적인 패턴을 잡을 수 있는 딥러닝 기반의 접근을 혼합하는 방식을 고려해볼 만합니다.