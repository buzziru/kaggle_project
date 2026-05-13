# Predict Future Sales: Time Series Forecasting

Kaggle [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales) 컴피티션 솔루션입니다. 1C Company의 일별 판매 데이터를 월별 시계열 데이터로 변환하여 **Month 34의 상점-상품별 월간 판매량**을 예측합니다.

## Result

- **Private Leaderboard Score (RMSE):** `0.8570` (LightGBM Single Model)
- **Rank:** Top 7%

## Key Approach

최종 재현 파이프라인은 `04_00_PREP.ipynb`와 `05_00_lgbm.ipynb`를 기준으로 `.py` 모듈로 리팩토링했습니다.

- **Preprocessing & Feature Engineering**
  - 이상치 제거, shop id 보정, test shop 기준 필터링
  - 월별 shop-item grid 생성
  - shop/item/category 텍스트 기반 파생 변수 생성
  - lag, rolling, diff, price trend, item age, last sale feature 생성

- **Modeling**
  - LightGBM 단일 모델
  - Tweedie objective
  - Train: Month 0-32
  - Validation: Month 33
  - Test: Month 34

## Project Structure

```text
PredictingSales/
├── README.md
├── data/                    # raw/intermediate/output data
├── notebooks/               # final reference notebooks
│   ├── 04_00_PREP.ipynb
│   └── 05_00_lgbm.ipynb
├── experiments/             # exploratory notebooks and failed model trials
├── src/
│   └── predicting_sales/
│       ├── config.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── modeling.py
│       └── pipeline.py
└── scripts/
    ├── prepare_data.py
    └── train_lgbm.py
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Kaggle 원본 CSV 파일을 `PredictingSales/data/`에 둡니다.

필요한 파일:

- `sales_train.csv`
- `items.csv`
- `item_categories.csv`
- `shops.csv`
- `test.csv`

전처리 및 feature matrix 생성:

```bash
python scripts/prepare_data.py --data-dir ./data --output ./data/all_data_result.joblib
```

LightGBM 학습 및 제출 파일 생성:

```bash
python scripts/train_lgbm.py --data-dir ./data --matrix ./data/all_data_result.joblib --submission ./data/submission_lgbm.csv
```

## Refactoring Notes

- `src/predicting_sales/pipeline.py`는 전처리부터 최종 feature matrix 생성까지 연결합니다.
- `src/predicting_sales/modeling.py`는 LightGBM 학습, 검증 RMSE 계산, test 예측, submission 생성을 담당합니다.
- XGBoost, CatBoost, Adversarial Validation, EDA 노트북은 최종 재현 코드에서 제외하고 `experiments/`에 보존했습니다.
- Optuna 튜닝은 기본 실행 경로에서 제외하고, 최종 LightGBM 파라미터를 `config.py`에 고정했습니다.

## Tech Stack

- Python
- Pandas, NumPy
- LightGBM, Scikit-learn
- Joblib
