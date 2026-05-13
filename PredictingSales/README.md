# Predict Future Sales: Time Series Forecasting

Kaggle [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales) 컴피티션 솔루션입니다. 러시아 소프트웨어 회사인 1C Company의 일별 판매 데이터를 월별 시계열 데이터로 변환하여 **Month 34의 상점-상품별 월간 판매량**을 예측하는 회귀 문제입니다.

## Result

- **Private Leaderboard Score (RMSE):** `0.8570` (LightGBM Single Model)
- **Rank:** Top 7%

## Key Approach

최종 재현 파이프라인은 `04_00_PREP.ipynb`와 `05_00_lgbm.ipynb`를 기준으로 `.py` 모듈로 리팩토링했습니다. 시계열 데이터의 특성을 반영한 **grid 데이터 구축**과 **lag feature 생성**에 집중했으며, 과적합을 방지하기 위한 **feature selection**이 성능 개선의 핵심이었습니다.

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

## Problem Solving & Improvements

초기 베이스라인 모델 수립 이후 성능을 끌어올리기 위해 여러 가설을 검증했습니다.

### Issue 1: More Features != Better Performance

Feature engineering을 고도화하면서 피처 수를 늘렸지만, validation 및 leaderboard 점수가 오히려 하락하거나 정체되는 현상이 있었습니다.

해결 방향:

- `Feature Importance (Gain)` 분석을 통해 기여도가 낮은 변수를 식별했습니다.
- Adversarial validation 결과를 참고하여 train/test 분포가 다른 변수를 제거했습니다.
- 불필요한 변수를 줄여 모델의 일반화 성능을 높였고, 최종적으로 `0.8570`의 private leaderboard 점수를 달성했습니다.

### Issue 2: Ensemble Failed

성능 향상을 위해 seed averaging 및 이종 모델(XGBoost, CatBoost) 간 blending을 시도했지만, 단일 LightGBM 모델보다 성능이 낮았습니다.

원인 분석:

- 생성된 모델들이 모두 `target_lag_1`에 강하게 의존했습니다.
- 모델들이 거의 같은 패턴으로 오답을 내면서 ensemble을 통한 분산 감소 효과가 작았습니다.
- LightGBM 단일 모델이 특정 패턴에 과도하게 수렴했을 가능성도 있었습니다.

결론적으로 이 프로젝트에서는 ensemble보다 feature selection을 거친 LightGBM 단일 모델이 가장 좋은 결과를 냈습니다.

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

## Conclusion

시계열 판매량 예측에서 직전 달 판매량 계열의 lag feature가 가장 강력한 신호임을 확인했습니다. 다만 이에 대한 과도한 의존은 모델 다양성을 낮춰 ensemble 효과를 제한할 수 있습니다. 향후에는 더 긴 rolling window statistics나 비선형 패턴을 포착하는 접근을 함께 검토할 수 있습니다.
