# Model catalogue and reproducibility notes

> Scope: This catalogue describes the **source-code snapshot** supplied for
> `Financial-Advisory-Chatbot`. It is not a benchmark report or certification.
> File references are relative to the repository root. Training artifacts,
> learned weights, dataset freshness and reported performance have **not been
> independently verified**. A number of scripts require local data and paths.

## System data flow (current design)

```text
Optional user financial questionnaire (BART-MNLI classification -> weighted scores)
                 |
Portfolio file -> user-risk assessment questionnaire (rule-based risk score)
                 |
Market data (yfinance) -> K-Means stock clustering -> diversified candidates
                                                     |
Reddit / YouTube -> BERT sentiment + Yahoo beta ----> | -> ranked recommendations
                                                     |
Market history -> Prophet / ARMA / SARIMAX / LSTM --->| (separate forecasts)
Gold / commodity history -> XGBoost ----------------->| (separate index insight)
                                                     |
                                    Random-search portfolio optimizers

NewsAPI -> FinBERT financial-news sentiment -> Streamlit sentiment display

NEW, OPT-IN: Streamlit research page -> LLM planning loop -> read-only Yahoo
          Finance adjusted-price/volatility tools -> evidence-aware response
```

**Integration note:** The diagram summarizes intended module relationships.
Several integrations depend on file-based hand-offs, manually copied scripts
and absolute filesystem paths. It is not a verified end-to-end runtime graph.
The new research page does **not** automatically consume other model results,
update recommendation CSVs or change user portfolios.

## Summary

| Capability | Implementation / type | Inputs -> outputs | Source |
| --- | --- | --- | --- |
| Financial answer interpretation | `facebook/bart-large-mnli` zero-shot classifier | Free-text questionnaire answer -> label + score; weighted risk score CSV | `Financial_Question/financial question.py` |
| Financial-risk aptitude | Weighted, rule-based multiple-choice questionnaire | Responses -> normalized risk tolerance | `User_Risk_Assessment_Module/User_risk_assessment.py`; `UI/app.py` |
| Market-news sentiment | `ProsusAI/finbert` 3-class BERT classifier | News headline/description -> positive/neutral/negative and confidence | `News_sentiment_analysis_training_of_FINBERT/News_sentiment_analysis.py` |
| Financial-news fine-tuning | FinBERT sequence classification (`BertForSequenceClassification`) | FinancialPhraseBank and news sentences -> fine-tuned checkpoint | `News_sentiment_analysis_training_of_FINBERT/training_of_FinBert.py` |
| Reddit/YouTube sentiment and risk | Local 3-class BERT checkpoint + beta blend | Posts/comments + Yahoo beta -> stock risk proxy | `Stock_risk_assessment/stock_risk.py` |
| Diversification | K-Means with scaled numeric/categorical features | Stock fundamental/return features -> cluster ID and outside-cluster candidates | `diversification_module/Diversification_clusters.py` |
| Price forecasting | Prophet, ARMA (via `ARIMA(p,0,q)`), SARIMAX, LSTM | Historical closes -> estimated future closes | `Stock_prediction_module/stock_prediction.py`; `Stock_price_prediction_using_other_commodities/USING_SARIMAX_LSTM.py` |
| Commodity-driven index prediction | Two XGBoost regressors | Gold price and lag/statistical features -> Nifty 50 / S&P 500 predictions | `Stock_price_prediction_using_other_commodities/commodities_price_prediction.py` |
| Recommendation ranking | Heuristic scoring (not learned model) | Cluster candidates + volatility/risk/return/size/P-E -> top six tickers | `Stock_Recommendation_Module/Stock_Recommendation.py` |
| Portfolio weight search | Repeated random perturbation/initializations (not a trained model) | Prices + profile tolerance -> candidate weights and objective | `Portfolio_weights_optimization/optimization.py`; `returns_optimization.py` |
| Text translation | Helsinki-NLP MarianMT models | Non-English answers -> English and back | `Financial_Question/Multi_Modal_Data_Handling/User_Interaction.py` |
| Transaction extraction | OpenAI Chat Completions (legacy code uses `gpt-4`) | Transcribed text -> asset/shares/date/price JSON | `Financial_Question/Multi_Modal_Data_Handling/User_Interaction.py` |
| Optional agent (this change) | Configured chat model as *planner*, not a market-prediction model | Question -> validated tool calls -> response | `agentic_advisor/` |

## 1. Financial questionnaire: BART-MNLI

- **Purpose:** Interpret free-form responses into one of the predefined labels
  for 17 finance/personal-profile questions. Numerical entries are handled
  separately.
- **Checkpoint / task:** `facebook/bart-large-mnli`, Transformers
  `pipeline('zero-shot-classification', multi_label=False)`, CPU (`device=-1`).
- **Features:** Question-specific candidate labels (for example, risk capacity
  `Very low` ... `Very high`, and investment horizon `0-3 years`, `3-5 years`,
  `5+ years`).
- **Output:** Top predicted label and a classification score; additional
  hand-coded functions normalize the answer's numeric contribution. The UI
  multiplies a base score by confidence before writing `answers.csv`.
- **Checkpoint origin:** Hugging Face pretrained. No local fine-tuning shown.
- **Limitations:** A zero-shot confidence is **not** a calibrated probability of
  a client's risk tolerance. Scores also encode assumptions about financial
  and personal attributes, so the resulting risk score should not be treated
  as a validated suitability assessment. Avoid transferring the raw sensitive
  questionnaire to third-party services without consent. Check the exact
  scoring scheme before deployment.

## 2. Financial-risk aptitude questionnaire

- **Implementation:** Multiple-choice answer weights summed and divided by
  the maximum possible score; labels map roughly to low, moderate and high
  risk tolerance. A parallel Streamlit questionnaire is in `UI/app.py`.
- **Inputs / outputs:** Approximately 20 fixed-option questions; score and
  categorical profile. It does **not** fit a statistical model.
- **Limitations:** Hardcoded conversion factor `USD_TO_INR_RATE = 100` is not a
  live exchange rate. The scoring is a subjective heuristic, not a substitute
  for a regulated risk/suitability process. The two questionnaire modules may
  use different question subsets or implementations.

## 3. Financial-news sentiment: FinBERT

**Inference:**

- **Checkpoint:** `ProsusAI/finbert`; three-class sequence classifier.
- **Inputs:** NewsAPI `title` plus `description`, truncated/padded to 128
  tokens; a max of 100 articles per requested ticker in the standalone module.
- **Outputs:** Label `Negative`/`Neutral`/`Positive` plus softmax confidence;
  top `n` articles by confidence are exported to `top_articles.csv`.
- **Limitations:** Confidence is not article relevance or factuality; ranking
  by confidence can prioritize strongly worded but irrelevant articles. A
  missing/failed data fetch should not be interpreted as neutral sentiment.
  Verify the hardcoded label order against `model.config.id2label` for the
  exact checkpoint revision; otherwise labels may be misassigned. In the
  standalone script the API credential is required at import time.

**Fine-tuning script:**

| Training property | Value visible in code |
| --- | --- |
| Base checkpoint | `ProsusAI/finbert` |
| Class IDs | negative 0, neutral 1, positive 2 |
| Datasets | FinancialPhraseBank v1.0 and `News_sentiment_Jan2017_to_Apr2021.csv` |
| Training / validation / test | 70% / 15% / 15%, stratified; `random_state=4` |
| Sequence length | 128 tokens |
| Batch size | 16 |
| Epochs | 4 |
| Learning rate | `2e-5` with `AdamW`, linear scheduler, zero warmup |
| Evaluation | Accuracy + classification report printed for validation and test |
| Saved artifact | `./temp` (`save_pretrained` weights and tokenizer) |

**Provenance and licensing:** FinancialPhraseBank data is described as 4,840
sentences with overlapping agreement subsets in its bundled README. Its
bundled license specifies **CC BY-NC-SA 3.0**, with commercial-use restrictions;
review upstream license terms before redistributing or using it commercially.
The training script removes duplicate sentences across agreement buckets; its
news data sampling should be audited for leakage and label bias. No numerical
benchmark results were provided as committed evidence here, so do not claim a
specific accuracy. The UI separately loads its local `temp` artifact and cannot
run from that path unless the weights are present.

## 4. Reddit / YouTube stock-risk proxy

- **Checkpoint:** Local BERT sequence-classification weights under
  `sentiment_weights/` with expected three-class labels. It is not established
  that this local checkpoint is identical to the FinBERT fine-tune above.
- **Inputs:** Reddit submissions (optional YouTube comments), yfinance beta.
- **Output:** Signed sentiment contribution combined with beta:
  `risk_score = 0.9 * beta + 0.1 * (-mean_adjusted_sentiment)`.
- **Important discrepancy:** A nearby comment says *80% beta/20% sentiment*,
  but the executable expression uses **90%/10%**. The score is not normalized
  to 0-1 and should not be compared directly to questionnaire risk tolerance.
- **Limitations:** Missing beta is silently set to zero, missing posts can
  produce NaN, public sentiment is manipulable, and scoring can be sensitive
  to small samples. The model and data access occur during module import.

## 5. K-Means portfolio diversification

- **Input:** Yahoo Finance tickers and daily price history. CSV feature
  columns include ticker, sector, market cap, P/E, average return, volatility.
- **Features actually used for clustering:** One-hot-encoded **sector**, and
  standardized **market cap** and daily-return **volatility**. Average return
  ranks candidates after clustering, not during the K-Means fit.
- **Training:** `KMeans(init='k-means++', random_state=42)`; an elbow-style
  inertia comparison chooses an `optimal_k` from a bounded range (`max_k=10`).
- **Outputs:** `indian_stocks_clusters.csv` and top-per-cluster candidates in
  `recommended_stocks.csv` for currently unrepresented user clusters.
- **Limitations:** Missing fundamental fields remove securities, new sectors
  and changing market caps shift clusters, and simplistic elbow logic is not
  a validated optimality measure. Silhouette score is imported but not used in
  the shown selection. Out-of-cluster != genuinely uncorrelated investments.

## 6. Price forecasting models

**Prophet:** Two years of historical closing prices by default; linear growth,
additive seasonality, daily on, weekly/yearly off. Explicit settings in
`Stock_prediction_module/stock_prediction.py` include
`changepoint_prior_scale=0.014900306553726704`,
`seasonality_prior_scale=0.3836198463360881`,
`changepoint_range=0.8205879350577062`, and `n_changepoints=41`. A two-last-
observations holdout is used to report MSE before 7-step prediction. Prophet is
**not** a SARIMA implementation despite a comment suggesting otherwise.

**ARMA:** `statsmodels.ARIMA` with differencing `d=0`, and `p`/`q` selected
from ACF/PACF using up to 10 lags and a 0.2 threshold. The script fits on 30
points preceding the two-point test and compares MSE with Prophet. Its default
ARMA date range ends **2024-11-06**, so results can be stale until updated.
A seven-step ARMA forecast is not necessarily seven exchange sessions unless
exchange calendars and index alignment are applied correctly.

**SARIMAX:** Iterates candidate `(p,d,q)` and `(P,D,Q,s)` orders; evaluates
rolling slices with RMSE and trains a selected order. The shown routine lacks
initialization of `best_rmse` before comparison. The full exhaustive grid is
computationally expensive and not safe to invoke on every UI request without
limits. No exogenous commodity regressors are visibly passed to `SARIMAX`.

**LSTM:** TensorFlow Sequential with two 50-unit LSTM layers followed by
`Dense(1)`, MinMax scaling, `look_back=60`, 20 epochs, batch size 32, Adam/MSE.
The dataset construction uses seven future target points but the final layer
emits **one** prediction; inspect shape/broadcasting during training and before
interpreting output as a seven-day forecast. The scaler is fitted before the
80/20 time split, creating potential leakage from future values into training
normalization. Validation is done against the test slice, which is not a
separate model-selection validation set.

**Model selection:** Scripts compare holdout MSE and choose the smallest;
some routines use only two test observations, while LSTM/SARIMAX paths use
different data windows. They should be compared on a common chronological
holdout and metric before claiming the model is best. Walk-forward validation,
corporate-action handling, and baselines (naive last close / drift) are needed.

## 7. Gold-to-index XGBoost regressors

- **Models:** Two `xgboost.XGBRegressor()` objects load JSON artifacts
  `xgboost_model_india.json` and `xgboost_model_us.json` from `PROJECT_DIR`.
- **Inputs:** Yahoo Finance gold futures `GC=F`; gold close, previous six
  closes, rolling mean, variance and difference. Commodity descriptive inputs
  also reference `CL=F`, `HG=F`, and ETF proxies `LIT` and `BATT`.
- **Outputs:** Predicted Nifty 50 / S&P 500 values or changes saved as text and
  accompanying price CSVs.
- **Limitations:** Model-training hyperparameters, train/test dates, fitted
  preprocessing, target definition and error metrics are **not in the supplied
  inference source**; report them only after locating the training pipeline
  and checkpoint metadata. ETF proxies are not spot commodity prices.
  Features must exactly match training feature names and order. Local absolute
  paths under `/home/ramanant/myenv/project/` must be replaced for portability.

## 8. Recommendation and optimization (heuristics, not predictive ML)

- **Recommendation:** Ranks cluster candidates by heuristics for stock risk,
  volatility, P/E ratio, market cap and recent returns; top six saved/displayed.
  In code, actual bonuses differ from the README's description of a `+3`
  matching-category bonus; compare code with intended scoring specification.
  Normalization may divide by zero when all tickers have the same risk score.
- **Sharpe optimizer:** Perturbs positive weights and renormalizes them with
  repeated random initializations. It is labeled stochastic gradient descent
  in comments, but no gradient is computed; it is a random local search. It
  combines US and INR securities, applies FX conversions and hardcodes
  approximate country risk-free rates. Output: weights and estimated Sharpe.
- **Risk-constrained return optimizer:** Uses a softmax-transformed correlation
  proxy and a questionnaire-derived 0-1 tolerance rather than standard
  portfolio variance in the constraint. It is named `convex_optimisation`, but
  its implementation is also random perturbation rather than a CVXPY solve.
  Objective/units should be independently validated before financial use.

## 9. Translation and transaction extraction

`Financial_Question/Multi_Modal_Data_Handling/User_Interaction.py` uses
MarianMT checkpoints `Helsinki-NLP/opus-mt-{src_lang}-en` and the reverse
language direction, then calls the legacy OpenAI Chat Completions interface for
asset name, share quantity, purchase date and price. It supports text, PDF
(with optional OCR) and audio. The module recursively reprocesses replies for
missing fields without maintaining a durable merged extraction state; an
explicit bounded merge-and-confirm loop is recommended before production. This
PR intentionally leaves that legacy path unchanged, avoiding an unreviewed
rewrite of sensitive transaction handling. The new read-only research agent
must not be confused with transaction extraction or order placement.

## Reproducibility and pre-production checklist

1. Record data sources, data retrieval date, corporate-action handling,
   calendars, target horizon and chronological train/validation/test splits.
2. Identify *exact* checkpoint versions/revisions and artifact hashes;
   document who trained them and on which data.
3. Run reproducible walk-forward baselines, stratified sentiment evaluation,
   calibration assessment and subgroup checks; publish actual MAE/RMSE,
   confusion matrices and confidence intervals rather than assumed accuracy.
4. Remove hardcoded absolute paths; verify all training and local inference
   assets are present, licensed, and referenced relative to an explicit
   configurable data directory.
5. Validate financial scores, API failures, missing data, FX conversion and
   cross-market time zones. Do not treat empty sentiment as neutral evidence.
6. Store credentials only in environment/secret management. Remove all
   examples of real tokens from source, logs and notebooks. Treat portfolio and
   questionnaire values as sensitive.
7. Test runtime on a clean environment; the full legacy application is not
   established to be runnable from repository checkout alone.

No module here has demonstrated broker-order permissions. These are research
and prototype scoring components; they do not constitute regulated investment
advice, guaranteed returns, or a production-grade financial suitability model.
