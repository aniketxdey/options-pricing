# Quantra: Real-time options pricing & implied volatility fititng

Quantra is an online, educational options pricing and analysis platform that provides real-time low-latency pricing, backtesting capabilities, and professional-grade volatility fitting through an intuitive Streamlit web interface. Quantra was developed to provide MATH 86 students at Dartmouth College with a platform to understand firsthand relationships & sesnitivities among the Greeks, hedging & portfolio management workflows, and how traditional pricing models are deployed.

[**Website Link**](https://advancedoptionspricing.streamlit.app/)

<img width="1119" height="729" alt="Screenshot 2026-05-03 at 5 19 28 PM" src="https://github.com/user-attachments/assets/c842c3f6-204f-4117-809e-ce0855d2a3d3" />

## Pricing

The pricing page values any listed option in real time. It pulls live quotes from YFinance, runs three independent pricing models on the same parameters, and shows the results side by side so model disagreement is visible at a glance.

- Live data fetched from YFinance: stock price, 30-day realized volatility, and the underlying chain
- Three pricing models calibrated in parallel: Black-Scholes (closed-form), Binomial Tree (Cox-Ross-Rubinstein, 100 steps), and Monte Carlo (10,000 GBM paths)
- Greeks computed two ways on the same view: analytical Greeks from the BS pricer, and numerical Greeks via central differences. The two are placed next to each other so the numerical pipeline can be checked against the closed form.
- Sensitivity chart that plots option value across an 80% to 120% spot range, with current spot and strike marked
- Hedging panel: enter a contract count and the page returns portfolio delta, gamma, vega, theta, the share count needed for delta-neutral hedging, and the dollar notional of that hedge
- Live volatility smile pulled for the chosen ticker after the calculation runs

<img width="1117" height="828" alt="Screenshot 2026-05-03 at 5 19 16 PM" src="https://github.com/user-attachments/assets/b7775344-f47d-4cd2-9c26-a907863308f7" />

## Volatility Surface

Fits an implied volatility surface to the live YFinance options chain for any ticker. Three solver options and three view modes cover the common workflows in surface analysis.

- IV solvers: Brent, Newton-Raphson, and Bisection, with automatic fallback if a solver fails to converge
- Three views of the same data: 3D surface, volatility smile across strikes for the nearest expiry, and a heatmap of moneyness vs days-to-expiry
- Surface diagnostics: current spot, ATM IV at short and long expiries, term-structure slope, skew, and total contract count
- Raw contract table with one-click CSV export so the fit can be carried into other tools

<img width="1049" height="751" alt="Screenshot 2026-05-03 at 5 27 15 PM" src="https://github.com/user-attachments/assets/3115e85f-a0b4-428c-a7db-3f23d5da9f49" />

## Backtesting

Replays the model stack against historical SPX options data and scores every model on per-contract pricing error.

- Sample size slider from 100 to 5,000 concurrent contracts per run
- Re-prices each contract with the three traditional models plus five scikit-learn baselines: Random Forest, two Gradient Boosting variants, Linear Regression, and SVR
- Performance summary table with MAE, RMSE, and MAPE for every model
- Full-width predicted-vs-actual scatter with a perfect-prediction reference line, so systematic bias by price level is easy to spot
- Dataset metrics: contract count, distinct trade days, strike range, and average days-to-expiry
- One-click CSV export of the full result set

<img width="1114" height="857" alt="Screenshot 2026-05-03 at 5 20 47 PM" src="https://github.com/user-attachments/assets/c3371128-b064-40dc-b45d-6bbbc9b54860" />

## Quick Start

```bash
git clone https://github.com/aniketxdey/options-pricing.git
cd options-pricing

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

## Data Requirements

The backtester reads from `src/backend/data/all_options_all_dates.csv`. A default SPX dataset is included. To swap in your own data, match these columns:

| Column | Description |
|---|---|
| `contractSymbol` | Option contract identifier |
| `strike` | Strike price |
| `bid` | Bid price |
| `ask` | Ask price |
| `impliedVolatility` | Market implied volatility |
| `Expiration` | Expiration date |
| `Type` | `call` or `put` |
| `lastTradeDate` | Trade date |
| `volume` | Trading volume |
| `openInterest` | Open interest |

## Advanced Configuration

Custom risk-free rate:

```python
from backend.data.data_fetcher import DataFetcher
fetcher = DataFetcher()
fetcher.risk_free_rate = 0.045  # 4.5%
```

Monte Carlo path count:

```python
from backend.models.option_models import OptionPricingModels
pricing = OptionPricingModels(S, K, T, r, sigma, option_type)
mc_price, paths = pricing.new_monte_carlo_option_price(num_simulations=50000)
```

Binomial tree resolution:

```python
bt_price = pricing.binomial_tree_option_price(N=200)
```

## License

MIT License. Open source for educational and research use.
