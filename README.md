# Options Pricing & Implied Volaitlity Surface Platform

Comprehensive options pricing and analysis application with multiple financial and machine learning models for options pricing. The app provides real-time low-latencty pricing, backtesting capabilities, and professional-grade analytics through an intuitive Streamlit web interface.

[**Live Demo**](https://advancedoptionspricing.streamlit.app/)

<img width="1458" alt="Screenshot 2025-06-29 at 10 13 47 PM" src="https://github.com/user-attachments/assets/49e4f22c-c44f-4ffb-80eb-2a2e56947e1e" />

<img width="1416" alt="Screenshot 2025-06-29 at 10 13 57 PM" src="https://github.com/user-attachments/assets/72612a03-9529-4235-ba39-1f1ab71ab12d" />

## Pricing Models
1. Black-Scholes Model:
    - Analytical European option pricing with complete Greeks
    - See detailed literature & research notes in `literature.md`
    - Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy, 81(3), 637-654
2. Monte Carlo Simulation:
    - Heston stochastic volatility model with multi-path visualization  
3. Binomial Tree Method:
     - Discrete-time lattice approach for American/European options
  
<img width="1451" alt="Screenshot 2025-06-29 at 10 14 15 PM" src="https://github.com/user-attachments/assets/523e8233-f544-49af-a2d5-7efebd2345ff" />

## Backtesting: 
- Pricing models are calibrated via a comprehensive historical backtesting engine using SPX options data
- Data is pre-processed and filtered to only yield high-quality contracts, and predicted vs. actual prices are measured
- Models are calibrated to SPX surfaces & optimized for backtesting

![Screenshot 2025-06-30 at 12 14 47 AM](https://github.com/user-attachments/assets/9ea0b4ed-6836-4558-b196-25e6eb4e2d60)

## Features

- Option Pricing Calculator:
   - Paraneters: Ticker symbol, option type (Call/Put), strike price and expiration, risk-free rate and volatility (Use Live Market Data/Historical)
   - Automatic price and volatility fetching; real-time risk-free rate updates
   - All model prices display side-by-side with complete Greeks calculation
   - Price sensitivity analysis & Monte Carlo path visualization
- Performance Metrics:
    - MAE, RMSE, percentage errors across all models
    - Statistical significance testing
    - Error distribution analysis
    - Performance by option characteristics
- Interactive Visualizations:
    - 3D heatmaps & convergence analysis for pricing models
    - Model accuracy scatter plots & error distribution box plots for backtesting
    - 3D volatility surfaces across strikes and expirations for volatility smile analysis

- <img width="740" alt="Screenshot 2025-06-29 at 10 14 27 PM" src="https://github.com/user-attachments/assets/fb9d24a3-b063-4c2e-b18b-cdb4f978f6ac" />

## Usage

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

```bash
# Clone the repository
git clone [your-repository-url]
cd quant-options-pricing

# Run setup script
chmod +x setup.sh
./setup.sh
```

### Quick Start

#### 1. Launch Streamlit Application

```bash
# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"

# Run the main application
streamlit run frontend/streamlit.py
```

The application will open in your browser at `http://localhost:8501`

#### 2. Run Backtesting Analysis

```bash
# Jupyter notebook
jupyter notebook backend/backtesting/backtesting.ipynb
```

#### 3. Data Requirements

Use the default SPX options dataset or use your SPX options dataset in `src/backend/data/all_options_all_dates.csv`

**Expected CSV columns:**
- `contractSymbol`: Option contract identifier
- `strike`: Strike price
- `bid`: Bid price
- `ask`: Ask price
- `impliedVolatility`: Market implied volatility
- `Expiration`: Expiration date
- `Type`: 'call' or 'put'
- `lastTradeDate`: Trade date
- `volume`: Trading volume
- `openInterest`: Open interest

## Advanced Configurations

### Custom Risk-Free Rate
```python
# In your Python scripts
from backend.data.data_fetcher import DataFetcher
fetcher = DataFetcher()
fetcher.risk_free_rate = 0.045  # 4.5%
```

### Monte Carlo Parameters
```python
# Adjust simulation parameters
from backend.models.option_models import OptionPricingModels
pricing = OptionPricingModels(S, K, T, r, sigma, option_type)
mc_price, paths = pricing.new_monte_carlo_option_price(num_simulations=50000)
```

### Binomial Tree Resolution
```python
# Higher resolution binomial tree
bt_price = pricing.binomial_tree_option_price(N=200)  # 200 steps
```

## License

MIT License - Open source for educational and research purposes.
