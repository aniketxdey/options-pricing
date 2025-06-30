import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

def format_currency(value):
    # Format number as currency
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"

def format_percentage(value, decimals=2):
    # Format number as percentage
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"

def calculate_moneyness(strike, spot_price):
    # Calculate option moneyness
    return strike / spot_price

def classify_moneyness(moneyness, threshold=0.05):
    # Classify option as ITM, ATM, or OTM
    if abs(moneyness - 1.0) <= threshold:
        return "ATM"
    elif moneyness < 1.0:
        return "ITM" 
    else:
        return "OTM"

def calculate_time_decay_factor(days_to_expiry):
    # Calculate time decay factor for theta analysis
    if days_to_expiry <= 0:
        return 0
    elif days_to_expiry <= 30:
        return 1.0  # High time decay
    elif days_to_expiry <= 90:
        return 0.5  # Medium time decay
    else:
        return 0.2  # Low time decay

def validate_option_parameters(S, K, T, r, sigma):
    # Validate option pricing parameters
    errors = []
    
    if S <= 0:
        errors.append("Stock price must be positive")
    if K <= 0:
        errors.append("Strike price must be positive")
    if T <= 0:
        errors.append("Time to expiration must be positive")
    if r < 0:
        errors.append("Risk-free rate cannot be negative")
    if sigma <= 0:
        errors.append("Volatility must be positive")
    
    return len(errors) == 0, errors

def create_comparison_table(results_dict, metrics=None):
    # Create comparison table for different models
    if metrics is None:
        metrics = ['price', 'delta', 'gamma', 'vega', 'theta', 'rho']
    
    comparison_data = []
    
    for model_name, results in results_dict.items():
        row = {'Model': model_name}
        for metric in metrics:
            if metric in results:
                if metric == 'price':
                    row[metric.title()] = format_currency(results[metric])
                else:
                    row[metric.title()] = f"{results[metric]:.4f}"
            else:
                row[metric.title()] = "N/A"
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def calculate_portfolio_greeks(positions):
    # Calculate portfolio-level Greeks
    portfolio_greeks = {
        'delta': 0,
        'gamma': 0,
        'vega': 0,
        'theta': 0,
        'rho': 0
    }
    
    for position in positions:
        quantity = position.get('quantity', 1)
        greeks = position.get('greeks', {})
        
        for greek in portfolio_greeks:
            if greek in greeks:
                portfolio_greeks[greek] += quantity * greeks[greek]
    
    return portfolio_greeks

def generate_strike_range(current_price, num_strikes=21, range_pct=0.20):
    # Generate strike prices around current price
    lower_bound = current_price * (1 - range_pct)
    upper_bound = current_price * (1 + range_pct)
    
    strikes = np.linspace(lower_bound, upper_bound, num_strikes)
    
    # Round to nearest 0.5 or 1.0 depending on price level
    if current_price < 50:
        strikes = np.round(strikes * 2) / 2  # Round to nearest 0.5
    else:
        strikes = np.round(strikes)  # Round to nearest 1.0
    
    return sorted(strikes)

def calculate_profit_loss(positions, spot_prices):
    # Calculate P&L for a portfolio across different spot prices
    pnl_data = []
    
    for spot_price in spot_prices:
        total_pnl = 0
        
        for position in positions:
            strike = position['strike']
            option_type = position['option_type']
            quantity = position['quantity']
            premium_paid = position.get('premium_paid', 0)
            
            # Calculate intrinsic value
            if option_type.lower() in ['call', 'c']:
                intrinsic = max(0, spot_price - strike)
            else:
                intrinsic = max(0, strike - spot_price)
            
            # P&L = (intrinsic value - premium paid) * quantity
            position_pnl = (intrinsic - premium_paid) * quantity
            total_pnl += position_pnl
        
        pnl_data.append({
            'spot_price': spot_price,
            'total_pnl': total_pnl
        })
    
    return pd.DataFrame(pnl_data)

def create_pnl_plot(pnl_df, title="Portfolio P&L"):
    # Create P&L visualization
    fig = go.Figure()
    
    # Add P&L line
    fig.add_trace(go.Scatter(
        x=pnl_df['spot_price'],
        y=pnl_df['total_pnl'],
        mode='lines',
        name='P&L',
        line=dict(color='blue', width=2)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Color regions
    fig.add_trace(go.Scatter(
        x=pnl_df['spot_price'],
        y=pnl_df['total_pnl'],
        fill='tonexty',
        fillcolor='rgba(0,255,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Profit'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Stock Price',
        yaxis_title='Profit/Loss',
        width=800,
        height=500
    )
    
    return fig

def safe_divide(numerator, denominator, default=0):
    # Safe division with default value
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def interpolate_volatility(strike_vol_pairs, target_strike):
    # Linear interpolation for volatility
    if not strike_vol_pairs:
        return None
    
    strikes = [pair[0] for pair in strike_vol_pairs]
    vols = [pair[1] for pair in strike_vol_pairs]
    
    if target_strike <= min(strikes):
        return vols[0]
    elif target_strike >= max(strikes):
        return vols[-1]
    else:
        return np.interp(target_strike, strikes, vols)

def create_greeks_heatmap(greeks_data, greek_name):
    # Create heatmap for Greeks visualization
    fig = go.Figure(data=go.Heatmap(
        z=greeks_data,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title=greek_name.title())
    ))
    
    fig.update_layout(
        title=f'{greek_name.title()} Heatmap',
        xaxis_title='Time to Expiration',
        yaxis_title='Strike Price',
        width=700,
        height=500
    )
    
    return fig

def ensure_output_directory(path="output"):
    # Ensure output directory exists
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_results_to_file(results, filename, output_dir="output"):
    # Save results dictionary to file
    ensure_output_directory(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    
    if filename.endswith('.csv'):
        if isinstance(results, dict):
            pd.DataFrame([results]).to_csv(filepath, index=False)
        elif isinstance(results, pd.DataFrame):
            results.to_csv(filepath, index=False)
    elif filename.endswith('.json'):
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    return filepath

def calculate_option_metrics(option_data):
    # Calculate comprehensive option metrics
    metrics = {}
    
    if 'bid' in option_data and 'ask' in option_data:
        metrics['bid_ask_spread'] = option_data['ask'] - option_data['bid']
        metrics['mid_price'] = (option_data['bid'] + option_data['ask']) / 2
        metrics['spread_percentage'] = safe_divide(
            metrics['bid_ask_spread'], 
            metrics['mid_price'], 
            0
        )
    
    if 'volume' in option_data:
        metrics['volume'] = option_data['volume']
    
    if 'openInterest' in option_data:
        metrics['open_interest'] = option_data['openInterest']
        if 'volume' in metrics:
            metrics['volume_oi_ratio'] = safe_divide(
                metrics['volume'], 
                metrics['open_interest'], 
                0
            )
    
    return metrics

def validate_csv_data(df, required_columns):
    # Validate CSV data structure
    missing_columns = []
    
    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)
    
    if missing_columns:
        return False, f"Missing columns: {missing_columns}"
    
    # Check for empty dataframe
    if df.empty:
        return False, "Dataframe is empty"
    
    return True, "Data validation passed"

def format_large_number(number):
    # Format large numbers with appropriate suffixes
    if pd.isna(number):
        return "N/A"
    
    abs_number = abs(number)
    
    if abs_number >= 1e9:
        return f"{number/1e9:.1f}B"
    elif abs_number >= 1e6:
        return f"{number/1e6:.1f}M"
    elif abs_number >= 1e3:
        return f"{number/1e3:.1f}K"
    else:
        return f"{number:.2f}"

def calculate_greeks_sensitivity(base_greeks, parameter_shift=0.01):
    # Calculate sensitivity of Greeks to parameter changes
    # This is a placeholder for more advanced sensitivity analysis
    sensitivity = {}
    
    for greek, value in base_greeks.items():
        # Simple sensitivity approximation
        sensitivity[f"{greek}_sensitivity"] = value * parameter_shift
    
    return sensitivity
