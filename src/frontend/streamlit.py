import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import time

# Add backend to path
sys.path.append('src/backend')

from models.option_models import OptionPricingModels, ImpliedVolatilityCalculator
from models.implied_volatility import ImpliedVolatilitySurface
from data.data_fetcher import DataFetcher
from utils.helpers import *
from backtesting.backtester import SPXBacktester

# Page configuration
st.set_page_config(
    page_title="Options Pricing & Implied Volatility Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalist black/white theme
st.markdown("""
<style>
    /* Import Satoshi font */
    @import url('https://fonts.googleapis.com/css2?family=Satoshi:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Main app styling */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Hide default streamlit elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stDecoration {display:none;}
    
    /* Full width container */
    .main .block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1rem;
    }
    

    
    .nav-button:hover {
        background-color: #333333;
        color: #ffffff;
        text-decoration: none;
    }
    
    .nav-button.active {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* Main title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
        margin: 2rem 0 3rem 0;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.1);
        font-family: 'Satoshi', sans-serif;
    }
    
    /* Page titles */
    .page-title {
        font-size: 2.8rem;
        font-weight: 600;
        color: #ffffff;
        margin: 1.5rem 0 2rem 0;
        text-align: center;
        font-family: 'Satoshi', sans-serif;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2rem;
        color: #ffffff;
        margin: 2rem 0 1.5rem 0;
        border-bottom: 2px solid #333;
        padding-bottom: 0.8rem;
        font-family: 'Satoshi', sans-serif;
        font-weight: 600;
    }
    
    /* Feature box headers */
    .feature-header {
        font-family: 'Satoshi', sans-serif;
        font-weight: 600;
    }
    
    /* Demo intro header */
    .demo-header {
        font-family: 'Satoshi', sans-serif;
        font-weight: 600;
    }
    
    /* Monospace labels */
    .monospace-label {
        font-family: 'Courier New', monospace;
        color: #cccccc;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Courier text for descriptions */
    .courier-text {
        font-family: 'Courier New', monospace;
        color: #cccccc;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Courier info text */
    .courier-info {
        font-family: 'Courier New', monospace;
        color: #cccccc;
        font-size: 1rem;
    }
    

    
    /* Metrics styling */
    .metric-container {
        background-color: #111111;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
        margin: 0.5rem 0;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    /* Home page content */
    .home-intro {
        background-color: #111111;
        padding: 3rem;
        border-radius: 20px;
        border: 1px solid #333;
        margin: 2rem 0;
        text-align: center;
        width: 100%;
    }
    
    .feature-box {
        background-color: #0a0a0a;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #333;
        margin: 1rem 0;
        min-height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    /* Override Streamlit default colors */
    .stSelectbox > div > div {
        background-color: #222222;
        color: #ffffff;
        border: 1px solid #444;
    }
    
    .stTextInput > div > div > input {
        background-color: #222222;
        color: #ffffff;
        border: 1px solid #444;
        padding: 0.8rem;
        font-size: 1rem;
    }
    
    .stNumberInput > div > div > input {
        background-color: #222222;
        color: #ffffff;
        border: 1px solid #444;
        padding: 0.8rem;
        font-size: 1rem;
    }
    
    .stSlider > div > div > div > div {
        background-color: #333333;
    }
    
    .stButton > button {
        background-color: white !important;
        color: black !important;
        border: 1px solid white !important;
        font-family: 'Satoshi', sans-serif !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 4px !important;
    }
    
    .stMetric {
        background-color: #1a1a1a !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid #333 !important;
        min-height: 120px !important;
    }
    
    .stMetric > div {
        color: white !important;
    }
    
    .stExpander {
        background-color: #1a1a1a !important;
        border: 1px solid #333 !important;
    }
    
    .stDataFrame {
        background-color: #1a1a1a !important;
    }
    
    .stSelectbox > div > div {
        background-color: #1a1a1a !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
    
    .stNumberInput > div > div {
        background-color: #1a1a1a !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
    
    .stTextInput > div > div {
        background-color: #1a1a1a !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
    
    .stSlider > div > div {
        color: white !important;
    }
    
    .stCheckbox > label {
        color: white !important;
    }
    
    .monospace-label {
        font-family: 'Courier New', monospace !important;
        color: white !important;
        font-size: 0.9rem !important;
        margin-bottom: 0.5rem !important;
        font-weight: normal !important;
    }
    
    .courier-info {
        font-family: 'Courier New', monospace !important;
        color: #ccc !important;
        font-size: 1rem !important;
        text-align: center !important;
        margin: 2rem 0 !important;
    }
    
    .feature-box {
        background-color: #1a1a1a !important;
        padding: 1.5rem !important;
        border-radius: 8px !important;
        border: 1px solid #333 !important;
        margin: 1rem 0 !important;
        min-height: 250px !important;
    }
    
    .feature-box h4 {
        color: white !important;
        font-family: 'Satoshi', sans-serif !important;
        font-size: 1.2rem !important;
        margin-bottom: 1rem !important;
    }
    
    .feature-box p {
        font-family: 'Courier New', monospace !important;
        color: #ccc !important;
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
    }
    
    /* Plotly charts dark theme */
    .js-plotly-plot {
        background-color: #000000 !important;
        border-radius: 10px;
    }
    
    /* Columns equal height */
    .row-widget.stHorizontal > div {
        display: flex;
        align-items: stretch;
    }
    
    .row-widget.stHorizontal > div > div {
        width: 100%;
    }
    
    /* Fix navigation button container */
    .nav-bar .stColumns {
        gap: 0 !important;
    }
    
    .nav-bar .stColumns > div {
        display: flex !important;
        align-items: stretch !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .nav-bar .stColumns > div > div {
        display: flex !important;
        height: 100% !important;
        width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Hide Streamlit footer */
    footer {
        visibility: hidden !important;
    }
    
    footer:after {
        content: '' !important;
        visibility: hidden !important;
        display: block !important;
    }
    
    .reportview-container .main footer {
        visibility: hidden !important;
    }
    
    /* Hide "Made with Streamlit" */
    .viewerBadge_container__1QSob {
        display: none !important;
    }
    
    .viewerBadge_link__1S137 {
        display: none !important;
    }
    
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Custom footer */
    .custom-footer {
        background-color: #000000;
        color: #ffffff;
        text-align: center;
        padding: 2rem 1rem;
        font-family: 'Satoshi', sans-serif;
        font-size: 0.9rem;
        border-top: 1px solid #333;
        margin-top: 3rem;
        width: 100%;
    }
    
    /* Navigation bar full width */
    .nav-bar {
        background-color: #111111;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid #333;
        border-radius: 10px;
        width: 100%;
    }
    
    /* Style navigation buttons */
    .nav-bar div[data-testid="column"] {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .nav-bar div[data-testid="stButton"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .nav-bar button {
        background-color: #222222 !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
        border-right: none !important;
        padding: 1rem 1.5rem !important;
        margin: 0 !important;
        cursor: pointer !important;
        font-size: 1rem !important;
        font-family: 'Satoshi', sans-serif !important;
        font-weight: 500 !important;
        height: 60px !important;
        width: 100% !important;
        border-radius: 0 !important;
        transition: all 0.3s ease !important;
    }
    
    .nav-bar div[data-testid="column"]:first-child button {
        border-radius: 5px 0 0 5px !important;
    }
    
    .nav-bar div[data-testid="column"]:last-child button {
        border-radius: 0 5px 5px 0 !important;
        border-right: 1px solid #444 !important;
    }
    
    .nav-bar button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    
    .nav-bar button:focus {
        background-color: #ffffff !important;
        color: #000000 !important;
        outline: none !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Function to generate random SPX volatility surface
def generate_demo_volatility_surface():
    """Generate a random SPX volatility surface for demo purposes"""
    # Create strike and expiry grids
    strikes = np.linspace(4800, 5200, 20)  # SPX strikes around current levels
    expiries = np.array([7, 14, 30, 60, 90, 120])  # Days to expiry
    
    strike_grid, expiry_grid = np.meshgrid(strikes, expiries)
    
    # Generate realistic volatility surface with smile and term structure
    base_vol = 0.18 + 0.02 * np.random.randn()  # Base volatility around 18%
    
    # Add volatility smile (higher vol for OTM options)
    current_spot = 5000 + 50 * np.random.randn()
    moneyness = strike_grid / current_spot
    smile_effect = 0.05 * (moneyness - 1)**2  # Parabolic smile
    
    # Add term structure (vol increases with time for SPX)
    term_effect = 0.02 * np.log(expiry_grid / 30)
    
    # Add some random noise
    noise = 0.01 * np.random.randn(*strike_grid.shape)
    
    vol_surface = base_vol + smile_effect + term_effect + noise
    vol_surface = np.maximum(vol_surface, 0.05)  # Minimum 5% vol
    
    return strike_grid, expiry_grid, vol_surface, current_spot

# Navigation bar with custom styling
st.markdown('<div class="nav-bar">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4, gap="small")

with col1:
    if st.button("Home", key="nav_home", use_container_width=True):
        st.session_state.current_page = "Home"
        
with col2:
    if st.button("Option Pricing", key="nav_pricing", use_container_width=True):
        st.session_state.current_page = "Option Pricing"
        
with col3:
    if st.button("Implied Volatility Surface", key="nav_iv", use_container_width=True):
        st.session_state.current_page = "Implied Volatility Surface"
        
with col4:
    if st.button("Backtesting Results", key="nav_backtest", use_container_width=True):
        st.session_state.current_page = "Backtesting Results"

st.markdown('</div>', unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-title">Options Pricing & Implied Volatility Platform</h1>', unsafe_allow_html=True)

# HOME PAGE
if st.session_state.current_page == "Home":
    # Main introduction
    st.markdown("""
    <div class="home-intro">
        <h3 class="demo-header" style="color: #ffffff; margin-bottom: 1.5rem;">Professional Options Analysis & Pricing Platform</h3>
        <p class="courier-text" style="font-size: 1.2rem; line-height: 1.8;">
            This platform combines traditional financial models with advanced machine learning techniques to provide 
            comprehensive options pricing and analysis. Calculate fair values using Black-Scholes, Monte Carlo, 
            and Binomial Tree methods, visualize implied volatility surfaces, and backtest strategies against 
            historical SPX options data with over 8,500 contracts.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature boxes with better layout
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4 class="feature-header" style="color: #ffffff; margin-bottom: 1.5rem; font-size: 1.3rem;">Pricing Models</h4>
            <p class="courier-text">
                Black-Scholes analytical solutions, Monte Carlo simulations with 10,000+ paths, 
                Binomial Tree discrete modeling, and machine learning models including Random Forest and XGBoost.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4 class="feature-header" style="color: #ffffff; margin-bottom: 1.5rem; font-size: 1.3rem;">Volatility Analysis</h4>
            <p class="courier-text">
                Generate 3D implied volatility surfaces, analyze volatility smiles and skews, 
                calculate term structure slopes, and export data for further analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h4 class="feature-header" style="color: #ffffff; margin-bottom: 1.5rem; font-size: 1.3rem;">Backtesting Engine</h4>
            <p class="courier-text">
                Comprehensive backtesting against SPX historical data with performance metrics, 
                model comparison charts, and statistical analysis including MAE, RMSE, and MAPE.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Generate and display the demo surface
    if st.button("Generate Random SPX Volatility Surface", type="primary", key="demo_surface"):
        st.session_state.demo_surface_generated = True
    
    # Auto-generate on first load or button press
    if not hasattr(st.session_state, 'demo_surface_generated'):
        st.session_state.demo_surface_generated = True
    
    if st.session_state.demo_surface_generated:
        strike_grid, expiry_grid, vol_surface, current_spot = generate_demo_volatility_surface()
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=strike_grid,
            y=expiry_grid,
            z=vol_surface * 100,  # Convert to percentage
            colorscale='Viridis',
            colorbar=dict(title="Implied Volatility (%)", titlefont=dict(color='white')),
            hovertemplate='Strike: %{x:.0f}<br>Days to Expiry: %{y:.0f}<br>IV: %{z:.1f}%<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'SPX Implied Volatility Surface (Spot: ${current_spot:.0f})',
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility (%)',
                bgcolor='black',
                xaxis=dict(backgroundcolor='black', gridcolor='gray', color='white'),
                yaxis=dict(backgroundcolor='black', gridcolor='gray', color='white'),
                zaxis=dict(backgroundcolor='black', gridcolor='gray', color='white')
            ),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            font_family='Satoshi',
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Surface statistics
        st.markdown('<h3 class="section-header">Live Surface Statistics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current SPX", f"${current_spot:.0f}", f"{np.random.choice(['+', '-'])}{abs(np.random.randn()*10):.1f}")
        with col2:
            atm_vol = vol_surface[2, 10] * 100  # Approximate ATM vol
            st.metric("ATM IV", f"{atm_vol:.1f}%", f"{np.random.choice(['+', '-'])}{abs(np.random.randn()*2):.2f}%")
        with col3:
            vol_range = (vol_surface.max() - vol_surface.min()) * 100
            st.metric("Vol Range", f"{vol_range:.1f}%")
        with col4:
            skew = (vol_surface[2, 5] - vol_surface[2, 15]) * 100  # Put-Call skew
            st.metric("Skew", f"{skew:.2f}%")
        with col5:
            term_slope = (vol_surface[5, 10] - vol_surface[0, 10]) * 100
            st.metric("Term Slope", f"{term_slope:.2f}%")
    
    # Platform statistics with better spacing
    st.markdown('<h3 class="section-header">Platform Statistics</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4, gap="large")
    
    with col1:
        st.metric("Pricing Models", "7+", "Traditional & ML")
    with col2:
        st.metric("Greeks Calculated", "5", "Complete Risk Profile")
    with col3:
        st.metric("SPX Data Points", "8,561", "Historical Options")
    with col4:
        st.metric("Supported Assets", "All", "Stocks, ETFs, Indices")

# OPTION PRICING PAGE
elif st.session_state.current_page == "Option Pricing":
    st.markdown('<h2 class="page-title">Option Pricing Calculator</h2>', unsafe_allow_html=True)
    
    # Input parameters with better layout
    st.markdown('<h3 class="section-header">Input Parameters</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown('<p class="monospace-label">ticker_symbol</p>', unsafe_allow_html=True)
        ticker = st.text_input("ticker_symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, MSFT)", label_visibility="collapsed")
        
        st.markdown('<p class="monospace-label">option_type</p>', unsafe_allow_html=True)
        option_type = st.selectbox("option_type", ["Call", "Put"], label_visibility="collapsed")
        
        st.markdown('<p class="monospace-label">strike_price</p>', unsafe_allow_html=True)
        strike_price = st.number_input("strike_price", min_value=0.01, value=150.0, step=0.5, label_visibility="collapsed")
        
    with col2:
        st.markdown('<p class="monospace-label">days_to_expiration</p>', unsafe_allow_html=True)
        days_to_expiry = st.number_input("days_to_expiration", min_value=1, value=30, step=1, key="days_exp", label_visibility="collapsed")
        
        st.markdown('<p class="monospace-label">risk_free_rate_percent</p>', unsafe_allow_html=True)
        risk_free_rate = st.number_input("risk_free_rate_percent", min_value=0.0, value=5.0, step=0.1, key="risk_rate", label_visibility="collapsed") / 100
        
        st.markdown('<p class="monospace-label">volatility_percent</p>', unsafe_allow_html=True)
        volatility = st.number_input("volatility_percent", min_value=0.1, value=25.0, step=0.5, key="volatility", label_visibility="collapsed") / 100
    
    with col3:
        st.markdown('<p class="monospace-label">use_live_market_data</p>', unsafe_allow_html=True)
        use_live_data = st.checkbox("use_live_market_data", value=True, label_visibility="collapsed")
        
        if use_live_data:
            with st.spinner("Fetching live data..."):
                current_price = st.session_state.data_fetcher.get_current_price(ticker)
                hist_vol = st.session_state.data_fetcher.calculate_historical_volatility(ticker)
                
                if current_price:
                    st.success(f"Current Price: ${current_price:.2f}")
                    stock_price = current_price
                else:
                    st.error("Could not fetch live data")
                    st.markdown('<p class="monospace-label">stock_price</p>', unsafe_allow_html=True)
                    stock_price = st.number_input("stock_price", min_value=0.01, value=150.0, step=0.1, key="stock_price", label_visibility="collapsed")
                
                if hist_vol:
                    st.info(f"Historical Vol: {hist_vol*100:.1f}%")
                    st.markdown('<p class="monospace-label">use_historical_volatility</p>', unsafe_allow_html=True)
                    if st.checkbox("use_historical_volatility", key="use_hist_vol", label_visibility="collapsed"):
                        volatility = hist_vol
        else:
            st.markdown('<p class="monospace-label">stock_price</p>', unsafe_allow_html=True)
            stock_price = st.number_input("stock_price_manual", min_value=0.01, value=150.0, step=0.1, key="manual_stock_price", label_visibility="collapsed")
    
    # Calculate button
    if st.button("Calculate Option Prices", type="primary"):
        time_to_expiry = days_to_expiry / 365.25
        
        # Validate inputs
        is_valid, errors = validate_option_parameters(stock_price, strike_price, time_to_expiry, risk_free_rate, volatility)
        
        if is_valid:
            with st.spinner("Calculating option prices..."):
                # Initialize pricing models
                pricing_models = OptionPricingModels(
                    S=stock_price,
                    K=strike_price,
                    T=time_to_expiry,
                    r=risk_free_rate,
                    sigma=volatility,
                    option_type=option_type.lower()
                )
                
                # Calculate prices
                bs_price = pricing_models.black_scholes_option()
                bt_price = pricing_models.binomial_tree_option_price(N=100)
                mc_price, mc_paths = pricing_models.new_monte_carlo_option_price(num_simulations=10000)
                
                # Calculate Greeks
                greeks = pricing_models.calculate_greeks()
                
                # Display results with better spacing
                st.markdown('<h3 class="section-header">Pricing Results</h3>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4, gap="large")
                
                with col1:
                    st.metric("Black-Scholes", f"${bs_price:.4f}", help="Analytical solution")
                with col2:
                    st.metric("Binomial Tree", f"${bt_price:.4f}", help="Discrete lattice model")
                with col3:
                    st.metric("Monte Carlo", f"${mc_price:.4f}", help="Simulation-based")
                with col4:
                    price_range = max(bs_price, bt_price, mc_price) - min(bs_price, bt_price, mc_price)
                    st.metric("Price Range", f"${price_range:.4f}", help="Difference between models")
                
                # Greeks display
                st.markdown('<h3 class="section-header">The Greeks</h3>', unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns(5, gap="large")
                
                with col1:
                    st.metric("Delta", f"{greeks['delta']:.4f}", help="Price sensitivity")
                with col2:
                    st.metric("Gamma", f"{greeks['gamma']:.6f}", help="Delta sensitivity")
                with col3:
                    st.metric("Vega", f"{greeks['vega']:.4f}", help="Volatility sensitivity")
                with col4:
                    st.metric("Theta", f"{greeks['theta']:.4f}", help="Time decay")
                with col5:
                    st.metric("Rho", f"{greeks['rho']:.4f}", help="Interest rate sensitivity")
                
                # Price sensitivity analysis
                st.markdown('<h3 class="section-header">Sensitivity Analysis</h3>', unsafe_allow_html=True)
                
                # Generate spot price range
                spot_range = np.linspace(stock_price * 0.8, stock_price * 1.2, 50)
                
                # Calculate prices across range
                sensitivity_prices = []
                for spot in spot_range:
                    temp_models = OptionPricingModels(
                        S=spot, K=strike_price, T=time_to_expiry,
                        r=risk_free_rate, sigma=volatility, option_type=option_type.lower()
                    )
                    price_result = temp_models.black_scholes_option()
                    sensitivity_prices.append(price_result)
                
                # Create sensitivity plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=spot_range,
                    y=sensitivity_prices,
                    mode='lines',
                    name='Option Price',
                    line=dict(color='white', width=3)
                ))
                
                # Add current price marker
                fig.add_vline(x=stock_price, line_dash="dash", line_color="red", 
                             annotation_text="Current Price")
                
                fig.update_layout(
                    title="Option Price vs Stock Price",
                    xaxis_title="Stock Price ($)",
                    yaxis_title="Option Price ($)",
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font_color='white',
                    font_family='Satoshi',
                    height=600,
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Integrated Volatility Smile Section
                st.markdown('<h3 class="section-header">Volatility Smile Analysis</h3>', unsafe_allow_html=True)
                
                with st.spinner("Generating volatility smile for selected ticker..."):
                    try:
                        # Initialize IV surface calculator with the ticker from pricing section
                        iv_surface = ImpliedVolatilitySurface(ticker)
                        
                        # Fetch options chain
                        options_data = iv_surface.fetch_options_chain()
                        
                        if not options_data.empty:
                            # Calculate IV for the chain
                            options_with_iv = iv_surface.calculate_iv_for_chain(options_data, method='brent')
                            
                            if not options_with_iv.empty:
                                # Generate volatility smile
                                smile_fig = iv_surface.analyze_volatility_smile(options_with_iv)
                                
                                if smile_fig:
                                    # Update the figure styling for dark theme
                                    smile_fig.update_layout(
                                        plot_bgcolor='black',
                                        paper_bgcolor='black',
                                        font_color='white',
                                        font_family='Satoshi',
                                        height=500,
                                        margin=dict(l=0, r=0, t=50, b=0)
                                    )
                                    st.plotly_chart(smile_fig, use_container_width=True)
                                
                                # Smile statistics
                                stats = iv_surface.calculate_surface_statistics(options_with_iv)
                                
                                col1, col2, col3 = st.columns(3, gap="large")
                                
                                with col1:
                                    st.metric("ATM IV Short", f"{stats.get('atm_iv_short', 0)*100:.1f}%")
                                
                                with col2:
                                    st.metric("ATM IV Long", f"{stats.get('atm_iv_long', 0)*100:.1f}%")
                                
                                with col3:
                                    st.metric("Volatility Skew", f"{stats.get('skew', 0)*100:.2f}%")
                            else:
                                st.info("No valid options data found for volatility smile analysis")
                        else:
                            st.info("No options data available for the selected ticker")
                            
                    except Exception as e:
                        st.info(f"Volatility smile analysis not available: {str(e)}")
                
        else:
            for error in errors:
                st.error(f"Error: {error}")

# IMPLIED VOLATILITY SURFACE PAGE
elif st.session_state.current_page == "Implied Volatility Surface":
    st.markdown('<h2 class="page-title">Implied Volatility Surface</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown('<h3 class="section-header">Surface Parameters</h3>', unsafe_allow_html=True)
        
        st.markdown('<p class="monospace-label">ticker_symbol</p>', unsafe_allow_html=True)
        iv_ticker = st.text_input("iv_ticker_symbol", value="SPY", key="iv_ticker", label_visibility="collapsed")
        
        st.markdown('<p class="monospace-label">surface_type</p>', unsafe_allow_html=True)
        surface_type = st.selectbox("surface_type", ["3D Surface", "Volatility Smile", "Heatmap"], label_visibility="collapsed")
        
        st.markdown('<p class="monospace-label">iv_calculation_method</p>', unsafe_allow_html=True)
        iv_method = st.selectbox("iv_calculation_method", ["Brent", "Newton-Raphson", "Bisection"], key="iv_method", label_visibility="collapsed")
        
        if st.button("Generate IV Surface", type="primary"):
            st.session_state.iv_surface_generated = True
            st.session_state.iv_ticker_selected = iv_ticker
            st.session_state.iv_surface_type = surface_type
            st.session_state.iv_method_selected = iv_method
    
    with col2:
        if hasattr(st.session_state, 'iv_surface_generated') and st.session_state.iv_surface_generated:
            with st.spinner("Fetching options data and calculating IV surface..."):
                try:
                    # Initialize IV surface calculator
                    iv_surface = ImpliedVolatilitySurface(st.session_state.iv_ticker_selected)
                    
                    # Fetch options chain
                    options_data = iv_surface.fetch_options_chain()
                    
                    if not options_data.empty:
                        # Calculate IV for the chain
                        options_with_iv = iv_surface.calculate_iv_for_chain(
                            options_data, 
                            method=st.session_state.iv_method_selected.lower()
                        )
                        
                        if not options_with_iv.empty:
                            # Generate surface plot
                            if st.session_state.iv_surface_type == "3D Surface":
                                fig = iv_surface.generate_surface_plot(options_with_iv, surface_type='3d')
                            elif st.session_state.iv_surface_type == "Volatility Smile":
                                fig = iv_surface.analyze_volatility_smile(options_with_iv)
                            else:  # Heatmap
                                fig = iv_surface.generate_surface_plot(options_with_iv, surface_type='heatmap')
                            
                            if fig:
                                # Update figure for dark theme
                                fig.update_layout(
                                    plot_bgcolor='black',
                                    paper_bgcolor='black',
                                    font_color='white',
                                    font_family='Satoshi',
                                    height=600,
                                    margin=dict(l=0, r=0, t=50, b=0)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Surface statistics
                            stats = iv_surface.calculate_surface_statistics(options_with_iv)
                            
                            st.markdown('<h3 class="section-header">Surface Statistics</h3>', unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns(3, gap="large")
                            
                            with col1:
                                st.metric("Current Price", f"${stats.get('current_price', 0):.2f}")
                                st.metric("ATM IV Short", f"{stats.get('atm_iv_short', 0)*100:.1f}%")
                            
                            with col2:
                                st.metric("ATM IV Long", f"{stats.get('atm_iv_long', 0)*100:.1f}%")
                                st.metric("Term Structure Slope", f"{stats.get('term_structure_slope', 0)*100:.2f}%")
                            
                            with col3:
                                st.metric("Volatility Skew", f"{stats.get('skew', 0)*100:.2f}%")
                                st.metric("Total Options", f"{stats.get('total_options', 0):,}")
                            
                            # Show data table
                            with st.expander("View Raw Data"):
                                st.dataframe(options_with_iv.head(20))
                                
                                if st.button("Export Data"):
                                    filepath = iv_surface.export_iv_data(options_with_iv)
                                    if filepath:
                                        st.success(f"Data exported to {filepath}")
                        else:
                            st.error("No valid options data found after IV calculation")
                    else:
                        st.error(f"No options data found for {st.session_state.iv_ticker_selected}")
                        
                except Exception as e:
                    st.error(f"Error generating IV surface: {e}")
        else:
            st.markdown('<p class="courier-info">Enter a ticker symbol and click \'Generate IV Surface\' to start</p>', unsafe_allow_html=True)

# BACKTESTING RESULTS PAGE
elif st.session_state.current_page == "Backtesting Results":
    st.markdown('<h2 class="page-title">Backtesting Results</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown('<h3 class="section-header">Backtesting Parameters</h3>', unsafe_allow_html=True)
        
        st.markdown('<p class="monospace-label">sample_size</p>', unsafe_allow_html=True)
        backtest_sample_size = st.slider("sample_size", min_value=100, max_value=5000, value=1000, step=100, label_visibility="collapsed")
        
        st.markdown('<p class="monospace-label">risk_free_rate_percent</p>', unsafe_allow_html=True)
        risk_free_rate_bt = st.number_input("risk_free_rate_percent", min_value=0.0, value=5.0, step=0.1, key="bt_rf", label_visibility="collapsed") / 100
        
        if st.button("Run SPX Backtesting", type="primary"):
            st.session_state.backtesting_running = True
    
    with col2:
        if hasattr(st.session_state, 'backtesting_running') and st.session_state.backtesting_running:
            with st.spinner("Running comprehensive backtesting analysis..."):
                try:
                    # Initialize backtester
                    backtester = SPXBacktester(output_folder='output')
                    
                    # Run analysis
                    results = backtester.run_full_analysis(sample_size=backtest_sample_size)
                    
                    # Reset the running flag immediately after completion
                    st.session_state.backtesting_running = False
                    
                    if results is not None and not results.empty:
                        # Store results in session state for display
                        st.session_state.backtest_results = results
                        st.session_state.backtest_completed = True
                        st.success("Backtesting completed successfully!")
                        
                    else:
                        st.error("Backtesting failed. Please check your data and try again.")
                        
                except Exception as e:
                    st.error(f"Error during backtesting: {e}")
                    st.session_state.backtesting_running = False
        
        # Display results if available
        elif hasattr(st.session_state, 'backtest_completed') and st.session_state.backtest_completed and hasattr(st.session_state, 'backtest_results'):
            results = st.session_state.backtest_results
            
            # Display key metrics
            st.markdown('<h3 class="section-header">Performance Summary</h3>', unsafe_allow_html=True)
            
            model_columns = [col for col in results.columns if col.endswith('_price')]
            
            metrics_data = []
            for col in model_columns:
                model_name = col.replace('_price', '')
                error_col = col.replace('_price', '_error')
                
                if error_col in results.columns:
                    mae = results[error_col].abs().mean()
                    rmse = np.sqrt((results[error_col] ** 2).mean())
                    mape = (results[error_col].abs() / results['mid_price']).mean() * 100
                    
                    metrics_data.append({
                        'Model': model_name,
                        'MAE': f"{mae:.4f}",
                        'RMSE': f"{rmse:.4f}",
                        'MAPE': f"{mape:.2f}%"
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
            
            # Model comparison chart
            if len(model_columns) >= 2:
                fig = go.Figure()
                
                for col in model_columns[:4]:  # Show first 4 models
                    model_name = col.replace('_price', '')
                    fig.add_trace(go.Scatter(
                        x=results['mid_price'],
                        y=results[col],
                        mode='markers',
                        name=model_name,
                        opacity=0.6
                    ))
                
                # Perfect prediction line
                min_price = results['mid_price'].min()
                max_price = results['mid_price'].max()
                fig.add_trace(go.Scatter(
                    x=[min_price, max_price],
                    y=[min_price, max_price],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                
                fig.update_layout(
                    title="Model Predictions vs Actual Prices",
                    xaxis_title="Actual Price ($)",
                    yaxis_title="Predicted Price ($)",
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font_color='white',
                    font_family='Satoshi',
                    height=700,
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Dataset information
            st.markdown('<h3 class="section-header">Dataset Information</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4, gap="large")
            
            with col1:
                st.metric("Total Contracts", f"{len(results):,}")
            with col2:
                st.metric("Date Range", f"{results['trade_date'].nunique()} days")
            with col3:
                strike_col = 'strike_price' if 'strike_price' in results.columns else 'strike'
                st.metric("Strike Range", f"${results[strike_col].min():.0f} - ${results[strike_col].max():.0f}")
            with col4:
                st.metric("Avg Time to Expiry", f"{results['days_to_expiry'].mean():.0f} days")
            
            # Download results
            if st.button("Download Results"):
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"spx_backtest_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            # Button to run new backtesting
            if st.button("Run New Backtesting"):
                st.session_state.backtest_completed = False
                st.session_state.backtest_results = None
                st.rerun()
        
        else:
            st.markdown('<p class="courier-info">Configure parameters and click \'Run SPX Backtesting\' to start</p>', unsafe_allow_html=True)

# Custom Footer
st.markdown("""
<div class="custom-footer">
    Built by Aniket Dey
</div>
""", unsafe_allow_html=True)
