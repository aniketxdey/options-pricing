"""
Main entry point for Streamlit Cloud deployment
"""

import sys
import os
import importlib.util

# Add src/backend to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, 'src', 'backend')
sys.path.insert(0, backend_path)

# Import the main streamlit application
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Import backend modules
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

# Load and execute the streamlit app
spec = importlib.util.spec_from_file_location("streamlit_app", os.path.join(current_dir, 'src', 'frontend', 'streamlit.py'))
streamlit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(streamlit_module) 