"""
Main entry point for Streamlit Cloud deployment
This file is automatically detected by Streamlit Cloud
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'backend'))

# Import and run the main streamlit application
from src.frontend.streamlit import * 