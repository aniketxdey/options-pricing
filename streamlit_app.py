"""
Main entry point for Streamlit Cloud deployment
"""

import sys
import os

# Add src/backend to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, 'src', 'backend')
sys.path.insert(0, backend_path)

# Execute the main streamlit application
exec(open(os.path.join(current_dir, 'src', 'frontend', 'streamlit.py')).read()) 