"""Entry point for Streamlit Cloud deployment.

Streamlit Cloud expects a top-level entry file. The actual app lives at
src/streamlit.py; this bootstrap just wires up sys.path so the backend
package imports resolve, then runs the app file.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "backend"))

with open(os.path.join(ROOT, "src", "streamlit.py")) as f:
    exec(f.read(), {"__name__": "__main__", "__file__": os.path.join(ROOT, "src", "streamlit.py")})
