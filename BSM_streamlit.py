import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px

def blackScholes(S, K, r, T, sigma, type="c"):
    "Calculate Black Scholes option price for a call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if type == "c":
            price = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
        elif type == "p":
            price = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)

        return price
    except:  
        st.sidebar.error("Please confirm all option parameters!")


def optionDelta (S, K, r, T, sigma, type="c"):
    "Calculates option delta"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        if type == "c":
            delta = norm.cdf(d1, 0, 1)
        elif type == "p":
            delta = -norm.cdf(-d1, 0, 1)

        return delta
    except:
        st.sidebar.error("Please confirm all option parameters!")

def optionGamma (S, K, r, T, sigma):
    "Calculates option gamma"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        gamma = norm.pdf(d1, 0, 1)/ (S * sigma * np.sqrt(T))
        return gamma
    except:
        st.sidebar.error("Please confirm all option parameters!")

def optionTheta(S, K, r, T, sigma, type="c"):
    "Calculates option theta"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        if type == "c":
            theta = - ((S * norm.pdf(d1, 0, 1) * sigma) / (2 * np.sqrt(T))) - r * K * np.exp(-r*T) * norm.cdf(d2, 0, 1)

        elif type == "p":
            theta = - ((S * norm.pdf(d1, 0, 1) * sigma) / (2 * np.sqrt(T))) + r * K * np.exp(-r*T) * norm.cdf(-d2, 0, 1)
        return theta/365
    except:
        st.sidebar.error("Please confirm all option parameters!")

def optionVega (S, K, r, T, sigma):
    "Calculates option vega"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        vega = S * np.sqrt(T) * norm.pdf(d1, 0, 1) * 0.01
        return vega
    except:
        st.sidebar.error("Please confirm all option parameters!")

def optionRho(S, K, r, T, sigma, type="c"):
    "Calculates option rho"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        if type == "c":
            rho = 0.01 * K * T * np.exp(-r*T) * norm.cdf(d2, 0, 1)
        elif type == "p":
            rho = 0.01 * -K * T * np.exp(-r*T) * norm.cdf(-d2, 0, 1)
        return rho
    except:
        st.sidebar.error("Please confirm all option parameters!")

# Black-Scholes Option Pricing Model
class BlackScholes:
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity (in years)
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        self.option_type = option_type.lower()
        
        # Calculate d1 and d2
        self.d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        self.d2 = self.d1 - sigma*np.sqrt(T)

    def price(self):
        if self.option_type == 'call':
            return (self.S * norm.cdf(self.d1) - 
                   self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2))
        else:
            return (self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2) - 
                   self.S * norm.cdf(-self.d1))

    def delta(self):
        if self.option_type == 'call':
            return norm.cdf(self.d1)
        else:
            return norm.cdf(self.d1) - 1

    def gamma(self):
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * np.sqrt(self.T) * norm.pdf(self.d1)

    def theta(self):
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        if self.option_type == 'call':
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            return term1 + term2
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            return term1 + term2

    def rho(self):
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)

def create_heatmap(x_values, y_values, z_values, x_label, y_label, title):
    fig = go.Figure(data=go.Heatmap(
        x=x_values,
        y=y_values,
        z=z_values,
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=800,
        height=600
    )
    
    return fig

st.set_page_config(page_title="Black-Scholes Option Calculator", layout="wide")
st.title("Black-Scholes Option Calculator")

# Input parameters
col1, col2, col3 = st.columns(3)

with col1:
    S = st.number_input("Stock Price", min_value=0.01, value=100.0)
    K = st.number_input("Strike Price", min_value=0.01, value=100.0)
    
with col2:
    T = st.number_input("Time to Maturity (years)", min_value=0.01, value=1.0)
    r = st.number_input("Risk-free Rate (%)", min_value=0.0, value=5.0) / 100
    
with col3:
    sigma = st.number_input("Volatility (%)", min_value=0.01, value=20.0) / 100
    option_type = st.selectbox("Option Type", ["Call", "Put"])

# Calculate option price and Greeks
bs = BlackScholes(S, K, T, r, sigma, option_type.lower())
price = bs.price()
delta = bs.delta()
gamma = bs.gamma()
vega = bs.vega()
theta = bs.theta()
rho = bs.rho()

# Display results
st.header("Option Metrics")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Price", f"${price:.2f}")
col2.metric("Delta", f"{delta:.4f}")
col3.metric("Gamma", f"{gamma:.4f}")
col4.metric("Vega", f"{vega:.4f}")
col5.metric("Theta", f"{theta:.4f}")
col6.metric("Rho", f"{rho:.4f}")

# Visualization section
st.header("Visualizations")
viz_type = st.selectbox(
    "Select Visualization",
    ["Price Surface", "Delta Surface", "Volatility Surface"]
)

# Generate surface data
stock_prices = np.linspace(S*0.5, S*1.5, 50)
times = np.linspace(T*0.1, T, 50)
S_mesh, T_mesh = np.meshgrid(stock_prices, times)
Z = np.zeros_like(S_mesh)

for i in range(len(times)):
    for j in range(len(stock_prices)):
        bs_temp = BlackScholes(stock_prices[j], K, times[i], r, sigma, option_type.lower())
        if viz_type == "Price Surface":
            Z[i,j] = bs_temp.price()
        elif viz_type == "Delta Surface":
            Z[i,j] = bs_temp.delta()
        else:  # Volatility Surface
            Z[i,j] = bs_temp.vega()

# Create and display 3D surface plot
fig = go.Figure(data=[go.Surface(x=S_mesh, y=T_mesh, z=Z)])
fig.update_layout(
    title=f"{viz_type}",
    scene=dict(
        xaxis_title="Stock Price",
        yaxis_title="Time to Maturity",
        zaxis_title=viz_type.split()[0]
    ),
    width=800,
    height=800
)
st.plotly_chart(fig)

# Create heatmap
st.header("Sensitivity Heatmap")
sensitivity = st.selectbox(
    "Select Sensitivity Analysis",
    ["Delta", "Gamma", "Vega"]
)

volatilities = np.linspace(sigma*0.5, sigma*1.5, 50)
prices = np.linspace(S*0.5, S*1.5, 50)
Z_heatmap = np.zeros((len(volatilities), len(prices)))

for i, vol in enumerate(volatilities):
    for j, price in enumerate(prices):
        bs_temp = BlackScholes(price, K, T, r, vol, option_type.lower())
        if sensitivity == "Delta":
            Z_heatmap[i,j] = bs_temp.delta()
        elif sensitivity == "Gamma":
            Z_heatmap[i,j] = bs_temp.gamma()
        else:
            Z_heatmap[i,j] = bs_temp.vega()

heatmap = create_heatmap(
    prices,
    volatilities,
    Z_heatmap,
    "Stock Price",
    "Volatility",
    f"{sensitivity} Heatmap"
)
st.plotly_chart(heatmap)

st.markdown("<h2 align='center'>Black-Scholes Option Price Calculator</h2>", unsafe_allow_html=True)
st.markdown("<h5 align='center'>Made by Tiago Moreira</h5>", unsafe_allow_html=True)
st.header("")
st.markdown("<h6>See project's description and assumptions here: <a href='https://github.com/TFSM00/Black-Scholes-Calculator'>https://github.com/TFSM00/Black-Scholes-Calculator</a></h6>", unsafe_allow_html=True)
st.markdown("<h6>See all my other projects here: <a href='https://github.com/TFSM00'>https://github.com/TFSM00</a></h6>", unsafe_allow_html=True)
st.header("")
st.markdown("<h3 align='center'>Option Prices and Greeks</h3>", unsafe_allow_html=True)
st.header("")
col1, col2, col3, col4, col5 = st.columns(5)
col2.metric("Call Price", str(round(blackScholes(S, K, r, T, sigma,type="c"), 3)))
col4.metric("Put Price", str(round(blackScholes(S, K, r, T, sigma,type="p"), 3)))

bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
bcol1.metric("Delta", str(round(blackScholes(S, K, r, T, sigma,type="c"), 3)))
bcol2.metric("Gamma", str(round(optionGamma(S, K, r, T, sigma), 3)))
bcol3.metric("Theta", str(round(optionTheta(S, K, r, T, sigma,type="c"), 3)))
bcol4.metric("Vega", str(round(optionVega(S, K, r, T, sigma), 3)))
bcol5.metric("Rho", str(round(optionRho(S, K, r, T, sigma,type="c"), 3)))

st.header("")
st.markdown("<h3 align='center'>Visualization of the Greeks</h3>", unsafe_allow_html=True)
st.header("")

