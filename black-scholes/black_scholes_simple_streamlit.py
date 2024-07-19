import streamlit as st
import math
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def main():
    st.title("Black-Scholes Option Pricing Calculator")

    st.sidebar.header("Input Parameters")
    S = st.sidebar.number_input("Current Stock Price (S)", value=42.0, min_value=0.0, step=1.0)
    K = st.sidebar.number_input("Strike Price (K)", value=40.0, min_value=0.0, step=1.0)
    T = st.sidebar.number_input("Time to Expiration (T) in years", value=0.5, min_value=0.0, step=0.1)
    r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.1, min_value=0.0, step=0.01)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, min_value=0.0, step=0.01)
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

    if st.sidebar.button("Calculate"):
        price = black_scholes(S, K, T, r, sigma, option_type)
        st.write(f"The {option_type} option price is: ${price:.2f}")
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        st.write(f"The value of d1 is: {d1:.4f}")
        st.write(f"The value of d2 is: {d2:.4f}")

if __name__ == "__main__":
    main()




