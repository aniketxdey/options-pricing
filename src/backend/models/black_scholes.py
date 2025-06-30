import numpy as np
from scipy.stats import norm
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma, option_type='call', dividend_yield=0.0):
        # Handle both single values and arrays
        self.S = np.array(S) if not isinstance(S, np.ndarray) else S
        self.K = np.array(K) if not isinstance(K, np.ndarray) else K
        self.T = np.array(T) if not isinstance(T, np.ndarray) else T
        self.r = r
        self.sigma = np.array(sigma) if not isinstance(sigma, np.ndarray) else sigma
        self.option_type = np.array(option_type) if not isinstance(option_type, np.ndarray) else option_type
        self.q = dividend_yield
        
        # Validate inputs
        self._validate_inputs()
        
        # Calculate d1 and d2
        self._calculate_d_parameters()
    
    def _validate_inputs(self):
        # Basic validation
        if np.any(self.S <= 0):
            raise ValueError("Stock price must be positive")
        if np.any(self.K <= 0):
            raise ValueError("Strike price must be positive") 
        if np.any(self.T <= 0):
            raise ValueError("Time to expiration must be positive")
        if np.any(self.sigma <= 0):
            raise ValueError("Volatility must be positive")
    
    def _calculate_d_parameters(self):
        # Calculate d1 and d2 parameters
        self.d1 = (np.log(self.S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma*np.sqrt(self.T)
    
    def price(self):
        # Calculate option prices using Black-Scholes formula
        call_prices = (self.S * np.exp(-self.q*self.T) * norm.cdf(self.d1) - 
                      self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2))
        
        put_prices = (self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2) - 
                     self.S * np.exp(-self.q*self.T) * norm.cdf(-self.d1))
        
        # Return appropriate prices based on option type
        prices = np.where(
            (self.option_type == 'call') | (self.option_type == 'c') | (self.option_type == 1), 
            call_prices, 
            put_prices
        )
        
        return prices
    
    def delta(self):
        # Calculate delta (price sensitivity to underlying price)
        call_delta = np.exp(-self.q*self.T) * norm.cdf(self.d1)
        put_delta = np.exp(-self.q*self.T) * (norm.cdf(self.d1) - 1)
        
        delta = np.where(
            (self.option_type == 'call') | (self.option_type == 'c') | (self.option_type == 1),
            call_delta,
            put_delta
        )
        
        return delta
    
    def gamma(self):
        # Calculate gamma (delta sensitivity to underlying price)
        gamma = (np.exp(-self.q*self.T) * norm.pdf(self.d1)) / (self.S * self.sigma * np.sqrt(self.T))
        return gamma
    
    def vega(self):
        # Calculate vega (price sensitivity to volatility)
        vega = self.S * np.exp(-self.q*self.T) * np.sqrt(self.T) * norm.pdf(self.d1)
        return vega / 100  # Convert to percentage points
    
    def theta(self):
        # Calculate theta (price sensitivity to time decay)
        term1 = -(self.S * np.exp(-self.q*self.T) * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        term2_call = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        term2_put = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        term3 = self.q * self.S * np.exp(-self.q * self.T)
        
        call_theta = term1 + term2_call + term3 * norm.cdf(self.d1)
        put_theta = term1 + term2_put - term3 * norm.cdf(-self.d1)
        
        theta = np.where(
            (self.option_type == 'call') | (self.option_type == 'c') | (self.option_type == 1),
            call_theta,
            put_theta
        )
        
        return theta / 365  # Convert to daily theta
    
    def rho(self):
        # Calculate rho (price sensitivity to interest rate)
        call_rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        put_rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        
        rho = np.where(
            (self.option_type == 'call') | (self.option_type == 'c') | (self.option_type == 1),
            call_rho,
            put_rho
        )
        
        return rho / 100  # Convert to percentage points
    
    def all_greeks(self):
        # Calculate all Greeks at once
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }
    
    def implied_volatility(self, market_price, max_iterations=100, tolerance=1e-6):
        # Calculate implied volatility using Newton-Raphson method
        if len(self.S) > 1:
            warnings.warn("Implied volatility calculation only supports single option. Using first option.")
        
        # Use first option for IV calculation
        S, K, T, r, option_type = self.S[0], self.K[0], self.T[0], self.r, self.option_type[0]
        
        # Initial guess
        sigma = 0.2
        
        for i in range(max_iterations):
            # Create temporary model with current sigma
            temp_model = BlackScholesModel(S, K, T, r, sigma, option_type, self.q)
            
            # Calculate price and vega
            price = temp_model.price()[0]
            vega = temp_model.vega()[0] * 100  # Convert back from percentage
            
            # Newton-Raphson update
            price_diff = price - market_price
            
            if abs(price_diff) < tolerance:
                return sigma
            
            if abs(vega) < 1e-10:
                break
            
            sigma = sigma - price_diff / vega
            
            # Ensure sigma stays positive
            sigma = max(sigma, 0.001)
        
        return sigma if sigma > 0 else np.nan
    
    def sensitivity_analysis(self, parameter='S', range_pct=0.2, num_points=50):
        # Analyze option price sensitivity to parameter changes
        if len(self.S) > 1:
            warnings.warn("Sensitivity analysis only supports single option. Using first option.")
        
        base_price = self.price()[0]
        base_value = getattr(self, parameter.lower())[0] if hasattr(self, parameter.lower()) else self.r
        
        # Create range around base value
        min_val = base_value * (1 - range_pct)
        max_val = base_value * (1 + range_pct)
        param_values = np.linspace(min_val, max_val, num_points)
        
        prices = []
        deltas = []
        gammas = []
        vegas = []
        thetas = []
        
        for val in param_values:
            # Create temporary model with modified parameter
            kwargs = {
                'S': self.S[0],
                'K': self.K[0], 
                'T': self.T[0],
                'r': self.r,
                'sigma': self.sigma[0],
                'option_type': self.option_type[0],
                'dividend_yield': self.q
            }
            kwargs[parameter.lower()] = val
            
            temp_model = BlackScholesModel(**kwargs)
            prices.append(temp_model.price()[0])
            deltas.append(temp_model.delta()[0])
            gammas.append(temp_model.gamma()[0])
            vegas.append(temp_model.vega()[0])
            thetas.append(temp_model.theta()[0])
        
        return pd.DataFrame({
            parameter.lower(): param_values,
            'price': prices,
            'delta': deltas,
            'gamma': gammas,
            'vega': vegas,
            'theta': thetas
        })
    
    def create_sensitivity_plot(self, parameter='S', range_pct=0.2):
        # Create sensitivity analysis plot
        data = self.sensitivity_analysis(parameter, range_pct)
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=data[parameter.lower()],
            y=data['price'],
            mode='lines',
            name='Option Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add current price point
        current_val = getattr(self, parameter.lower())[0] if hasattr(self, parameter.lower()) else self.r
        current_price = self.price()[0]
        
        fig.add_trace(go.Scatter(
            x=[current_val],
            y=[current_price],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Current Value'
        ))
        
        fig.update_layout(
            title=f'Option Price Sensitivity to {parameter.upper()}',
            xaxis_title=parameter.upper(),
            yaxis_title='Option Price',
            width=800,
            height=500
        )
        
        return fig
    
    def create_greeks_plot(self, parameter='S', range_pct=0.2):
        # Create Greeks sensitivity plot
        data = self.sensitivity_analysis(parameter, range_pct)
        
        fig = go.Figure()
        
        # Add each Greek
        greeks = ['delta', 'gamma', 'vega', 'theta']
        colors = ['blue', 'red', 'green', 'orange']
        
        for greek, color in zip(greeks, colors):
            fig.add_trace(go.Scatter(
                x=data[parameter.lower()],
                y=data[greek],
                mode='lines',
                name=greek.title(),
                line=dict(color=color)
            ))
        
        fig.update_layout(
            title=f'Greeks Sensitivity to {parameter.upper()}',
            xaxis_title=parameter.upper(),
            yaxis_title='Greek Value',
            width=800,
            height=500
        )
        
        return fig
    
    def parity_check(self):
        # Verify put-call parity
        if len(self.S) > 1:
            warnings.warn("Parity check only supports single option. Using first option.")
        
        # Calculate call and put prices
        call_model = BlackScholesModel(
            self.S[0], self.K[0], self.T[0], self.r, self.sigma[0], 'call', self.q
        )
        put_model = BlackScholesModel(
            self.S[0], self.K[0], self.T[0], self.r, self.sigma[0], 'put', self.q
        )
        
        call_price = call_model.price()[0]
        put_price = put_model.price()[0]
        
        # Put-call parity: C - P = S*e^(-q*T) - K*e^(-r*T)
        left_side = call_price - put_price
        right_side = self.S[0] * np.exp(-self.q * self.T[0]) - self.K[0] * np.exp(-self.r * self.T[0])
        
        parity_diff = abs(left_side - right_side)
        
        return {
            'call_price': call_price,
            'put_price': put_price,
            'left_side': left_side,
            'right_side': right_side,
            'difference': parity_diff,
            'parity_holds': parity_diff < 1e-10
        }

# Standalone functions for backward compatibility
def blackScholes(S, K, r, T, sigma, type="c"):
    # Original function signature maintained for compatibility
    try:
        model = BlackScholesModel(S, K, T, r, sigma, type)
        price = model.price()
        return price[0] if isinstance(price, np.ndarray) and len(price) == 1 else price
    except Exception:
        return None

def optionDelta(S, K, r, T, sigma, type="c"):
    # Calculate option delta
    try:
        model = BlackScholesModel(S, K, T, r, sigma, type)
        delta = model.delta()
        return delta[0] if isinstance(delta, np.ndarray) and len(delta) == 1 else delta
    except Exception:
        return None

def optionGamma(S, K, r, T, sigma):
    # Calculate option gamma
    try:
        model = BlackScholesModel(S, K, T, r, sigma, 'call')  # Gamma same for calls/puts
        gamma = model.gamma()
        return gamma[0] if isinstance(gamma, np.ndarray) and len(gamma) == 1 else gamma
    except Exception:
        return None

def optionTheta(S, K, r, T, sigma, type="c"):
    # Calculate option theta
    try:
        model = BlackScholesModel(S, K, T, r, sigma, type)
        theta = model.theta()
        return theta[0] if isinstance(theta, np.ndarray) and len(theta) == 1 else theta
    except Exception:
        return None

def optionVega(S, K, r, T, sigma):
    # Calculate option vega
    try:
        model = BlackScholesModel(S, K, T, r, sigma, 'call')  # Vega same for calls/puts
        vega = model.vega()
        return vega[0] if isinstance(vega, np.ndarray) and len(vega) == 1 else vega
    except Exception:
        return None

def optionRho(S, K, r, T, sigma, type="c"):
    # Calculate option rho
    try:
        model = BlackScholesModel(S, K, T, r, sigma, type)
        rho = model.rho()
        return rho[0] if isinstance(rho, np.ndarray) and len(rho) == 1 else rho
    except Exception:
        return None