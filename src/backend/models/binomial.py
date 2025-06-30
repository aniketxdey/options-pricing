import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings

class BinomialTreeModel:
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
    
    def european_option_price(self, N=100, return_tree=False):
        # European option pricing using binomial tree
        prices = np.zeros_like(self.S)
        trees = [] if return_tree else None
        
        # Handle vectorized inputs
        for i in range(len(self.S)):
            price, tree = self._calculate_single_option(
                self.S[i], self.K[i], self.T[i], self.sigma[i], 
                self.option_type[i], N, return_tree
            )
            prices[i] = price
            if return_tree:
                trees.append(tree)
        
        if return_tree:
            return prices, trees
        return prices
    
    def _calculate_single_option(self, S, K, T, sigma, option_type, N, return_tree):
        # Calculate single option price
        dt = T / N
        
        # Up and down factors
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability
        q = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        asset_prices = np.array([S * (u**j) * (d**(N-j)) for j in range(N+1)])
        
        # Initialize option values at maturity
        if option_type.lower() in ['call', 'c']:
            option_values = np.maximum(0, asset_prices - K)
        else:
            option_values = np.maximum(0, K - asset_prices)
        
        # Store tree if requested
        tree = {'asset_prices': [], 'option_values': []} if return_tree else None
        if return_tree:
            tree['asset_prices'].append(asset_prices.copy())
            tree['option_values'].append(option_values.copy())
        
        # Backward induction
        for step in range(N-1, -1, -1):
            # Asset prices at this step
            asset_prices = np.array([S * (u**j) * (d**(step-j)) for j in range(step+1)])
            
            # Option values at this step
            option_values = np.exp(-(self.r - self.q) * dt) * (
                q * option_values[1:step+2] + (1-q) * option_values[0:step+1]
            )
            
            if return_tree:
                tree['asset_prices'].insert(0, asset_prices.copy())
                tree['option_values'].insert(0, option_values.copy())
        
        return option_values[0], tree
    
    def american_option_price(self, N=100, return_tree=False):
        # American option pricing using binomial tree
        prices = np.zeros_like(self.S)
        trees = [] if return_tree else None
        
        for i in range(len(self.S)):
            price, tree = self._calculate_american_option(
                self.S[i], self.K[i], self.T[i], self.sigma[i], 
                self.option_type[i], N, return_tree
            )
            prices[i] = price
            if return_tree:
                trees.append(tree)
        
        if return_tree:
            return prices, trees
        return prices
    
    def _calculate_american_option(self, S, K, T, sigma, option_type, N, return_tree):
        # Calculate American option price with early exercise
        dt = T / N
        
        # Up and down factors
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability
        q = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        
        # Initialize asset price tree
        asset_tree = np.zeros((N+1, N+1))
        option_tree = np.zeros((N+1, N+1))
        
        # Fill asset price tree
        for i in range(N+1):
            for j in range(i+1):
                asset_tree[i, j] = S * (u**j) * (d**(i-j))
        
        # Initialize option values at maturity
        for j in range(N+1):
            if option_type.lower() in ['call', 'c']:
                option_tree[N, j] = max(0, asset_tree[N, j] - K)
            else:
                option_tree[N, j] = max(0, K - asset_tree[N, j])
        
        # Backward induction with early exercise check
        for i in range(N-1, -1, -1):
            for j in range(i+1):
                # European value
                european_value = np.exp(-(self.r - self.q) * dt) * (
                    q * option_tree[i+1, j+1] + (1-q) * option_tree[i+1, j]
                )
                
                # Intrinsic value (early exercise)
                if option_type.lower() in ['call', 'c']:
                    intrinsic_value = max(0, asset_tree[i, j] - K)
                else:
                    intrinsic_value = max(0, K - asset_tree[i, j])
                
                # American value is max of European and intrinsic
                option_tree[i, j] = max(european_value, intrinsic_value)
        
        tree = {'asset_tree': asset_tree, 'option_tree': option_tree} if return_tree else None
        return option_tree[0, 0], tree
    
    def calculate_greeks(self, N=100, bump_size=0.01):
        # Calculate Greeks using finite differences
        base_price = self.european_option_price(N)
        
        # Delta (dV/dS)
        self.S += bump_size
        price_up = self.european_option_price(N)
        self.S -= 2 * bump_size
        price_down = self.european_option_price(N)
        self.S += bump_size  # Reset to original
        delta = (price_up - price_down) / (2 * bump_size)
        
        # Gamma (d²V/dS²)
        gamma = (price_up - 2 * base_price + price_down) / (bump_size ** 2)
        
        # Vega (dV/dσ)
        self.sigma += bump_size
        vega_up = self.european_option_price(N)
        self.sigma -= 2 * bump_size
        vega_down = self.european_option_price(N)
        self.sigma += bump_size  # Reset to original
        vega = (vega_up - vega_down) / (2 * bump_size)
        
        # Theta (dV/dT)
        self.T -= bump_size / 365  # One day
        theta_down = self.european_option_price(N)
        self.T += bump_size / 365  # Reset to original
        theta = (theta_down - base_price) / (bump_size / 365)
        
        # Rho (dV/dr)
        original_r = self.r
        self.r += bump_size
        rho_up = self.european_option_price(N)
        self.r -= 2 * bump_size
        rho_down = self.european_option_price(N)
        self.r = original_r  # Reset to original
        rho = (rho_up - rho_down) / (2 * bump_size)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def convergence_analysis(self, max_steps=200, step_increment=10):
        # Analyze convergence as number of steps increases
        steps = range(step_increment, max_steps + 1, step_increment)
        prices = []
        
        for N in steps:
            price = self.european_option_price(N)
            prices.append(price[0] if isinstance(price, np.ndarray) else price)
        
        return pd.DataFrame({
            'steps': steps,
            'price': prices
        })
    
    def visualize_tree(self, N=5, figsize=(12, 8)):
        # Visualize binomial tree structure
        if len(self.S) > 1:
            warnings.warn("Visualization only supports single option. Using first option.")
        
        price, tree = self._calculate_single_option(
            self.S[0], self.K[0], self.T[0], self.sigma[0], 
            self.option_type[0], N, return_tree=True
        )
        
        # Create visualization using plotly
        fig = go.Figure()
        
        # Plot tree structure
        for i, (asset_prices, option_values) in enumerate(zip(tree['asset_prices'], tree['option_values'])):
            x_coords = [i] * len(asset_prices)
            
            # Asset prices
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=asset_prices,
                mode='markers+text',
                text=[f'S={price:.2f}' for price in asset_prices],
                textposition='top center',
                marker=dict(size=8, color='blue'),
                name=f'Asset Prices Step {i}' if i == 0 else '',
                showlegend=i == 0,
                legendgroup='asset'
            ))
            
            # Option values
            fig.add_trace(go.Scatter(
                x=[x + 0.2 for x in x_coords],
                y=asset_prices,
                mode='text',
                text=[f'V={value:.2f}' for value in option_values],
                textposition='top center',
                showlegend=False
            ))
        
        fig.update_layout(
            title=f'Binomial Tree Visualization - {self.option_type[0].title()} Option (N={N})',
            xaxis_title='Time Step',
            yaxis_title='Asset Price',
            width=800,
            height=600
        )
        
        return fig
    
    def create_convergence_plot(self, max_steps=200):
        # Plot convergence analysis
        convergence_data = self.convergence_analysis(max_steps)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=convergence_data['steps'],
            y=convergence_data['price'],
            mode='lines+markers',
            name='Binomial Price',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title='Binomial Tree Convergence Analysis',
            xaxis_title='Number of Steps',
            yaxis_title='Option Price',
            width=800,
            height=500
        )
        
        return fig

# Standalone functions for backward compatibility
def binomial_call_option_price(S0, sigma, r, K, T, N, visualize=False):
    # Original function signature maintained for compatibility
    model = BinomialTreeModel(S0, K, T, r, sigma, 'call')
    price = model.european_option_price(N)
    
    if visualize:
        fig = model.visualize_tree(N)
        fig.show()
    
    return price[0] if isinstance(price, np.ndarray) else price

def binomial_put_option_price(S0, sigma, r, K, T, N, visualize=False):
    # Put option pricing
    model = BinomialTreeModel(S0, K, T, r, sigma, 'put')
    price = model.european_option_price(N)
    
    if visualize:
        fig = model.visualize_tree(N)
        fig.show()
    
    return price[0] if isinstance(price, np.ndarray) else price