import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
from scipy.stats import norm

class MonteCarloOptionPricing:
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
    
    def standard_monte_carlo(self, num_simulations=100000, num_steps=252, antithetic=True, control_variate=True):
        # Standard GBM Monte Carlo simulation
        prices = np.zeros_like(self.S)
        confidence_intervals = []
        
        for i in range(len(self.S)):
            price, ci = self._simulate_gbm_option(
                self.S[i], self.K[i], self.T[i], self.sigma[i], 
                self.option_type[i], num_simulations, num_steps, 
                antithetic, control_variate
            )
            prices[i] = price
            confidence_intervals.append(ci)
        
        return prices, confidence_intervals
    
    def _simulate_gbm_option(self, S, K, T, sigma, option_type, num_sims, num_steps, antithetic, control_variate):
        # Simulate single option using GBM
        dt = T / num_steps
        
        # Generate random numbers
        if antithetic:
            # Use antithetic variates to reduce variance
            num_base_sims = num_sims // 2
            Z = np.random.standard_normal((num_base_sims, num_steps))
            Z_anti = np.vstack([Z, -Z])
        else:
            Z = np.random.standard_normal((num_sims, num_steps))
            Z_anti = Z
        
        # Initialize paths
        paths = np.zeros((len(Z_anti), num_steps + 1))
        paths[:, 0] = S
        
        # Generate stock price paths using GBM
        for t in range(num_steps):
            paths[:, t+1] = paths[:, t] * np.exp(
                (self.r - self.q - 0.5 * sigma**2) * dt + 
                sigma * np.sqrt(dt) * Z_anti[:, t]
            )
        
        # Calculate payoffs
        final_prices = paths[:, -1]
        if option_type.lower() in ['call', 'c']:
            payoffs = np.maximum(0, final_prices - K)
        else:
            payoffs = np.maximum(0, K - final_prices)
        
        # Apply control variate if requested
        if control_variate:
            # Use delta-hedged portfolio as control variate
            from .black_scholes import BlackScholesModel
            bs_model = BlackScholesModel(S, K, T, self.r, sigma, option_type, self.q)
            analytical_price = bs_model.price()[0]
            
            # Calculate control variate adjustment
            asset_returns = final_prices - S * np.exp((self.r - self.q) * T)
            cov_payoff_asset = np.cov(payoffs, asset_returns)[0, 1]
            var_asset = np.var(asset_returns)
            
            if var_asset > 0:
                beta = cov_payoff_asset / var_asset
                controlled_payoffs = payoffs - beta * asset_returns
                payoffs = controlled_payoffs
        
        # Discount and calculate price
        discounted_payoffs = np.exp(-self.r * T) * payoffs
        option_price = np.mean(discounted_payoffs)
        
        # Calculate confidence interval
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        confidence_interval = (
            option_price - 1.96 * std_error,
            option_price + 1.96 * std_error
        )
        
        return option_price, confidence_interval
    
    def path_dependent_option(self, barrier_type='up_and_out', barrier_level=None, 
                            num_simulations=100000, num_steps=252):
        # Price path-dependent options (barriers, Asian, etc.)
        if barrier_level is None:
            barrier_level = self.S[0] * 1.2 if 'up' in barrier_type else self.S[0] * 0.8
        
        prices = []
        
        for i in range(len(self.S)):
            price = self._simulate_path_dependent(
                self.S[i], self.K[i], self.T[i], self.sigma[i], 
                self.option_type[i], barrier_type, barrier_level,
                num_simulations, num_steps
            )
            prices.append(price)
        
        return np.array(prices)
    
    def _simulate_path_dependent(self, S, K, T, sigma, option_type, barrier_type, 
                               barrier_level, num_sims, num_steps):
        # Simulate path-dependent option
        dt = T / num_steps
        Z = np.random.standard_normal((num_sims, num_steps))
        
        # Initialize paths
        paths = np.zeros((num_sims, num_steps + 1))
        paths[:, 0] = S
        
        # Generate stock price paths
        for t in range(num_steps):
            paths[:, t+1] = paths[:, t] * np.exp(
                (self.r - self.q - 0.5 * sigma**2) * dt + 
                sigma * np.sqrt(dt) * Z[:, t]
            )
        
        # Apply path-dependent conditions
        if 'barrier' in barrier_type.lower():
            payoffs = self._calculate_barrier_payoffs(
                paths, K, option_type, barrier_type, barrier_level
            )
        elif 'asian' in barrier_type.lower():
            payoffs = self._calculate_asian_payoffs(paths, K, option_type)
        else:
            # Standard European payoffs
            final_prices = paths[:, -1]
            if option_type.lower() in ['call', 'c']:
                payoffs = np.maximum(0, final_prices - K)
            else:
                payoffs = np.maximum(0, K - final_prices)
        
        # Discount and return price
        return np.exp(-self.r * T) * np.mean(payoffs)
    
    def _calculate_barrier_payoffs(self, paths, K, option_type, barrier_type, barrier_level):
        # Calculate barrier option payoffs
        final_prices = paths[:, -1]
        
        # Check barrier conditions
        if 'up_and_out' in barrier_type:
            knocked_out = np.any(paths >= barrier_level, axis=1)
        elif 'down_and_out' in barrier_type:
            knocked_out = np.any(paths <= barrier_level, axis=1)
        elif 'up_and_in' in barrier_type:
            knocked_out = ~np.any(paths >= barrier_level, axis=1)
        elif 'down_and_in' in barrier_type:
            knocked_out = ~np.any(paths <= barrier_level, axis=1)
        else:
            knocked_out = np.zeros(len(paths), dtype=bool)
        
        # Calculate payoffs
        if option_type.lower() in ['call', 'c']:
            payoffs = np.maximum(0, final_prices - K)
        else:
            payoffs = np.maximum(0, K - final_prices)
        
        # Apply barrier condition
        payoffs[knocked_out] = 0
        
        return payoffs
    
    def _calculate_asian_payoffs(self, paths, K, option_type):
        # Calculate Asian option payoffs (arithmetic average)
        avg_prices = np.mean(paths, axis=1)
        
        if option_type.lower() in ['call', 'c']:
            payoffs = np.maximum(0, avg_prices - K)
        else:
            payoffs = np.maximum(0, K - avg_prices)
        
        return payoffs
    
    def calculate_greeks(self, num_simulations=100000, bump_size=0.01):
        # Calculate Greeks using finite differences
        base_prices, _ = self.standard_monte_carlo(num_simulations)
        
        # Delta (dV/dS)
        self.S += bump_size
        prices_up, _ = self.standard_monte_carlo(num_simulations)
        self.S -= 2 * bump_size
        prices_down, _ = self.standard_monte_carlo(num_simulations)
        self.S += bump_size  # Reset
        delta = (prices_up - prices_down) / (2 * bump_size)
        
        # Gamma (d²V/dS²)
        gamma = (prices_up - 2 * base_prices + prices_down) / (bump_size ** 2)
        
        # Vega (dV/dσ)
        self.sigma += bump_size
        vega_up, _ = self.standard_monte_carlo(num_simulations)
        self.sigma -= 2 * bump_size
        vega_down, _ = self.standard_monte_carlo(num_simulations)
        self.sigma += bump_size  # Reset
        vega = (vega_up - vega_down) / (2 * bump_size)
        
        # Theta (dV/dT)
        self.T -= bump_size / 365
        theta_down, _ = self.standard_monte_carlo(num_simulations)
        self.T += bump_size / 365  # Reset
        theta = (theta_down - base_prices) / (bump_size / 365)
        
        # Rho (dV/dr)
        original_r = self.r
        self.r += bump_size
        rho_up, _ = self.standard_monte_carlo(num_simulations)
        self.r -= 2 * bump_size
        rho_down, _ = self.standard_monte_carlo(num_simulations)
        self.r = original_r  # Reset
        rho = (rho_up - rho_down) / (2 * bump_size)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def convergence_analysis(self, max_simulations=1000000, step_size=10000):
        # Analyze convergence as number of simulations increases
        if len(self.S) > 1:
            warnings.warn("Convergence analysis only supports single option. Using first option.")
        
        sim_counts = range(step_size, max_simulations + 1, step_size)
        prices = []
        std_errors = []
        
        for num_sims in sim_counts:
            price, ci = self._simulate_gbm_option(
                self.S[0], self.K[0], self.T[0], self.sigma[0], 
                self.option_type[0], num_sims, 252, True, True
            )
            prices.append(price)
            std_errors.append((ci[1] - ci[0]) / (2 * 1.96))
        
        return pd.DataFrame({
            'simulations': sim_counts,
            'price': prices,
            'std_error': std_errors
        })
    
    def create_convergence_plot(self, max_simulations=100000):
        # Plot convergence analysis
        convergence_data = self.convergence_analysis(max_simulations)
        
        fig = go.Figure()
        
        # Price convergence
        fig.add_trace(go.Scatter(
            x=convergence_data['simulations'],
            y=convergence_data['price'],
            mode='lines',
            name='MC Price',
            line=dict(color='blue')
        ))
        
        # Confidence bands
        upper_bound = convergence_data['price'] + 1.96 * convergence_data['std_error']
        lower_bound = convergence_data['price'] - 1.96 * convergence_data['std_error']
        
        fig.add_trace(go.Scatter(
            x=convergence_data['simulations'],
            y=upper_bound,
            mode='lines',
            line=dict(color='lightblue', width=1),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=convergence_data['simulations'],
            y=lower_bound,
            mode='lines',
            line=dict(color='lightblue', width=1),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.3)',
            name='95% CI'
        ))
        
        fig.update_layout(
            title='Monte Carlo Price Convergence',
            xaxis_title='Number of Simulations',
            yaxis_title='Option Price',
            width=800,
            height=500
        )
        
        return fig

class HestonMonteCarloModel:
    def __init__(self, S, K, T, r, V0, kappa, theta, sigma_v, rho, option_type='call', dividend_yield=0.0):
        # Heston stochastic volatility model parameters
        self.S = S          # Initial stock price
        self.K = K          # Strike price
        self.T = T          # Time to maturity
        self.r = r          # Risk-free rate
        self.V0 = V0        # Initial variance
        self.kappa = kappa  # Mean reversion rate
        self.theta = theta  # Long-run variance
        self.sigma_v = sigma_v  # Vol of vol
        self.rho = rho      # Correlation between price and volatility
        self.option_type = option_type.lower()
        self.q = dividend_yield
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        # Validate Heston model parameters
        if self.S <= 0:
            raise ValueError("Stock price must be positive")
        if self.K <= 0:
            raise ValueError("Strike price must be positive")
        if self.T <= 0:
            raise ValueError("Time to expiration must be positive")
        if self.V0 <= 0:
            raise ValueError("Initial variance must be positive")
        if self.kappa <= 0:
            raise ValueError("Mean reversion rate must be positive")
        if self.theta <= 0:
            raise ValueError("Long-run variance must be positive")
        if self.sigma_v <= 0:
            raise ValueError("Vol of vol must be positive")
        if abs(self.rho) > 1:
            raise ValueError("Correlation must be between -1 and 1")
    
    def price_option(self, num_simulations=100000, num_steps=252, scheme='euler'):
        # Price option using Heston model
        if scheme.lower() == 'euler':
            return self._euler_scheme(num_simulations, num_steps)
        elif scheme.lower() == 'milstein':
            return self._milstein_scheme(num_simulations, num_steps)
        else:
            raise ValueError("Scheme must be 'euler' or 'milstein'")
    
    def _euler_scheme(self, num_sims, num_steps):
        # Euler discretization scheme
        dt = self.T / num_steps
        
        # Generate correlated random numbers
        Z1 = np.random.standard_normal((num_sims, num_steps))
        Z2 = np.random.standard_normal((num_sims, num_steps))
        W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        
        # Initialize paths
        S_paths = np.zeros((num_sims, num_steps + 1))
        V_paths = np.zeros((num_sims, num_steps + 1))
        S_paths[:, 0] = self.S
        V_paths[:, 0] = self.V0
        
        # Simulate paths
        for t in range(num_steps):
            # Ensure variance stays positive (reflection at zero)
            V_paths[:, t] = np.maximum(V_paths[:, t], 0)
            
            # Update variance
            dV = (self.kappa * (self.theta - V_paths[:, t]) * dt + 
                  self.sigma_v * np.sqrt(V_paths[:, t]) * np.sqrt(dt) * W2[:, t])
            V_paths[:, t+1] = V_paths[:, t] + dV
            
            # Update stock price
            dS = ((self.r - self.q) * S_paths[:, t] * dt + 
                  np.sqrt(V_paths[:, t]) * S_paths[:, t] * np.sqrt(dt) * Z1[:, t])
            S_paths[:, t+1] = S_paths[:, t] + dS
        
        # Calculate payoffs
        final_prices = S_paths[:, -1]
        if self.option_type in ['call', 'c']:
            payoffs = np.maximum(0, final_prices - self.K)
        else:
            payoffs = np.maximum(0, self.K - final_prices)
        
        # Discount and return price
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.std(payoffs) * np.exp(-self.r * self.T) / np.sqrt(num_sims)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval': (
                option_price - 1.96 * std_error,
                option_price + 1.96 * std_error
            ),
            'paths': {'S': S_paths, 'V': V_paths}
        }
    
    def _milstein_scheme(self, num_sims, num_steps):
        # Milstein discretization scheme (more accurate for Heston)
        dt = self.T / num_steps
        
        # Generate correlated random numbers
        Z1 = np.random.standard_normal((num_sims, num_steps))
        Z2 = np.random.standard_normal((num_sims, num_steps))
        W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        
        # Initialize paths
        S_paths = np.zeros((num_sims, num_steps + 1))
        V_paths = np.zeros((num_sims, num_steps + 1))
        S_paths[:, 0] = self.S
        V_paths[:, 0] = self.V0
        
        # Simulate paths with Milstein correction
        for t in range(num_steps):
            V_paths[:, t] = np.maximum(V_paths[:, t], 0)
            sqrt_V = np.sqrt(V_paths[:, t])
            sqrt_dt = np.sqrt(dt)
            
            # Milstein scheme for variance
            dW2 = sqrt_dt * W2[:, t]
            V_paths[:, t+1] = (V_paths[:, t] + 
                              self.kappa * (self.theta - V_paths[:, t]) * dt +
                              self.sigma_v * sqrt_V * dW2 +
                              0.25 * self.sigma_v**2 * (dW2**2 - dt))
            
            # Update stock price
            dW1 = sqrt_dt * Z1[:, t]
            S_paths[:, t+1] = (S_paths[:, t] * 
                              (1 + (self.r - self.q) * dt + sqrt_V * dW1))
        
        # Calculate payoffs
        final_prices = S_paths[:, -1]
        if self.option_type in ['call', 'c']:
            payoffs = np.maximum(0, final_prices - self.K)
        else:
            payoffs = np.maximum(0, self.K - final_prices)
        
        # Discount and return price
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.std(payoffs) * np.exp(-self.r * self.T) / np.sqrt(num_sims)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval': (
                option_price - 1.96 * std_error,
                option_price + 1.96 * std_error
            ),
            'paths': {'S': S_paths, 'V': V_paths}
        }
    
    def create_path_plot(self, num_paths=100):
        # Visualize sample paths
        result = self.price_option(num_paths, 252)
        S_paths = result['paths']['S']
        V_paths = result['paths']['V']
        
        time_grid = np.linspace(0, self.T, S_paths.shape[1])
        
        fig = go.Figure()
        
        # Plot sample stock price paths
        for i in range(min(20, num_paths)):  # Show max 20 paths
            fig.add_trace(go.Scatter(
                x=time_grid,
                y=S_paths[i, :],
                mode='lines',
                line=dict(width=1, color='blue'),
                opacity=0.3,
                showlegend=False
            ))
        
        # Add mean path
        mean_path = np.mean(S_paths, axis=0)
        fig.add_trace(go.Scatter(
            x=time_grid,
            y=mean_path,
            mode='lines',
            line=dict(width=3, color='red'),
            name='Mean Path'
        ))
        
        fig.update_layout(
            title='Heston Model Stock Price Paths',
            xaxis_title='Time',
            yaxis_title='Stock Price',
            width=800,
            height=500
        )
        
        return fig

# Standalone function for backward compatibility
def heston_MC(S, K, T, r, V, q, rho, kappa, theta, sigma, CallPut, n, m):
    # Original Heston function maintained for compatibility
    model = HestonMonteCarloModel(S, K, T, r, V, kappa, theta, sigma, rho, CallPut, q)
    result = model.price_option(m, n)
    return result['price']