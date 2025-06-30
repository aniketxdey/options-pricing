import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import brentq, newton
import warnings

class OptionPricingModels:
    def __init__(self, S, K, T, r, sigma, option_type):
        # Handle both single values and arrays
        self.S = np.atleast_1d(np.array(S))
        self.K = np.atleast_1d(np.array(K))
        self.T = np.atleast_1d(np.array(T))
        self.r = r
        self.sigma = np.atleast_1d(np.array(sigma))
        self.option_type = np.atleast_1d(np.array(option_type))
        
        # Store original shape info
        self.is_scalar = np.isscalar(S)
        
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
    
    def black_scholes_option(self):
        # Calculate d1 and d2
        d1 = (np.log(self.S/self.K) + (self.r + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        
        # Calculate option prices
        call_prices = self.S * norm.cdf(d1) - self.K * np.exp(-self.r*self.T) * norm.cdf(d2)
        put_prices = self.K * np.exp(-self.r*self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        
        # Return appropriate prices based on option type
        prices = np.where(
            (self.option_type == 'call') | (self.option_type == 'c') | (self.option_type == 1), 
            call_prices, 
            put_prices
        )
        
        # Return scalar if input was scalar
        return prices.item() if self.is_scalar else prices
    
    def binomial_tree_option_price(self, N=100):
        # Vectorized binomial tree implementation
        dt = self.T / N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(self.r * dt) - d) / (u - d)
        
        prices = np.zeros_like(self.S)
        
        for i in range(len(self.S)):
            # Asset prices at maturity
            asset_prices = np.array([self.S[i] * (u[i]**j) * (d[i]**(N-j)) for j in range(N+1)])
            
            # Option values at maturity
            if self.option_type[i] in ['call', 'c', 1]:
                option_values = np.maximum(0, asset_prices - self.K[i])
            else:
                option_values = np.maximum(0, self.K[i] - asset_prices)
            
            # Backward induction
            for step in range(N-1, -1, -1):
                option_values = np.exp(-self.r * dt[i]) * (
                    q[i] * option_values[1:step+2] + (1-q[i]) * option_values[0:step+1]
                )
            
            prices[i] = option_values[0]
        
        # Return scalar if input was scalar
        return prices.item() if self.is_scalar else prices
    
    def new_monte_carlo_option_price(self, num_simulations=10000):
        # Standard GBM Monte Carlo
        dt = 0.01  # Small time step
        n_steps = np.maximum(1, (self.T / dt).astype(int))
        
        prices = np.zeros_like(self.S)
        paths_collection = []
        
        for i in range(len(self.S)):
            # Generate random paths
            Z = np.random.standard_normal((num_simulations, n_steps[i]))
            
            # Initialize paths
            paths = np.zeros((num_simulations, n_steps[i] + 1))
            paths[:, 0] = self.S[i]
            
            # Generate stock price paths
            for t in range(n_steps[i]):
                paths[:, t+1] = paths[:, t] * np.exp(
                    (self.r - 0.5 * self.sigma[i]**2) * dt + 
                    self.sigma[i] * np.sqrt(dt) * Z[:, t]
                )
            
            # Calculate payoffs
            final_prices = paths[:, -1]
            if self.option_type[i] in ['call', 'c', 1]:
                payoffs = np.maximum(0, final_prices - self.K[i])
            else:
                payoffs = np.maximum(0, self.K[i] - final_prices)
            
            # Discount and average
            prices[i] = np.exp(-self.r * self.T[i]) * np.mean(payoffs)
            paths_collection.append(paths)
        
        # Return scalar if input was scalar
        if self.is_scalar:
            return prices.item(), paths_collection[0] if paths_collection else None
        else:
            return prices, paths_collection
    
    def calculate_greeks(self):
        # Calculate all Greeks using Black-Scholes
        d1 = (np.log(self.S/self.K) + (self.r + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        
        # Delta
        delta = np.where(
            (self.option_type == 'call') | (self.option_type == 'c') | (self.option_type == 1),
            norm.cdf(d1),
            norm.cdf(d1) - 1
        )
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        
        # Vega (same for calls and puts)
        vega = self.S * np.sqrt(self.T) * norm.pdf(d1)
        
        # Theta
        term1 = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        term2_call = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        term2_put = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        theta = np.where(
            (self.option_type == 'call') | (self.option_type == 'c') | (self.option_type == 1),
            term1 + term2_call,
            term1 + term2_put
        )
        
        # Rho
        rho = np.where(
            (self.option_type == 'call') | (self.option_type == 'c') | (self.option_type == 1),
            self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2),
            -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        )
        
        # Return scalars if input was scalar
        if self.is_scalar:
            return {
                'delta': delta.item(),
                'gamma': gamma.item(), 
                'vega': vega.item(),
                'theta': theta.item(),
                'rho': rho.item()
            }
        else:
            return {
                'delta': delta,
                'gamma': gamma, 
                'vega': vega,
                'theta': theta,
                'rho': rho
            }

class ImpliedVolatilityCalculator:
    def __init__(self):
        self.max_iterations = 100
        self.tolerance = 1e-6
    
    def _black_scholes_price(self, S, K, T, r, sigma, option_type):
        # Single value Black-Scholes calculation
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type in ['call', 'c']:
            return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def _vega(self, S, K, T, r, sigma):
        # Vega for IV calculation
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return S * np.sqrt(T) * norm.pdf(d1)
    
    def newton_raphson_iv(self, market_price, S, K, T, r, option_type, initial_guess=0.2):
        # Newton-Raphson method for implied volatility
        sigma = initial_guess
        
        for i in range(self.max_iterations):
            try:
                price = self._black_scholes_price(S, K, T, r, sigma, option_type)
                vega = self._vega(S, K, T, r, sigma)
                
                if abs(vega) < 1e-10:
                    break
                    
                price_diff = price - market_price
                
                if abs(price_diff) < self.tolerance:
                    return sigma
                
                sigma = sigma - price_diff / vega
                
                # Ensure sigma stays positive
                sigma = max(sigma, 0.001)
                
            except:
                break
        
        return sigma if sigma > 0 else np.nan
    
    def bisection_iv(self, market_price, S, K, T, r, option_type, low=0.001, high=5.0):
        # Bisection method for implied volatility
        try:
            def price_diff(sigma):
                return self._black_scholes_price(S, K, T, r, sigma, option_type) - market_price
            
            # Check if root exists in interval
            if price_diff(low) * price_diff(high) > 0:
                return np.nan
            
            result = brentq(price_diff, low, high, xtol=self.tolerance, maxiter=self.max_iterations)
            return result
            
        except:
            return np.nan
    
    def brent_method_iv(self, market_price, S, K, T, r, option_type, low=0.001, high=5.0):
        # Brent's method (more robust than bisection)
        try:
            def price_diff(sigma):
                return self._black_scholes_price(S, K, T, r, sigma, option_type) - market_price
            
            result = brentq(price_diff, low, high, xtol=self.tolerance, maxiter=self.max_iterations)
            return result
            
        except:
            return np.nan
    
    def calculate_iv(self, market_price, S, K, T, r, option_type, method='brent'):
        # Main IV calculation method with fallback
        if method == 'newton':
            iv = self.newton_raphson_iv(market_price, S, K, T, r, option_type)
        elif method == 'bisection':
            iv = self.bisection_iv(market_price, S, K, T, r, option_type)
        else:  # brent (default)
            iv = self.brent_method_iv(market_price, S, K, T, r, option_type)
        
        # Fallback to Newton-Raphson if other methods fail
        if np.isnan(iv) and method != 'newton':
            iv = self.newton_raphson_iv(market_price, S, K, T, r, option_type)
        
        return iv