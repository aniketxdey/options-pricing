import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from .option_models import ImpliedVolatilityCalculator

class ImpliedVolatilitySurface:
    def __init__(self, ticker):
        self.ticker = ticker
        self.iv_calculator = ImpliedVolatilityCalculator()
        self.stock_price = None
        self.risk_free_rate = 0.05  # Default 5%
        
    def fetch_stock_price(self):
        # Get current stock price
        try:
            stock = yf.Ticker(self.ticker)
            hist = stock.history(period="1d")
            self.stock_price = hist['Close'].iloc[-1]
            return self.stock_price
        except Exception as e:
            print(f"Error fetching stock price for {self.ticker}: {e}")
            return None
    
    def fetch_options_chain(self, expiration_dates=None):
        # Fetch options chain data
        try:
            stock = yf.Ticker(self.ticker)
            
            # Get available expiration dates if not provided
            if expiration_dates is None:
                exp_dates = stock.options[:5]  # Get first 5 expiration dates
            else:
                exp_dates = expiration_dates
            
            options_data = []
            current_price = self.fetch_stock_price()
            
            if current_price is None:
                return pd.DataFrame()
            
            for exp_date in exp_dates:
                try:
                    # Get options chain for this expiration
                    opt_chain = stock.option_chain(exp_date)
                    
                    # Process calls
                    calls = opt_chain.calls.copy()
                    calls['expiration'] = exp_date
                    calls['option_type'] = 'call'
                    calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
                    
                    # Process puts  
                    puts = opt_chain.puts.copy()
                    puts['expiration'] = exp_date
                    puts['option_type'] = 'put'
                    puts['mid_price'] = (puts['bid'] + puts['ask']) / 2
                    
                    # Combine and filter
                    combined = pd.concat([calls, puts], ignore_index=True)
                    
                    # Filter out options with zero bid/ask
                    combined = combined[(combined['bid'] > 0) & (combined['ask'] > 0)]
                    
                    options_data.append(combined)
                    
                except Exception as e:
                    print(f"Error fetching options for {exp_date}: {e}")
                    continue
            
            if options_data:
                full_chain = pd.concat(options_data, ignore_index=True)
                
                # Add time to expiration
                full_chain['expiration'] = pd.to_datetime(full_chain['expiration'])
                today = pd.Timestamp.now().normalize()
                full_chain['days_to_expiry'] = (full_chain['expiration'] - today).dt.days
                full_chain['time_to_expiry'] = full_chain['days_to_expiry'] / 365.25
                
                # Add current stock price
                full_chain['stock_price'] = current_price
                
                return full_chain
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching options chain for {self.ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_iv_for_chain(self, options_df, method='brent'):
        # Calculate implied volatility for entire options chain
        if options_df.empty:
            return options_df
        
        iv_list = []
        
        for idx, row in options_df.iterrows():
            try:
                if row['mid_price'] > 0 and row['time_to_expiry'] > 0:
                    iv = self.iv_calculator.calculate_iv(
                        market_price=row['mid_price'],
                        S=row['stock_price'],
                        K=row['strike'],
                        T=row['time_to_expiry'],
                        r=self.risk_free_rate,
                        option_type=row['option_type'],
                        method=method
                    )
                    iv_list.append(iv)
                else:
                    iv_list.append(np.nan)
            except Exception as e:
                iv_list.append(np.nan)
        
        options_df['calculated_iv'] = iv_list
        
        # Remove rows with invalid IV
        options_df = options_df.dropna(subset=['calculated_iv'])
        options_df = options_df[options_df['calculated_iv'] > 0]
        
        return options_df
    
    def generate_surface_plot(self, options_df, surface_type='3d', iv_column='impliedVolatility'):
        # Generate 3D volatility surface plot
        if options_df.empty:
            return None
        
        # Filter for reasonable strikes (around ATM)
        current_price = options_df['stock_price'].iloc[0]
        strike_range = (current_price * 0.8, current_price * 1.2)
        filtered_df = options_df[
            (options_df['strike'] >= strike_range[0]) & 
            (options_df['strike'] <= strike_range[1])
        ].copy()
        
        if filtered_df.empty:
            return None
        
        # Create moneyness
        filtered_df['moneyness'] = filtered_df['strike'] / current_price
        
        # Separate calls and puts
        calls = filtered_df[filtered_df['option_type'] == 'call']
        puts = filtered_df[filtered_df['option_type'] == 'put']
        
        if surface_type == '3d':
            fig = go.Figure()
            
            # Add call surface
            if not calls.empty:
                fig.add_trace(go.Scatter3d(
                    x=calls['days_to_expiry'],
                    y=calls['moneyness'],
                    z=calls[iv_column] * 100,  # Convert to percentage
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='blue',
                        opacity=0.7
                    ),
                    name='Calls'
                ))
            
            # Add put surface
            if not puts.empty:
                fig.add_trace(go.Scatter3d(
                    x=puts['days_to_expiry'],
                    y=puts['moneyness'],
                    z=puts[iv_column] * 100,  # Convert to percentage
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='red',
                        opacity=0.7
                    ),
                    name='Puts'
                ))
            
            fig.update_layout(
                title=f'Implied Volatility Surface - {self.ticker}',
                scene=dict(
                    xaxis_title='Days to Expiration',
                    yaxis_title='Moneyness (K/S)',
                    zaxis_title='Implied Volatility (%)'
                ),
                width=800,
                height=600
            )
            
        elif surface_type == 'heatmap':
            # Create pivot table for heatmap
            pivot_data = calls.pivot_table(
                values=iv_column,
                index='moneyness',
                columns='days_to_expiry',
                aggfunc='mean'
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values * 100,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='Viridis',
                colorbar=dict(title='IV (%)')
            ))
            
            fig.update_layout(
                title=f'IV Heatmap - {self.ticker} Calls',
                xaxis_title='Days to Expiration',
                yaxis_title='Moneyness (K/S)',
                width=800,
                height=600
            )
        
        return fig
    
    def analyze_volatility_smile(self, options_df, expiration_days=None):
        # Analyze volatility smile for specific expiration
        if options_df.empty:
            return None
        
        # Filter by expiration if specified
        if expiration_days:
            target_exp = options_df[
                abs(options_df['days_to_expiry'] - expiration_days) <= 2
            ]
        else:
            # Use nearest expiration
            target_exp = options_df[
                options_df['days_to_expiry'] == options_df['days_to_expiry'].min()
            ]
        
        if target_exp.empty:
            return None
        
        current_price = target_exp['stock_price'].iloc[0]
        target_exp = target_exp.copy()
        target_exp['moneyness'] = target_exp['strike'] / current_price
        
        # Separate calls and puts
        calls = target_exp[target_exp['option_type'] == 'call']
        puts = target_exp[target_exp['option_type'] == 'put']
        
        fig = go.Figure()
        
        if not calls.empty:
            fig.add_trace(go.Scatter(
                x=calls['moneyness'],
                y=calls['impliedVolatility'] * 100,
                mode='markers+lines',
                name='Calls',
                marker=dict(color='blue')
            ))
        
        if not puts.empty:
            fig.add_trace(go.Scatter(
                x=puts['moneyness'],
                y=puts['impliedVolatility'] * 100,
                mode='markers+lines',
                name='Puts',
                marker=dict(color='red')
            ))
        
        fig.update_layout(
            title=f'Volatility Smile - {self.ticker}',
            xaxis_title='Moneyness (K/S)',
            yaxis_title='Implied Volatility (%)',
            width=800,
            height=500
        )
        
        return fig
    
    def calculate_surface_statistics(self, options_df):
        # Calculate key volatility surface statistics
        if options_df.empty:
            return {}
        
        current_price = options_df['stock_price'].iloc[0]
        
        # ATM volatility for different expirations
        atm_data = []
        for days in sorted(options_df['days_to_expiry'].unique()):
            exp_data = options_df[options_df['days_to_expiry'] == days]
            
            # Find closest to ATM
            atm_idx = (exp_data['strike'] - current_price).abs().idxmin()
            atm_option = exp_data.loc[atm_idx]
            
            atm_data.append({
                'days_to_expiry': days,
                'atm_iv': atm_option['impliedVolatility'],
                'option_type': atm_option['option_type']
            })
        
        atm_df = pd.DataFrame(atm_data)
        
        # Calculate term structure slope
        if len(atm_df) >= 2:
            short_term = atm_df[atm_df['days_to_expiry'] <= 30]['atm_iv'].mean()
            long_term = atm_df[atm_df['days_to_expiry'] >= 60]['atm_iv'].mean()
            term_structure_slope = (long_term - short_term) if pd.notna(long_term) and pd.notna(short_term) else 0
        else:
            term_structure_slope = 0
        
        # Calculate skew (difference between OTM put and call IV)
        current_exp = options_df[options_df['days_to_expiry'] == options_df['days_to_expiry'].min()]
        otm_put_iv = current_exp[
            (current_exp['option_type'] == 'put') & 
            (current_exp['strike'] < current_price * 0.95)
        ]['impliedVolatility'].mean()
        
        otm_call_iv = current_exp[
            (current_exp['option_type'] == 'call') & 
            (current_exp['strike'] > current_price * 1.05)
        ]['impliedVolatility'].mean()
        
        skew = (otm_put_iv - otm_call_iv) if pd.notna(otm_put_iv) and pd.notna(otm_call_iv) else 0
        
        return {
            'current_price': current_price,
            'atm_iv_short': short_term if 'short_term' in locals() else np.nan,
            'atm_iv_long': long_term if 'long_term' in locals() else np.nan,
            'term_structure_slope': term_structure_slope,
            'skew': skew,
            'total_options': len(options_df),
            'expiration_range': f"{options_df['days_to_expiry'].min()}-{options_df['days_to_expiry'].max()} days"
        }
    
    def export_iv_data(self, options_df, filename=None, format='csv'):
        # Export IV data to file
        if options_df.empty:
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.ticker}_iv_surface_{timestamp}"
        
        if format == 'csv':
            filepath = f"output/{filename}.csv"
            options_df.to_csv(filepath, index=False)
        elif format == 'excel':
            filepath = f"output/{filename}.xlsx"
            options_df.to_excel(filepath, index=False)
        
        return filepath
