import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings

class DataFetcher:
    def __init__(self):
        self.risk_free_rate = 0.05  # Default 5%
        
    def get_stock_data(self, ticker, period='1y', interval='1d'):
        # Fetch historical stock data
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                print(f"No data found for ticker {ticker}")
                return pd.DataFrame()
            
            # Clean and format data
            hist.reset_index(inplace=True)
            hist['ticker'] = ticker
            
            return hist
            
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, ticker):
        # Get current stock price
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            return float(current_price)
            
        except Exception as e:
            print(f"Error fetching current price for {ticker}: {e}")
            return None
    
    def calculate_historical_volatility(self, ticker, period='252d', window=30):
        # Calculate historical volatility from stock returns
        try:
            # Get historical data
            stock_data = self.get_stock_data(ticker, period=period)
            
            if stock_data.empty:
                return None
            
            # Calculate daily returns
            stock_data['returns'] = stock_data['Close'].pct_change()
            
            # Calculate rolling volatility (annualized)
            stock_data['volatility'] = stock_data['returns'].rolling(
                window=window
            ).std() * np.sqrt(252)
            
            # Return most recent volatility
            latest_volatility = stock_data['volatility'].dropna().iloc[-1]
            
            return float(latest_volatility)
            
        except Exception as e:
            print(f"Error calculating historical volatility for {ticker}: {e}")
            return None
    
    def get_risk_free_rate(self, source='default'):
        # Get risk-free rate (using treasury rate proxy)
        try:
            if source == 'fred':
                # Try to get from FRED (requires additional setup)
                # For now, return default
                return self.risk_free_rate
            elif source == 'yahoo':
                # Get 10-year treasury as proxy
                treasury = yf.Ticker("^TNX")
                hist = treasury.history(period="5d")
                
                if not hist.empty:
                    rate = hist['Close'].iloc[-1] / 100  # Convert percentage
                    return float(rate)
                else:
                    return self.risk_free_rate
            else:
                return self.risk_free_rate
                
        except Exception as e:
            print(f"Error fetching risk-free rate: {e}")
            return self.risk_free_rate
    
    def fetch_options_chain(self, ticker, expiration=None):
        # Fetch complete options chain for a ticker
        try:
            stock = yf.Ticker(ticker)
            
            # Get available expiration dates
            exp_dates = stock.options
            
            if not exp_dates:
                print(f"No options available for {ticker}")
                return pd.DataFrame()
            
            # Use specific expiration or nearest one
            if expiration:
                if expiration in exp_dates:
                    target_exp = [expiration]
                else:
                    print(f"Expiration {expiration} not available")
                    return pd.DataFrame()
            else:
                # Use next 3 expirations
                target_exp = exp_dates[:3]
            
            all_options = []
            current_price = self.get_current_price(ticker)
            
            for exp_date in target_exp:
                try:
                    # Get options chain
                    opt_chain = stock.option_chain(exp_date)
                    
                    # Process calls
                    calls = opt_chain.calls.copy()
                    calls['expiration'] = exp_date
                    calls['option_type'] = 'call'
                    calls['underlying_price'] = current_price
                    
                    # Process puts
                    puts = opt_chain.puts.copy()
                    puts['expiration'] = exp_date
                    puts['option_type'] = 'put'
                    puts['underlying_price'] = current_price
                    
                    # Combine
                    combined = pd.concat([calls, puts], ignore_index=True)
                    all_options.append(combined)
                    
                except Exception as e:
                    print(f"Error fetching options for {exp_date}: {e}")
                    continue
            
            if all_options:
                final_df = pd.concat(all_options, ignore_index=True)
                
                # Add calculated fields
                final_df['mid_price'] = (final_df['bid'] + final_df['ask']) / 2
                final_df['expiration'] = pd.to_datetime(final_df['expiration'])
                
                # Calculate time to expiration
                today = pd.Timestamp.now().normalize()
                final_df['days_to_expiry'] = (final_df['expiration'] - today).dt.days
                final_df['time_to_expiry'] = final_df['days_to_expiry'] / 365.25
                
                # Filter out invalid options
                final_df = final_df[
                    (final_df['bid'] > 0) & 
                    (final_df['ask'] > 0) & 
                    (final_df['time_to_expiry'] > 0)
                ]
                
                return final_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching options chain for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_market_info(self, ticker):
        # Get comprehensive market information for a ticker
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key information
            market_data = {
                'ticker': ticker,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': self.get_current_price(ticker),
                'market_cap': info.get('marketCap', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', None),
                '52_week_high': info.get('fiftyTwoWeekHigh', None),
                '52_week_low': info.get('fiftyTwoWeekLow', None)
            }
            
            # Add calculated metrics
            if market_data['current_price']:
                hist_vol = self.calculate_historical_volatility(ticker)
                market_data['historical_volatility'] = hist_vol
            
            return market_data
            
        except Exception as e:
            print(f"Error fetching market info for {ticker}: {e}")
            return {}
    
    def validate_ticker(self, ticker):
        # Validate if ticker exists and has options
        try:
            stock = yf.Ticker(ticker)
            
            # Check if stock data exists
            hist = stock.history(period="5d")
            if hist.empty:
                return False, "Ticker not found"
            
            # Check if options are available
            try:
                options = stock.options
                if not options:
                    return False, "No options available for this ticker"
            except:
                return False, "No options available for this ticker"
            
            return True, "Valid ticker with options"
            
        except Exception as e:
            return False, f"Error validating ticker: {e}"
    
    def get_earnings_calendar(self, ticker):
        # Get upcoming earnings date (if available)
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is not None and not calendar.empty:
                next_earnings = calendar.index[0]
                return pd.to_datetime(next_earnings)
            else:
                return None
                
        except Exception as e:
            print(f"Error fetching earnings calendar for {ticker}: {e}")
            return None
    
    def get_multiple_tickers_data(self, tickers, period='1y'):
        # Fetch data for multiple tickers efficiently
        try:
            # Download data for all tickers at once
            data = yf.download(tickers, period=period, group_by='ticker')
            
            results = {}
            
            if len(tickers) == 1:
                # Single ticker case
                ticker = tickers[0]
                results[ticker] = data
            else:
                # Multiple tickers case
                for ticker in tickers:
                    if ticker in data.columns.levels[0]:
                        results[ticker] = data[ticker]
            
            return results
            
        except Exception as e:
            print(f"Error fetching multiple ticker data: {e}")
            return {}
    
    def get_financial_ratios(self, ticker):
        # Calculate key financial ratios
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            
            if hist.empty:
                return {}
            
            current_price = hist['Close'].iloc[-1]
            
            ratios = {
                'price_to_earnings': info.get('trailingPE'),
                'price_to_book': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'return_on_equity': info.get('returnOnEquity'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio')
            }
            
            # Calculate additional ratios from price data
            returns = hist['Close'].pct_change().dropna()
            ratios['annual_volatility'] = returns.std() * np.sqrt(252)
            ratios['sharpe_ratio'] = (returns.mean() * 252) / ratios['annual_volatility'] if ratios['annual_volatility'] > 0 else 0
            
            return ratios
            
        except Exception as e:
            print(f"Error calculating financial ratios for {ticker}: {e}")
            return {}
