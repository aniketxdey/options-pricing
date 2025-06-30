# Black-Scholes Model Literature & Implementation Notes

## Overview

The Black-Scholes model provides analytical solutions for European option pricing based on delta hedging and no-arbitrage principles. When dividends are included, it becomes the Black-Scholes-Merton model. This document outlines the theoretical foundation supporting the implementation in `black_scholes.py`.

## Mathematical Foundation

The model creates a risk-free portfolio through the relationship Π = V(S,t) - ΔS, where delta hedging eliminates randomness by setting Δ = ∂V/∂S. The no-arbitrage principle requires that all risk-free portfolios earn the risk-free rate, leading to the Black-Scholes partial differential equation:

```
∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
```

A key insight is that the drift rate (μ) disappears from the final equation due to perfect hedging, which is why only the risk-free rate matters in the pricing formula.

## Model Assumptions

The model requires several critical assumptions that enable analytical solutions. The underlying asset must follow a lognormal random walk with constant volatility, while the risk-free rate remains constant. The model assumes no dividends during the option's life (unless using the dividend-adjusted version), continuous delta hedging capability, no transaction costs, perfect market liquidity, and European exercise only.

## Pricing Formulas

The standard Black-Scholes formulas implemented in `BlackScholesModel.price()` are:

**Call Option:** c = S₀N(d₁) - Ke^(-rT)N(d₂)
**Put Option:** p = Ke^(-rT)N(-d₂) - S₀N(-d₁)

Where d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T) and d₂ = d₁ - σ√T.

The implementation uses `scipy.stats.norm.cdf` for the cumulative standard normal distribution N(x) and handles both single values and arrays while validating all inputs for positivity constraints.

## The Greeks

The Greeks measure option price sensitivities and are implemented as partial derivatives in the `BlackScholesModel` class.

**Delta (Δ)** measures price sensitivity to underlying asset changes. Call delta equals N(d₁) while put delta equals N(d₁) - 1, implemented in `BlackScholesModel.delta()`.

**Gamma (Γ)** measures delta sensitivity using the formula φ(d₁)/(S₀σ√T), where φ is the standard normal probability density function. Gamma is identical for calls and puts.

**Vega (ν)** calculates volatility sensitivity as S₀φ(d₁)√T, converted to percentage points by dividing by 100 in the implementation.

**Theta (Θ)** represents time decay with different formulas for calls and puts, then converted to daily values by dividing by 365.

**Rho (ρ)** measures interest rate sensitivity, with call rho as KTe^(-rT)N(d₂) and put rho as the negative of this value, converted to percentage points.

## Extensions and Applications

**Dividend-Adjusted Model:** For dividend-paying assets, the model replaces S₀ with S₀e^(-DT) in calculations, where D is the continuous dividend yield. This is implemented through the `dividend_yield` parameter in `BlackScholesModel.__init__()`.

**Implied Volatility:** The `BlackScholesModel.implied_volatility()` method uses Newton-Raphson iteration with the update σ_new = σ_old - (BS_price - market_price) / vega. The implementation includes convergence criteria and boundary conditions to ensure σ > 0.

**Put-Call Parity:** The relationship C - P = S*e^(-q*T) - K*e^(-r*T) is implemented in `BlackScholesModel.parity_check()` for model validation and arbitrage detection.

## Practical Considerations

The model's limitations include the constant volatility assumption, ignored transaction costs, and impossible continuous hedging in practice. The implementation addresses these through robust input validation, array broadcasting for batch calculations, and graceful error handling in edge cases while maintaining backward compatibility with standalone functions.

## References

Black, Fischer, and Myron Scholes. "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, vol. 81, no. 3, 1973, pp. 637-654.

Hull, John C. *Options, Futures, and Other Derivatives*. 10th ed., Pearson, 2017.

Wilmott, Paul. *Paul Wilmott Introduces Quantitative Finance*. 2nd ed., John Wiley & Sons, 2007.