#pragma once
// Black-Scholes European option pricer and Greeks.
// Implemented independently for this repository; no code imported from
// any other project in this portfolio.

namespace bs {

enum class OptionType { Call, Put };

struct Greeks {
    double price;
    double delta;
    double gamma;
    double vega;   // per 1.00 (100 vol points) change in sigma
    double theta;  // per 1.00 year change in time to maturity T
                   // (i.e. d(price)/dT, NOT the conventional "per calendar day"
                   // sign-flipped theta; see README for the exact convention)
};

// S: spot, K: strike, T: time to maturity in years, r: continuously
// compounded risk-free rate, sigma: annualized volatility.
// T <= 0 or sigma <= 0 are handled as degenerate (expired / zero-vol) cases
// and return the intrinsic value with zero sensitivities.
Greeks price_and_greeks(double S, double K, double T, double r, double sigma, OptionType type);

// Convenience: price only, used for full repricing (the reference oracle).
double price(double S, double K, double T, double r, double sigma, OptionType type);

}  // namespace bs
