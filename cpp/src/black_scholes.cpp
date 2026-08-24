#include "black_scholes.hpp"

#include <cmath>

namespace bs {

namespace {

constexpr double kInvSqrt2Pi = 0.3989422804014327;  // 1 / sqrt(2*pi)

double norm_pdf(double x) {
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

}  // namespace

Greeks price_and_greeks(double S, double K, double T, double r, double sigma, OptionType type) {
    Greeks g{};

    if (T <= 0.0 || sigma <= 0.0) {
        // Expired or degenerate: value is intrinsic, sensitivities are zero
        // except delta, which is 0 or 1 (call) / 0 or -1 (put) at the strike
        // boundary. This branch only fires for edge-case inputs exercised by
        // the C++ test suite; the simulator never produces T <= 0 mid-session.
        double intrinsic = (type == OptionType::Call) ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
        g.price = intrinsic;
        if (type == OptionType::Call) {
            g.delta = (S > K) ? 1.0 : 0.0;
        } else {
            g.delta = (S < K) ? -1.0 : 0.0;
        }
        g.gamma = 0.0;
        g.vega = 0.0;
        g.theta = 0.0;
        return g;
    }

    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;

    double Nd1 = norm_cdf(d1);
    double Nd2 = norm_cdf(d2);
    double pdf_d1 = norm_pdf(d1);
    double disc = std::exp(-r * T);

    if (type == OptionType::Call) {
        g.price = S * Nd1 - K * disc * Nd2;
        g.delta = Nd1;
        // d(price)/dT for a call, i.e. sensitivity to *increasing* time to
        // maturity (see header for the sign convention used throughout).
        g.theta = (S * pdf_d1 * sigma) / (2.0 * sqrtT) + r * K * disc * Nd2;
    } else {
        double Nnegd1 = norm_cdf(-d1);
        double Nnegd2 = norm_cdf(-d2);
        g.price = K * disc * Nnegd2 - S * Nnegd1;
        g.delta = Nd1 - 1.0;
        g.theta = (S * pdf_d1 * sigma) / (2.0 * sqrtT) - r * K * disc * Nnegd2;
    }

    g.gamma = pdf_d1 / (S * sigma * sqrtT);
    g.vega = S * pdf_d1 * sqrtT;

    return g;
}

double price(double S, double K, double T, double r, double sigma, OptionType type) {
    return price_and_greeks(S, K, T, r, sigma, type).price;
}

}  // namespace bs
