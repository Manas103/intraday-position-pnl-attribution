// Hand-rolled assertion-based test suite for the Black-Scholes pricer.
// No GoogleTest dependency: keeps the WSL build self-contained.
// Run: ./build/test_black_scholes  (exits non-zero and prints on failure)

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../src/black_scholes.hpp"

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        g_failures++;
    } else {
        std::printf("ok: %s\n", what);
    }
}

bool close(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

}  // namespace

int main() {
    // 1) Known reference value: S=100, K=100, T=1, r=0.05, sigma=0.2 (call).
    // Standard textbook value (Hull): call price ~= 10.4506.
    {
        auto g = bs::price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.2, bs::OptionType::Call);
        check(close(g.price, 10.4506, 1e-3), "ATM call price matches textbook reference (Hull example)");
    }

    // 2) Put-call parity: C - P = S - K*exp(-rT), must hold to near machine precision.
    {
        double S = 123.4, K = 110.0, T = 0.73, r = 0.03, sigma = 0.35;
        double c = bs::price(S, K, T, r, sigma, bs::OptionType::Call);
        double p = bs::price(S, K, T, r, sigma, bs::OptionType::Put);
        double parity_rhs = S - K * std::exp(-r * T);
        check(close(c - p, parity_rhs, 1e-9), "put-call parity holds to 1e-9");
    }

    // 3) Greeks cross-checked against central finite differences of price().
    {
        double S = 87.0, K = 95.0, T = 0.4, r = 0.02, sigma = 0.28;
        auto g = bs::price_and_greeks(S, K, T, r, sigma, bs::OptionType::Call);

        double hS = 1e-4 * S;
        double fd_delta = (bs::price(S + hS, K, T, r, sigma, bs::OptionType::Call) -
                            bs::price(S - hS, K, T, r, sigma, bs::OptionType::Call)) / (2 * hS);
        check(close(g.delta, fd_delta, 1e-5), "delta matches finite-difference derivative w.r.t. S");

        double fd_gamma = (bs::price(S + hS, K, T, r, sigma, bs::OptionType::Call) -
                            2 * bs::price(S, K, T, r, sigma, bs::OptionType::Call) +
                            bs::price(S - hS, K, T, r, sigma, bs::OptionType::Call)) / (hS * hS);
        check(close(g.gamma, fd_gamma, 1e-4), "gamma matches finite-difference second derivative w.r.t. S");

        double hVol = 1e-5;
        double fd_vega = (bs::price(S, K, T, r, sigma + hVol, bs::OptionType::Call) -
                           bs::price(S, K, T, r, sigma - hVol, bs::OptionType::Call)) / (2 * hVol);
        check(close(g.vega, fd_vega, 1e-4), "vega matches finite-difference derivative w.r.t. sigma");

        double hT = 1e-5;
        double fd_theta = (bs::price(S, K, T + hT, r, sigma, bs::OptionType::Call) -
                            bs::price(S, K, T - hT, r, sigma, bs::OptionType::Call)) / (2 * hT);
        check(close(g.theta, fd_theta, 1e-3), "theta matches finite-difference derivative w.r.t. T");
    }

    // 4) Put greeks cross-checked the same way.
    {
        double S = 110.0, K = 100.0, T = 0.6, r = 0.015, sigma = 0.22;
        auto g = bs::price_and_greeks(S, K, T, r, sigma, bs::OptionType::Put);

        double hS = 1e-4 * S;
        double fd_delta = (bs::price(S + hS, K, T, r, sigma, bs::OptionType::Put) -
                            bs::price(S - hS, K, T, r, sigma, bs::OptionType::Put)) / (2 * hS);
        check(close(g.delta, fd_delta, 1e-5), "put delta matches finite-difference derivative w.r.t. S");
        check(g.delta < 0.0, "put delta is negative");
    }

    // 5) Degenerate/edge case: expired option (T = 0) returns intrinsic value.
    {
        auto g_itm = bs::price_and_greeks(120.0, 100.0, 0.0, 0.03, 0.2, bs::OptionType::Call);
        check(close(g_itm.price, 20.0, 1e-9), "expired ITM call prices at intrinsic value");
        auto g_otm = bs::price_and_greeks(80.0, 100.0, 0.0, 0.03, 0.2, bs::OptionType::Call);
        check(close(g_otm.price, 0.0, 1e-9), "expired OTM call prices at zero");
    }

    // 6) Deep ITM call delta approaches 1, deep OTM approaches 0.
    {
        auto deep_itm = bs::price_and_greeks(300.0, 100.0, 0.5, 0.02, 0.2, bs::OptionType::Call);
        check(deep_itm.delta > 0.999, "deep ITM call delta approaches 1");
        auto deep_otm = bs::price_and_greeks(20.0, 100.0, 0.5, 0.02, 0.2, bs::OptionType::Call);
        check(deep_otm.delta < 0.001, "deep OTM call delta approaches 0");
    }

    if (g_failures > 0) {
        std::fprintf(stderr, "\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
