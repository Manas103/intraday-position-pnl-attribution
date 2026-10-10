// attribution_main: line-delimited CSV subprocess protocol used by the
// Python pipeline to reach the C++ revaluation/attribution hot path.
//
// Input (stdin), one position-session per line, no header:
//   id,S0,K,T0,r,sigma0,type,S1,sigma1,T1
//     id      - opaque string identifying the (instrument, session)
//     S0,K,T0,r,sigma0,type - start-of-day market state and contract terms
//                              type is "C" or "P"
//     S1,sigma1,T1 - end-of-day spot, vol, time-to-maturity (same K, r)
//
// Output (stdout), one line per input line, same order, CSV, no header:
//   id,price0,price1,actual_pnl,delta0,gamma0,vega0,theta0,
//   delta_pnl,gamma_pnl,vega_pnl,theta_pnl,taylor_sum,residual,
//   vanna0,volga0,vanna_pnl,volga_pnl,taylor_sum_full,residual_full
//
// The first 14 fields are the original 4-Greek (delta/gamma/vega/theta)
// decomposition and are untouched by this extension: taylor_sum and
// residual here are exactly what they were before. The 6 fields appended
// at the end are the cross-Greek extension: vanna0/volga0 are the
// start-of-day cross Greeks, vanna_pnl/volga_pnl are their simultaneous
// (not sequential) contribution to the same P&L move, taylor_sum_full
// adds them to the original taylor_sum, and residual_full is what is left
// over once they are included.
//
// All floating point fields are printed with 15 significant digits
// (round-trip precision for IEEE-754 double) so results are exact and
// reproducible byte-for-byte given the same input.
//
// Why a subprocess/CSV protocol and not pybind11/nanobind: the attribution
// math only needs to run once per (position, session) pair in batch, not in
// a tight interactive loop, so process-launch overhead is negligible next
// to the I/O and SQLite work already dominating a session. A subprocess
// with a documented text protocol needs zero build-system integration
// between CPython's ABI and the C++ toolchain, builds identically on WSL2
// g++ used here and on a real desk's Linux boxes, and is trivial to test
// standalone (see cpp/tests). See README for the measured overhead.

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "black_scholes.hpp"

namespace {

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream ss(line);
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    std::vector<std::string> out_lines;

    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f = split_csv(line);
        if (f.size() != 10) {
            std::cerr << "malformed line, expected 10 fields, got " << f.size() << ": " << line << "\n";
            return 2;
        }
        const std::string& id = f[0];
        double S0 = std::stod(f[1]);
        double K = std::stod(f[2]);
        double T0 = std::stod(f[3]);
        double r = std::stod(f[4]);
        double sigma0 = std::stod(f[5]);
        bs::OptionType type = (f[6] == "C") ? bs::OptionType::Call : bs::OptionType::Put;
        double S1 = std::stod(f[7]);
        double sigma1 = std::stod(f[8]);
        double T1 = std::stod(f[9]);

        bs::Greeks g0 = bs::price_and_greeks(S0, K, T0, r, sigma0, type);
        double price1 = bs::price(S1, K, T1, r, sigma1, type);

        double dS = S1 - S0;
        double dVol = sigma1 - sigma0;
        double dT = T1 - T0;

        double actual_pnl = price1 - g0.price;
        double delta_pnl = g0.delta * dS;
        double gamma_pnl = 0.5 * g0.gamma * dS * dS;
        double vega_pnl = g0.vega * dVol;
        double theta_pnl = g0.theta * dT;
        double taylor_sum = delta_pnl + gamma_pnl + vega_pnl + theta_pnl;
        double residual = actual_pnl - taylor_sum;

        // Cross-Greek extension: vanna and volga, computed simultaneously
        // with everything above from the same SOD Greeks and the same
        // (dS, dVol) deltas, not as a second pass that bumps one factor on
        // top of an already-moved one. That is what makes the vanna/volga
        // buckets attribute to themselves rather than to whichever of
        // delta/gamma or vega happened to be "priced second" in a
        // sequential waterfall; see README and
        // scripts/run_sequential_order_check.py.
        double vanna_pnl = g0.vanna * dS * dVol;
        double volga_pnl = 0.5 * g0.volga * dVol * dVol;
        double taylor_sum_full = taylor_sum + vanna_pnl + volga_pnl;
        double residual_full = actual_pnl - taylor_sum_full;

        char buf[896];
        std::snprintf(buf, sizeof(buf),
                      "%s,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,"
                      "%.15g,%.15g,%.15g,%.15g,%.15g,%.15g",
                      id.c_str(), g0.price, price1, actual_pnl, g0.delta, g0.gamma, g0.vega, g0.theta,
                      delta_pnl, gamma_pnl, vega_pnl, theta_pnl, taylor_sum, residual, g0.vanna, g0.volga,
                      vanna_pnl, volga_pnl, taylor_sum_full, residual_full);
        out_lines.emplace_back(buf);
    }

    for (const auto& l : out_lines) {
        std::cout << l << "\n";
    }
    return 0;
}
