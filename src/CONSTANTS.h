#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <complex>

struct Constants
{
    static const std::complex<double> pi;
    static const std::complex<double> c0;
    static const std::complex<double> f0;
    static const std::complex<double> lam0;
    static const std::complex<double> mu_rd;
    static const std::complex<double> eps_rd;
    static const std::complex<double> tau;
    static const std::complex<double> mu_r;
    static const std::complex<double> eps_r;
    static const std::complex<double> k0;
    static std::complex<double> eta;
};

#endif // CONSTANTS_H