#define _USE_MATH_DEFINES 
#include "CONSTANTS.h"
#include <math.h>
#include <cmath>
#include <complex>

double PI = M_PI;
const std::complex<double> Constants::pi = std::complex<double>(PI, 0);
const std::complex<double> Constants::c0 = std::complex<double>(3 * pow(10, 8), 0);
const std::complex<double> Constants::f0 = std::complex<double>(5 * pow(10, 9), 0);
const std::complex<double> Constants::lam0 = Constants::c0 / Constants::f0;
const std::complex<double> Constants::mu_rd = std::complex<double>(1, 0);
const std::complex<double> Constants::eps_rd = std::complex<double>(6.645, -2.7);
const std::complex<double> Constants::tau = std::complex<double>(0.01, 0) * Constants::lam0;
const std::complex<double> Constants::mu_r = std::complex<double>(1, 0);
const std::complex<double> Constants::eps_r = std::complex<double>(1, 0);
const std::complex<double> Constants::k0 = std::complex<double>(2, 0) * Constants::pi / Constants::lam0;
std::complex<double> Constants::eta = std::complex<double>(0, 1) * sqrt(Constants::mu_rd / Constants::eps_rd) * tan(Constants::k0 * sqrt(Constants::eps_rd / Constants::mu_rd) * Constants::tau);
