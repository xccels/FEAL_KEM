#include <iostream>
#include <complex>
#include <Eigen/Dense>
#include "CONSTANTS.h"
#include <math.h>
#include <cmath>
#include <complex>

using namespace std;


Eigen::ArrayXXcd Einc_f(Eigen::ArrayXXd p)
{
	Constants Cons;
	complex<double> k0 = Cons.k0;
	complex<double> alp = k0 * complex<double>(0, 1);

	Eigen::ArrayXXcd Einc = Eigen::ArrayXXcd::Zero(p.rows(), p.cols());
	//Einc(Eigen::all, 0) = (alp * p(Eigen::all, 2)).exp();
	//Einc(Eigen::all, 2) = (alp * p(Eigen::all, 0)).exp();
	//Einc(Eigen::all, 0) = (-alp * p(Eigen::all, 2)).exp();
	Einc(Eigen::all, 0) = (-alp * p(Eigen::all, 2)).exp();

	return Einc;
}

Eigen::ArrayXXcd CurlEinc_f(Eigen::ArrayXXd p)
{
	Constants Cons;
	complex<double> k0 = Cons.k0;
	complex<double> alp = k0 * complex<double>(0, 1);

	Eigen::ArrayXXcd cEinc = Eigen::ArrayXXcd::Zero(p.rows(), p.cols());
	//cEinc(Eigen::all, 1) = alp * (alp * p(Eigen::all, 2)).exp();
	cEinc(Eigen::all, 1) = -alp * (alp * p(Eigen::all, 0)).exp();
	//cEinc(Eigen::all, 1) = -alp * (alp * p(Eigen::all, 2)).exp();

	return cEinc;
}

Eigen::ArrayXXcd Einc_Angle(const Eigen::ArrayXXd& p, complex<double> alpha, double theta, double phi, complex<double> Eth, complex<double> Eph) 
{

	Constants Cons;
	complex<double> pi = Cons.pi;
	double PI = pi.real();

	double theta_rad = theta * PI / 180.0;
	double phi_rad = phi * PI / 180.0;

	double kx = -sin(theta_rad) * cos(phi_rad);
	double ky = -sin(theta_rad) * sin(phi_rad);
	double kz = -cos(theta_rad);

	complex<double> E0x = Eth * cos(theta_rad) * cos(phi_rad) - Eph * sin(phi_rad);
	complex<double> E0y = Eth * cos(theta_rad) * sin(phi_rad) + Eph * cos(phi_rad);
	complex<double> E0z = -Eth * sin(theta_rad);

	Eigen::VectorXd  phase = kx * p.col(0) + ky * p.col(1) + kz * p.col(2);
	Eigen::VectorXcd phasor = (-alpha * phase.array()).exp();

	Eigen::MatrixXcd val = Eigen::MatrixXcd::Zero(p.rows(), 3);

	val.col(0) = (E0x * phasor).matrix();
	val.col(1) = (E0y * phasor).matrix();
	val.col(2) = (E0z * phasor).matrix();

	return val;
}

Eigen::MatrixXcd cEinc_Angle(const Eigen::MatrixXd& p, complex<double> alpha, double theta, double phi, complex<double> Eth, complex<double> Eph) {

	Constants Cons;
	complex<double> pi = Cons.pi;
	double PI = pi.real();

	double theta_rad = theta * PI / 180.0;
	double phi_rad = phi * PI / 180.0;

	double kx = -sin(theta_rad) * cos(phi_rad);
	double ky = -sin(theta_rad) * sin(phi_rad);
	double kz = -cos(theta_rad);

	complex<double> E0x = Eth * cos(theta_rad) * cos(phi_rad) - Eph * sin(phi_rad);
	complex<double> E0y = Eth * cos(theta_rad) * sin(phi_rad) + Eph * cos(phi_rad);
	complex<double> E0z = -Eth * sin(theta_rad);

	Eigen::VectorXd  phase = kx * p.col(0) + ky * p.col(1) + kz * p.col(2);
	Eigen::VectorXcd phasor = (-alpha * phase.array()).exp();

	Eigen::MatrixXcd val = Eigen::MatrixXcd::Zero(p.rows(), 3);

	val.col(0) = ((-alpha) * (ky * E0z - kz * E0y) * phasor).matrix();
	val.col(1) = ((-alpha) * (kz * E0x - kx * E0z) * phasor).matrix();
	val.col(2) = ((-alpha) * (kx * E0y - ky * E0x) * phasor).matrix();

	return val;
}

