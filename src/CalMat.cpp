#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace std;
using Eigen::seq; using Eigen::all;
using Eigen::MatrixXi; using Eigen::VectorXi; using Eigen::ArrayXXi;
using Eigen::MatrixXd; using Eigen::VectorXd; using Eigen::ArrayXXd;
using Eigen::MatrixXcd; using Eigen::VectorXcd; using Eigen::ArrayXd;

MatrixXd invMat3(MatrixXd A)
{
	double a1 = A(0, 0), a2 = A(0, 1), a3 = A(0, 2);
	double b1 = A(1, 0), b2 = A(1, 1), b3 = A(1, 2);
	double c1 = A(2, 0), c2 = A(2, 1), c3 = A(2, 2);
	double D = a1 * b2 * c3 + b1 * c2 * a3 + c1 * a2 * b3 - a3 * b2 * c1 - b3 * c2 * a1 - c3 * a2 * b1;

	MatrixXd invA(3, 3);
	invA(0, 0) = b2 * c3 - b3 * c2; invA(0, 1) = c2 * a3 - c3 * a2; invA(0, 2) = a2 * b3 - a3 * b2;
	invA(1, 0) = b3 * c1 - b1 * c3; invA(1, 1) = c3 * a1 - c1 * a3; invA(1, 2) = a3 * b1 - a1 * b3;
	invA(2, 0) = b1 * c2 - b2 * c1; invA(2, 1) = c1 * a2 - c2 * a1; invA(2, 2) = a1 * b2 - a2 * b1;
	invA = invA / D;

	return invA;
}

ArrayXXd CrossMat(MatrixXd N1, VectorXd nv)
{
	int ngp = N1.rows();
	Eigen::ArrayXXd nce1_t1(ngp, 3), nce1_t2(ngp, 3);
	nce1_t1(all, 0) = nv(1) * N1(all, 2);
	nce1_t1(all, 1) = nv(2) * N1(all, 0);
	nce1_t1(all, 2) = nv(0) * N1(all, 1);

	nce1_t2(all, 0) = nv(2) * N1(all, 1);
	nce1_t2(all, 1) = nv(0) * N1(all, 2);
	nce1_t2(all, 2) = nv(1) * N1(all, 0);
	Eigen::ArrayXXd nce1 = nce1_t1 - nce1_t2;
	return nce1;
}

Eigen::ArrayXXcd CrossMat_c(Eigen::ArrayXXcd Einc, VectorXd nv)
{
	int ngp = Einc.rows();
	Eigen::ArrayXXcd NCE_1(ngp, 3), NCE_2(ngp, 3);
	NCE_1(all, 0) = nv(1) * Einc(all, 2);
	NCE_1(all, 1) = nv(2) * Einc(all, 0);
	NCE_1(all, 2) = nv(0) * Einc(all, 1);

	NCE_2(all, 0) = nv(2) * Einc(all, 1);
	NCE_2(all, 1) = nv(0) * Einc(all, 2);
	NCE_2(all, 2) = nv(1) * Einc(all, 0);
	Eigen::ArrayXXcd NCE1 = NCE_1 - NCE_2;

	return NCE1;
}

