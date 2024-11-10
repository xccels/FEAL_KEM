#include "get_Nedelec0_basis_functions.h"
#include <Eigen/Dense>
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::pair<MatrixXd, MatrixXd> Nedelec0BasisFunctions::getNedelec0BasisFunction(MatrixXd ip, int num) {

    int  M = ip.rows(), dim = 3;
    MatrixXd N(M, dim);
    MatrixXd cN(M, dim);

    if (num == 1) {
        N << Eigen::MatrixXd::Ones(M, 1) - ip.col(2) - ip.col(1), ip.col(0), ip.col(0);
        cN << Eigen::MatrixXd::Zero(M, 1), -2 * Eigen::MatrixXd::Ones(M, 1), 2 * Eigen::MatrixXd::Ones(M, 1);
    }
    else if (num == 2) {
        N << ip.col(1), Eigen::MatrixXd::Ones(M, 1) - ip.col(2) - ip.col(0), ip.col(1);
        cN << 2 * Eigen::MatrixXd::Ones(M, 1), Eigen::MatrixXd::Zero(M, 1), -2 * Eigen::MatrixXd::Ones(M, 1);
    }
    else if (num == 3) {
        N << ip.col(2), ip.col(2), Eigen::MatrixXd::Ones(M, 1) - ip.col(1) - ip.col(0);
        cN << -2 * Eigen::MatrixXd::Ones(M, 1), 2 * Eigen::MatrixXd::Ones(M, 1), Eigen::MatrixXd::Zero(M, 1);
    }
    else if (num == 4) {
        N << -ip.col(1), ip.col(0), Eigen::MatrixXd::Zero(M, 1);
        cN << Eigen::MatrixXd::Zero(M, 1), Eigen::MatrixXd::Zero(M, 1), 2 * Eigen::MatrixXd::Ones(M, 1);
    }
    else if (num == 5) {
        N << Eigen::MatrixXd::Zero(M, 1), -ip.col(2), ip.col(1);
        cN << 2 * Eigen::MatrixXd::Ones(M, 1), Eigen::MatrixXd::Zero(M, 1), Eigen::MatrixXd::Zero(M, 1);
    }
    else if (num == 6) {
        N << ip.col(2), Eigen::MatrixXd::Zero(M, 1), -ip.col(0);
        cN << Eigen::MatrixXd::Zero(M, 1), 2 * Eigen::MatrixXd::Ones(M, 1), Eigen::MatrixXd::Zero(M, 1);
    }
	return std::make_pair(N, cN);
}
