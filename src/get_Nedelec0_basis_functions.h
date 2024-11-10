#ifndef GET_NEDELE0_BASIS_FUNCTIONS_H
#define GET_NEDELE0_BASIS_FUNCTIONS_H

#include <vector>
#include <Eigen/Dense>
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class Nedelec0BasisFunctions {
public:
    static std::pair<MatrixXd, MatrixXd> getNedelec0BasisFunction(MatrixXd ip, int num);
};

#endif // GET_NEDELE0_BASIS_FUNCTIONS_H