#ifndef GET_GAUSSIAN_QUADRATURE_H
#define GET_GAUSSIAN_QUADRATURE_H

#include <vector>
#include <Eigen/Dense>
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class GaussianQuadrature {
public:
    static std::pair<MatrixXd, MatrixXd> getGaussianQuadrature(int p, int dim);
};

#endif // GET_GAUSSIAN_QUADRATURE_H