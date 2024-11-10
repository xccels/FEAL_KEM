#ifndef GET_GAUSSIAN_QUADRATURE_FACE_H
#define GET_GAUSSIAN_QUADRATURE_FACE_H

#include <vector>
#include <Eigen/Dense>
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class GaussianQuadratureFace {
public:
    static std::pair<MatrixXd, MatrixXd> getGaussianQuadratureFace(int p, int faceNum);
};

#endif // GET_GAUSSIAN_QUADRATURE_FACE_H