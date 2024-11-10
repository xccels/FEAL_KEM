#include "get_gaussian_quadrature_face.h"
#include <vector>
#include <set>
#include <algorithm>
#include <Eigen/Dense>
#include <igl/sort.h>
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void appendMatricesF(MatrixXd& dest, MatrixXd& src) {
    if (dest.size() == 0) {
        dest = src;
    }
    else {
        dest.conservativeResize(dest.rows() + src.rows(), Eigen::NoChange);
        dest.bottomRows(src.rows()) = src;
    }
}

std::pair<MatrixXd, MatrixXd> GaussianQuadratureFace::getGaussianQuadratureFace(int p, int faceNum) {
    MatrixXd X, W;

    if (faceNum == 0) {
        if (p == 1) {
            MatrixXd X1; X1 << 0.333333333333333, 0.333333333333333, 0;
            MatrixXd W1; W1 << 1;
            appendMatricesF(X, X1);
            appendMatricesF(W, W1);
        }
        else if (p == 2) {
            MatrixXd X1(3, 3);
            X1.row(2) << 0.166666666666667, 0.666666666666667, 0;
            X1.row(1) << 0.666666666666667, 0.166666666666667, 0;
            X1.row(0) << 0.166666666666667, 0.166666666666667, 0;
            MatrixXd W1(3, 1);
            W1 << 0.333333333333333, 0.333333333333333, 0.333333333333333;
            appendMatricesF(X, X1);
            appendMatricesF(W, W1);
        }
        else if (p == 3) {
            MatrixXd X1(4, 3);
            X1.row(0) << 0.333333333333333, 0.333333333333333, 0;
            X1.row(1) << 0.200000000000000, 0.600000000000000, 0;
            X1.row(2) << 0.600000000000000, 0.200000000000000, 0;
            X1.row(3) << 0.200000000000000, 0.200000000000000, 0;
            MatrixXd W1(4, 1);
            W1<< -0.281250000000000, 0.260416666666667, 0.260416666666667, 0.260416666666667;
            W1 = W1 * 2;
            appendMatricesF(X, X1);
            appendMatricesF(W, W1);
        }
    }

    else if (faceNum == 1) {
        if (p == 1) {
            MatrixXd X1; X1 << 0.3, 0, 0.3;
            MatrixXd W1; W1 << 1;
            appendMatricesF(X, X1);
            appendMatricesF(W, W1);
        }
        else if (p == 2) {
            MatrixXd X1(3, 3);
            X1.row(2) << 0.166666666666667, 0, 0.666666666666667;
            X1.row(1) << 0.666666666666667, 0, 0.166666666666667;
            X1.row(0) << 0.166666666666667, 0, 0.166666666666667;
            MatrixXd W1(3, 1);
            W1 << 0.333333333333333, 0.333333333333333, 0.333333333333333;
            appendMatricesF(X, X1);
            appendMatricesF(W, W1);
        }
        else if (p == 3) {
            MatrixXd X1(4, 3);
            X1.row(0) << 0.333333333333333, 0, 0.333333333333333;
            X1.row(1) << 0.200000000000000, 0, 0.600000000000000;
            X1.row(2) << 0.600000000000000, 0, 0.200000000000000;
            X1.row(3) << 0.200000000000000, 0, 0.200000000000000;
            MatrixXd W1(4, 1);
            W1 << -0.281250000000000, 0.260416666666667, 0.260416666666667, 0.260416666666667;
            W1 = W1 * 2;
            appendMatricesF(X, X1);
            appendMatricesF(W, W1);
        }
    }

    else if (faceNum == 2) {
        if (p == 1) {
            MatrixXd X1; X1 << 0, 0.3, 0.3;
            MatrixXd W1; W1 << 0.5;
            appendMatricesF(X, X1);
            appendMatricesF(W, W1);
        }
        else if (p == 2) {
            MatrixXd X1(3, 3);
            X1.row(2) << 0, 0.166666666666667, 0.666666666666667;
            X1.row(1) << 0, 0.666666666666667, 0.166666666666667;
            X1.row(0) << 0, 0.166666666666667, 0.166666666666667;
            MatrixXd W1(3, 1);
            W1 << 0.333333333333333, 0.333333333333333, 0.333333333333333;
            appendMatricesF(X, X1);
            appendMatricesF(W, W1);
        }
        else if (p == 3) {
            MatrixXd X1(4, 3);
            X1.row(0) << 0, 0.333333333333333, 0.333333333333333;
            X1.row(1) << 0, 0.200000000000000, 0.600000000000000;
            X1.row(2) << 0, 0.600000000000000, 0.200000000000000;
            X1.row(3) << 0, 0.200000000000000, 0.200000000000000;
            MatrixXd W1(4, 1);
            W1 << -0.281250000000000, 0.260416666666667, 0.260416666666667, 0.260416666666667;
            W1 = W1 * 2;
            appendMatricesF(X, X1);
            appendMatricesF(W, W1);
        }
    }

    else if (faceNum == 3) {
        if (p == 1) {
            MatrixXd X1; X1 << 0.3, 0, 0.3;
            MatrixXd W1; W1 << 0.5;
            appendMatricesF(X, X1);
            appendMatricesF(W, W1);
        }
        else if (p == 2) {
            MatrixXd X1(3, 3);
            X1.row(2) << 0.166666666666667, 0.166666666666667, 0.666666666666667;
            X1.row(1) << 0.166666666666667, 0.666666666666667, 0.166666666666667;
            X1.row(0) << 0.666666666666667, 0.166666666666667, 0.166666666666667;
            MatrixXd W1(3, 1);
            W1 << 0.333333333333333, 0.333333333333333, 0.333333333333333;
            appendMatricesF(X, X1);
            appendMatricesF(W, W1);
        }
        else if (p == 3) {
            MatrixXd X1(4, 3);
            X1.row(0) << 0.333333333333333, 0.333333333333333, 0.333333333333333;
            X1.row(1) << 0.200000000000000, 0.200000000000000, 0.600000000000000;
            X1.row(2) << 0.200000000000000, 0.600000000000000, 0.200000000000000;
            X1.row(3) << 0.600000000000000, 0.200000000000000, 0.200000000000000;
            MatrixXd W1(4, 1);
            W1 << -0.281250000000000, 0.260416666666667, 0.260416666666667, 0.260416666666667;
            W1 = W1 * 2;
            appendMatricesF(X, X1);
            appendMatricesF(W, W1);
        }
        
    }
    return std::make_pair(X, W);
  
}
