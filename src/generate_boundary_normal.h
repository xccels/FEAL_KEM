#ifndef GENERATE_BOUNDARY_NORMAL_H
#define GENERATE_BOUNDARY_NORMAL_H

#include <Eigen/Core>
#include <Eigen/Dense>

using VectorXi = Eigen::VectorXi;
using VectorXd = Eigen::VectorXd;
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Function to compute boundary faces to nodes
VectorXd generateBoundaryNormalVector(const MatrixXd& curEV, const MatrixXd& curFV, const int& option);

#endif // GENERATE_BOUNDARY_NORMAL_H
