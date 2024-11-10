#ifndef GENERATE_SIGNS_MATRIX_H
#define GENERATE_SIGNS_MATRIX_H

#include <Eigen/Core>

Eigen::MatrixXi generateSignsMatrix(const Eigen::MatrixXi& elems2nodes, const Eigen::MatrixXi& order);

#endif // GENERATE_SIGNS_MATRIX_H
