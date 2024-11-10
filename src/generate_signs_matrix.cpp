#include "generate_signs_matrix.h"
#include <Eigen/Dense>

Eigen::MatrixXi generateSignsMatrix(const Eigen::MatrixXi& elems2nodes, const Eigen::MatrixXi& order) {
    Eigen::MatrixXi signs2matrix(elems2nodes.rows(), order.rows());
    for (size_t k = 0; k < order.rows(); k++) {
        Eigen::MatrixXi tmp = elems2nodes.col(order(k, 0)) - elems2nodes.col(order(k, 1));
        signs2matrix.col(k) = tmp.array() / tmp.array().abs();
    }
    return signs2matrix;
}