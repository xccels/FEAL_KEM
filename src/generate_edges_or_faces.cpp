#include "generate_edges_or_faces.h"
#include "igl/sort.h"
#include "igl/unique_rows.h"
#include "igl/slice.h"

using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::pair<MatrixXi, MatrixXi> generateEdgesOrFaces(const MatrixXi& elems2nodes, const MatrixXi& order) {
    MatrixXi result(order.rows() * elems2nodes.rows(), order.cols());
    for (size_t j = 0; j < elems2nodes.rows(); j++) {
        for (size_t k = 0; k < order.rows(); k++) {
            for (size_t m = 0; m < order.cols(); m++) {
                result(j * order.rows() + k, m) = elems2nodes(j, order(k, m));
            }
        }
    }
    MatrixXi matrixs, dummy;
    Eigen::VectorXi J, I;
    igl::sort(result, 2, true, matrixs);
    igl::unique_rows(matrixs, dummy, J, I);
    MatrixXi target2nodes = igl::slice(result, J, 1);
    
    MatrixXi elems2target(elems2nodes.rows(), order.rows());
    elems2target = Eigen::Map<MatrixXi>(I.data(), elems2nodes.rows(), order.rows());
    return std::make_pair(elems2target, target2nodes);
}