#ifndef GENERATE_EDGES_OR_FACES_H
#define GENERATE_EDGES_OR_FACES_H

#include <Eigen/Core>
#include <utility>
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::pair<MatrixXi, MatrixXi> generateEdgesOrFaces(const MatrixXi& elems2nodes, const MatrixXi& order);

#endif // GENERATE_EDGES_OR_FACES_H
