#ifndef GENERATE_BOUNDARY_EDGES_H
#define GENERATE_BOUNDARY_EDGES_H

#include <Eigen/Dense>
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

MatrixXi generateBoundaryEdgesToEdges(    const MatrixXi& bfaces2nodes, const MatrixXi& edges2nodes);

#endif  // GENERATE_BOUNDARY_EDGES_H