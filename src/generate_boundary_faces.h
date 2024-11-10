#ifndef GENERATE_BOUNDARY_FACES_H
#define GENERATE_BOUNDARY_FACES_H

#include <Eigen/Core>
#include <igl/sort.h>
#include <igl/find.h>
#include <igl/slice.h>
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Function to compute boundary faces to nodes
std::pair<MatrixXi, MatrixXi> generateBoundaryFacesToNodes(const MatrixXi& elems2faces, const MatrixXi& faces2nodes);

#endif // GENERATE_BOUNDARY_FACES_H
