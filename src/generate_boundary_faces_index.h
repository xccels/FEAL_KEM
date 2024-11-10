#ifndef GENERATE_BOUNDARY_FACES_INDEX_H
#define GENERATE_BOUNDARY_FACES_INDEX_H

#include <Eigen/Core>
#include <igl/find.h>
#include <igl/ismember_rows.h>

using VectorXi = Eigen::VectorXi;
using VectorXd = Eigen::VectorXd;
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Function to compute boundary faces to nodes
VectorXi generateBoundaryFacesIndex(const MatrixXi& bfaces2elems, const MatrixXi& bfaces2nodes, const MatrixXi& elems2faces, const MatrixXi& faces2nodes);

#endif // GENERATE_BOUNDARY_FACES_INDEX_H

