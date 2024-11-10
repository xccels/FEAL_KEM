#include "generate_boundary_faces.h"
#include <igl/sort.h>
#include <igl/find.h>
#include <igl/slice.h>

using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Function to compute boundary faces to nodes
std::pair<MatrixXi, MatrixXi> generateBoundaryFacesToNodes(
    const MatrixXi& elems2faces, const MatrixXi& faces2nodes) {

    int nFaces = elems2faces.maxCoeff() + 1;
    MatrixXi A, I;
    igl::sort(elems2faces, 1, true, A, I);
    
    MatrixXi E = MatrixXi::Ones(nFaces, elems2faces.cols() * 2) * -1;
    for (size_t j = 0; j < A.rows(); j++) {
        for (size_t k = 0; k < A.cols(); k++) {
            E(A(j, k), k) = I(j, k);
        }
    }
    MatrixXi ID = ((A.bottomRows(A.rows() - 1).array() - A.topRows(A.rows() - 1).array()) == 0).cast<int>();
    for (int j = 0; j < ID.cols(); ++j) {
        for (int k = 0; k < ID.rows(); ++k) {
            if (ID(k, j) == 1) {
                E(A(k, j), j + 4) = 0;
            }
        }
    }
    igl::sort(E, 2, false, E);
    Eigen::VectorXi ind(E.rows());
    ind.setZero();
    for (size_t j = 0; j < E.rows(); ++j) {
        if (E(j, 1) == -1 && E(j, 0) != -1) {
            ind(j) = 1;
        }
    }
    MatrixXi ind_nnz;
    
    igl::find(ind, ind_nnz);
    MatrixXi bfaces2nodes, bfaces2elem, bfaces2elems;
    bfaces2nodes = igl::slice(faces2nodes, ind_nnz, 1);
    bfaces2elem = igl::slice(E, ind_nnz, 1); 
    bfaces2elems = bfaces2elem.col(0);
    
    return std::make_pair(bfaces2nodes, bfaces2elems);

}