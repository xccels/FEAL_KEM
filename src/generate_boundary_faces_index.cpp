#include "generate_boundary_faces_index.h"
#include <igl/find.h>
#include <igl/sort.h>
#include <igl/ismember_rows.h>

using VectorXi = Eigen::VectorXi;
using VectorXd = Eigen::VectorXd;
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Function to compute boundary faces to nodes
VectorXi generateBoundaryFacesIndex(const MatrixXi& bfaces2elems, const MatrixXi& bfaces2nodes, const MatrixXi& elems2faces, const MatrixXi& faces2nodes){

    int dim = 3;
    int nbFaces = bfaces2elems.rows();
    
    VectorXi FaceIdx(nbFaces);
    for (int j = 0; j < nbFaces; j++) {
        MatrixXi targetF = bfaces2nodes.row(j);
        MatrixXi curF = elems2faces.row(bfaces2elems(j));
        MatrixXi curFV(curF.cols(), dim);
        for (size_t nv = 0; nv < curFV.rows(); nv++) {
            curFV.row(nv) = faces2nodes.row(curF(nv));
        }
        VectorXi IA, IB, ROW, COL, VAL;
        MatrixXi targetF_sort, targetF_sort_idx, curFV_sort, curFV_sort_idx;
        igl::sort(targetF, 2, true, targetF_sort, targetF_sort_idx);
        igl::sort(curFV, 2, true, curFV_sort, curFV_sort_idx);
    /*    std::cout <<"\n-------\n" << targetF << " >> \n" << targetF_sort << "\n\n"
            << curFV << " >>\n " << curFV_sort << "\n";*/
        igl::ismember_rows(curFV_sort, targetF_sort, IA, IB);
        igl::find(IA, ROW, COL, VAL);
        FaceIdx.row(j) << ROW;
    }
    return FaceIdx;
}