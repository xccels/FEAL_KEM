#include "generate_boundary_edges.h"
#include "generate_edges_or_faces.h"
#include <igl/sortrows.h>
#include <igl/find.h>
#include <igl/slice.h>

MatrixXi generateBoundaryEdgesToEdges(const MatrixXi& bfaces2nodes, const MatrixXi& edges2nodes) {

    MatrixXi orderBEdges(3, 2);
    orderBEdges << 0, 1, 1, 2, 2, 0;
    
    auto bedgesResult = generateEdgesOrFaces(bfaces2nodes, orderBEdges);
    MatrixXi bedges2nodes = bedgesResult.second;

    MatrixXi matrix(edges2nodes.rows() + bedges2nodes.rows(), edges2nodes.cols());
    matrix << edges2nodes, bedges2nodes;

    igl::sort(matrix, 2, true, matrix);
    //std::cout << "\n matrix :\n" << matrix << std::endl;
    MatrixXi tags, matrixs;
    igl::sortrows(matrix, true, matrixs, tags);

    MatrixXi diff = (matrixs.bottomRows(matrixs.rows() - 1).array() - matrixs.topRows(matrixs.rows() - 1).array()).cast<int>();
    for (int j = 0; j < diff.rows(); j++) {
        for (int k = 0; k < diff.cols(); k++) {
            if (diff(j, k) == 0) {
                diff(j, k) = 1;
            }
            else
                diff(j, k) = 0;
        }
    }

    //std::cout << "\n tags :\n" << tags << std::endl;


    Eigen::VectorXi k = diff.rowwise().all();
    //std::cout << "\n k :\n" << k << std::endl;
    Eigen::VectorXi k_nnz;
    igl::find(k, k_nnz);
    //std::cout << "\n k_nnz :\n" << k_nnz << std::endl;
    Eigen::VectorXi ones = Eigen::VectorXi::Ones(k_nnz.size());
    Eigen::VectorXi kp = k_nnz + ones;
    //std::cout << "\n kp :\n" << kp << std::endl;
    MatrixXi tags_kp = igl::slice(tags, kp, 1);
    //std::cout << "\n tags_kp :\n" << tags_kp << std::endl;
    MatrixXi tags2;
    igl::sort(tags_kp, 1, true, tags_kp, tags2);
    //std::cout << "\n tags_kp :\n" << tags_kp << std::endl;
    //std::cout << "\n tags2 :\n" << tags2 << std::endl;
    MatrixXi k_tags2 = igl::slice(k_nnz, tags2, 1);
    MatrixXi bedges2edges = igl::slice(tags, k_tags2);

    return bedges2edges;
}