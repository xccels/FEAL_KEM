#include "generate_boundary_normal.h"
#include <math.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
using VectorXi = Eigen::VectorXi;
using VectorXd = Eigen::VectorXd;
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

VectorXd generateBoundaryNormalVector(const MatrixXd& curEV, const MatrixXd& curFV, const int& option) {

    Eigen::Vector3d v1 = curFV.row(1) - curFV.row(0);
    Eigen::Vector3d v2 = curFV.row(2) - curFV.row(0);

    VectorXd n = v1.cross(v2);
    
    // Find outward normal vector
    VectorXd curCV = (curEV.colwise().sum())/4;
    VectorXd curFCV = curFV.colwise().mean();
    
    //VectorXd v3 = curCV - curFV.row(0);
    VectorXd v3 = curFCV - curCV;
    VectorXd normal;

    if (option == 0) { // option=0 : outer boundary element
        if (n.dot(v3) < 0) {
            normal = n * -1;
        }
        else {
            normal = n;
        }
    }

    if (option == 1) { // option=1 : inner boundary element
        if (n.dot(v3) >= 0) {
            normal = n;
        }
        else {
            normal = n * -1;
        }
    }
       
    return normal;
}

    