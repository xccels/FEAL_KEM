#include "get_gaussian_quadrature.h"
#include <vector>
#include <set>
#include <algorithm>
#include <Eigen/Dense>
#include <igl/sort.h>
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


std::pair<std::vector<double>, double> s4(double w) {
    std::vector<double> X = { 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0 };
    double W = w;
    return std::make_pair(X, W * 1 / 6);
}

std::pair<MatrixXd, MatrixXd> s31(double a, double w) {
    MatrixXd X;
    MatrixXd W;

    std::vector<double> baryc = { a, a, a, 1 - 3 * a };
    sort(baryc.begin(), baryc.end());
    // Generate all unique 3-dimensional permutations from baryc
    std::vector<std::vector<double>> permutations;
    do {
        permutations.push_back({ baryc[0], baryc[1], baryc[2], baryc[3] });
    } while (std::next_permutation(baryc.begin(), baryc.end()));

    // Remove duplicates
    std::set<std::vector<double>> uniquePerms(permutations.begin(), permutations.end());
    X.resize(uniquePerms.size(), 3);
    W.resize(uniquePerms.size(), 1);

    int row = 0;
    for (const auto& perm : uniquePerms) {
        for (int col = 0; col < 3; ++col) {
            X(row, col) = perm[col];
        }
        W(row, 0) = w;
        row++;
    }
    return std::make_pair(X, W*1/6);
}

std::pair<MatrixXd, MatrixXd> s22(double a, double w) {
    MatrixXd X;
    MatrixXd W;

    std::vector<double> baryc = { a, a, 0.5 - a, 0.5 - a };
    sort(baryc.begin(), baryc.end());
    
    std::vector<std::vector<double>> permutations;
    do {
        permutations.push_back({ baryc[0], baryc[1], baryc[2], baryc[3] });
    } while (std::next_permutation(baryc.begin(), baryc.end()));

    // Remove duplicates
    std::set<std::vector<double>> uniquePerms(permutations.begin(), permutations.end());
    
    X.resize(uniquePerms.size(), 3);
    W.resize(uniquePerms.size(), 1);

    int row = 0;
    for (const auto& perm : uniquePerms) {
        for (int col = 0; col < 3; ++col) {
            X(row, col) = perm[col];
        }
        W(row, 0) = w;
        row++;
    }
    return std::make_pair(X, W*1/6);
}


void appendMatrices(MatrixXd& dest, MatrixXd& src) {
    if (dest.size() == 0) {
        dest = src;
    }
    else {
        dest.conservativeResize(dest.rows() + src.rows(), Eigen::NoChange);
        dest.bottomRows(src.rows()) = src;
    }
}

std::pair<MatrixXd, MatrixXd> GaussianQuadrature::getGaussianQuadrature(int p, int dim) {
    MatrixXd X, W;

    if (dim == 2) {
        if (p == 1) {
            MatrixXd X1; X1 << 0.3, 0.3, 0;
            MatrixXd W1; W1 << 0.5;
            appendMatrices(X, X1);
            appendMatrices(W, W1);
        }
        else if (p == 2) {
            MatrixXd X1(3, 3); 
            X1.row(0) << 0.166666666666667, 0.666666666666667, 0;
            X1.row(1) << 0.666666666666667, 0.166666666666667, 0;
            X1.row(2) << 0.166666666666667, 0.166666666666667, 0;
            MatrixXd W1(3,1);
            W1 << 0.166666666666667, 0.166666666666667, 0.166666666666667;
            appendMatrices(X, X1);
            appendMatrices(W, W1);
        }
        else if (p == 3) {
            MatrixXd X1(4, 3);
            X1.row(0) << 0.333333333333333, 0.333333333333333, 0;
            X1.row(1) << 0.200000000000000, 0.600000000000000, 0;
            X1.row(2) << 0.600000000000000, 0.200000000000000, 0;
            X1.row(3) << 0.200000000000000, 0.200000000000000, 0;
            MatrixXd W1(4, 1);
            W1 << -0.281250000000000, 0.260416666666667, 0.260416666666667, 0.260416666666667;
            appendMatrices(X, X1);
            appendMatrices(W, W1);
        }
        
    }

    if (dim == 3) {
        if (p == 4) {
            p = 5;
        }
        else if (p > 14) {
            p = 14;
        }

        if (p == 1) {
            double w = 1.0;
            std::pair<std::vector<double>, double> result = s4(w);
            X = MatrixXd::Map(result.first.data(), 1, result.first.size());
            W = MatrixXd::Constant(1, 1, result.second);
        }
        else if (p == 2) {
            double w = 0.25;
            double a = 0.1381966011250105151795413165634361;
            std::pair<MatrixXd, MatrixXd> result = s31(a, w);
            X = result.first;
            W = result.second;
        }

        else if (p == 3) {
            double w1 = 0.1385279665118621423236176983756412;
            double a1 = 0.3280546967114266473358058199811974;
            std::pair<MatrixXd, MatrixXd> result1 = s31(a1, w1);
            MatrixXd X1 = result1.first;
            MatrixXd W1 = result1.second;
            double w2 = 0.1114720334881378576763823016243588;
            double a2 = 0.1069522739329306827717020415706165;
            std::pair<MatrixXd, MatrixXd> result2 = s31(a2, w2);
            MatrixXd X2 = result2.first;
            MatrixXd W2 = result2.second;

            appendMatrices(X, X1);
            appendMatrices(W, W1);
            appendMatrices(X, X2);
            appendMatrices(W, W2);
        }
        else if (p == 5) {
            double w1 = 0.1126879257180158507991856523332863;
            double a1 = 0.3108859192633006097973457337634578;
            std::pair<MatrixXd, MatrixXd> result1 = s31(a1, w1);
            MatrixXd X1 = result1.first;
            MatrixXd W1 = result1.second;

            double w2 = 0.0734930431163619495437102054863275;
            double a2 = 0.0927352503108912264023239137370306;
            std::pair<MatrixXd, MatrixXd> result2 = s31(a2, w2);
            MatrixXd X2 = result2.first;
            MatrixXd W2 = result2.second;

            double w3 = 0.0425460207770814664380694281202574;
            double a3 = 0.0455037041256496494918805262793394;
            std::pair<MatrixXd, MatrixXd> result3 = s22(a3, w3);
            MatrixXd X3 = result3.first;
            MatrixXd W3 = result3.second;

            appendMatrices(X, X1);
            appendMatrices(W, W1);
            appendMatrices(X, X2);
            appendMatrices(W, W2);
            appendMatrices(X, X3);
            appendMatrices(W, W3);
        }
    }
    return std::make_pair(X, W);
}
