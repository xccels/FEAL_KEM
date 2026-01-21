#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <Eigen/Dense>
#include <mpi.h>
#include <igl/ismember.h> 
#include <igl/find.h>
#include <igl/cat.h>

using namespace std;
using Eigen::seq;
using Eigen::MatrixXi; using Eigen::VectorXi; using Eigen::ArrayXXi;
using Eigen::MatrixXd; using Eigen::VectorXd; using Eigen::ArrayXXd;
using Eigen::ArrayXi;

tuple<VectorXi, VectorXi, VectorXi> DUAL_PRIMAL(ArrayXi Esum)
{
	Eigen::Array<bool, Eigen::Dynamic, 1> boolcor = (Esum >= 3); int num5cor = boolcor.count();
	Eigen::Array<bool, Eigen::Dynamic, 1> booldul = (Esum == 2); int num5inter = booldul.count();
	VectorXi Ec, IAec, ICec;
	igl::find(boolcor, Ec, IAec, ICec);
	VectorXi Eb, IAeb, ICeb;
	igl::find(booldul, Eb, IAeb, ICeb);
	VectorXi num5glo(2); num5glo(0) = num5cor; num5glo(1) = num5inter;

	return make_tuple(Ec, Eb, num5glo);
}

VectorXi Bool_Bcs(MatrixXi info4edge, VectorXi Ec, int n5i, int n5b, int n5c, int num5part)
{
	VectorXi Ecs = info4edge(seq(n5i + n5b, n5i + n5b + n5c - 1), num5part + 3);
	VectorXi ETAec, Bcs;
	igl::ismember(Ecs, Ec, ETAec, Bcs);
	return Bcs;
}

pair<VectorXi, VectorXi> Bool_Qs(int num5part, int n5b, int prank)
{
	VectorXi num5dual(num5part);
	MPI_Allgather(&n5b, 1, MPI_INT, num5dual.data(), 1, MPI_INT, MPI_COMM_WORLD);
	int a = num5dual(seq(0, prank), 0).sum() - n5b; int b = num5dual(seq(0, prank), 0).sum() - 1;
	VectorXi Qs = VectorXi::LinSpaced(n5b, a, b);

	return make_pair(Qs, num5dual);
}

pair<VectorXi, int> Bool_Tsq(MatrixXi info4edge, int num5part, int n5i, int n5b, int ik)
{
	MatrixXi Eb_info = info4edge(seq(n5i, n5i + n5b - 1), { ik + 2, num5part + 3 });
	VectorXi Esq_ind, ed1, ed2;
	igl::find(Eb_info(seq(0, n5b - 1), 0), Esq_ind, ed1, ed2);
	int num5sq = Esq_ind.size();

	return make_pair(Esq_ind, num5sq);
}
