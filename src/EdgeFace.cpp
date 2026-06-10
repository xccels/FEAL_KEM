#include <iostream>
#include <Eigen/Dense>
#include <igl/sort.h>
#include <igl/unique_rows.h>
#include <igl/boundary_facets.h>
#include <igl/ismember.h>
#include <igl/setdiff.h>

using namespace std;
using Eigen::seq;
using Eigen::MatrixXi; using Eigen::MatrixXd; using Eigen::VectorXi;
using Eigen::VectorXd; using Eigen::ArrayXXd;

std::pair<MatrixXi, MatrixXi> Edge_Numbering(MatrixXi Tri)
{
	int num5tri = Tri.size() / 4;
	MatrixXi edge1, edge2, edge3, edge4, edge5, edge6;
	edge1 = Tri(seq(0, num5tri - 1), { 0, 1 });
	edge2 = Tri(seq(0, num5tri - 1), { 0, 2 });
	edge3 = Tri(seq(0, num5tri - 1), { 0, 3 });
	edge4 = Tri(seq(0, num5tri - 1), { 1, 2 });
	edge5 = Tri(seq(0, num5tri - 1), { 2, 3 });
	edge6 = Tri(seq(0, num5tri - 1), { 3, 1 });

	MatrixXi edges_gat = MatrixXi(num5tri * 6, 2);
	edges_gat(seq(0, num5tri * 6 - 1, 6), { 0,1 }) = edge1;
	edges_gat(seq(1, num5tri * 6 - 1, 6), { 0,1 }) = edge2;
	edges_gat(seq(2, num5tri * 6 - 1, 6), { 0,1 }) = edge3;
	edges_gat(seq(3, num5tri * 6 - 1, 6), { 0,1 }) = edge4;
	edges_gat(seq(4, num5tri * 6 - 1, 6), { 0,1 }) = edge5;
	edges_gat(seq(5, num5tri * 6 - 1, 6), { 0,1 }) = edge6;

	MatrixXi edges_sort, Je;
	igl::sort(edges_gat, 2, true, edges_sort, Je);

	MatrixXi edges_unique;
	VectorXi IAe, ICe;
	igl::unique_rows(edges_sort, edges_unique, IAe, ICe);

	MatrixXi edges2nodes, elems2edges;

	edges2nodes = edges_gat(IAe, { 0, 1 });
	elems2edges = ICe.reshaped(6, num5tri).transpose();
	int num5edge = edges2nodes.size() / 2;

	return std::make_pair(edges2nodes, elems2edges);
}

std::pair<MatrixXi, MatrixXi> Face_Numbering(MatrixXi Tri)
{
	int num5tri = Tri.size() / 4;
	MatrixXi face1, face2, face3, face4;
	face1 = Tri(seq(0, num5tri - 1), { 0, 1, 2 });
	face2 = Tri(seq(0, num5tri - 1), { 0, 1, 3 });
	face3 = Tri(seq(0, num5tri - 1), { 0, 2, 3 });
	face4 = Tri(seq(0, num5tri - 1), { 1, 2, 3 });
	MatrixXi faces_gat = MatrixXi(num5tri * 4, 3);

	faces_gat(seq(0, num5tri * 4 - 1, 4), { 0, 1, 2 }) = face1;
	faces_gat(seq(1, num5tri * 4 - 1, 4), { 0, 1, 2 }) = face2;
	faces_gat(seq(2, num5tri * 4 - 1, 4), { 0, 1, 2 }) = face3;
	faces_gat(seq(3, num5tri * 4 - 1, 4), { 0, 1, 2 }) = face4;

	MatrixXi faces_sort, Jf;
	igl::sort(faces_gat, 2, true, faces_sort, Jf);

	MatrixXi faces_unique;
	VectorXi IAf, ICf;
	igl::unique_rows(faces_sort, faces_unique, IAf, ICf);

	MatrixXi faces2nodes, elems2faces;
	faces2nodes = faces_gat(IAf, { 0, 1, 2 });
	elems2faces = ICf.reshaped(4, num5tri).transpose();
	int num5face = faces2nodes.size() / 3;

	return make_pair(faces2nodes, elems2faces);
}
std::pair<VectorXi, MatrixXi> BDRY_FACE(MatrixXi Tri, MatrixXi faces2nodes)
{
	VectorXi ind2bfaces;
	MatrixXi bdry5faces;

	MatrixXi Bface;
	igl::boundary_facets(Tri, Bface);
	int num5bface = Bface.size() / 3;

	MatrixXi face_s, bface_s;
	igl::sort(faces2nodes, 2, true, face_s);
	igl::sort(Bface, 2, true, bface_s);

	MatrixXi face_union(face_s.size() / 3 + bface_s.size() / 3, 3);
	face_union(seq(0, face_s.size() / 3 - 1), { 0, 1, 2 }) = face_s;
	face_union(seq(face_s.size() / 3, face_s.size() / 3 + bface_s.size() / 3 - 1), { 0, 1, 2 }) = bface_s;

	VectorXi IFA, IFC;
	MatrixXi face_dummy;

	igl::unique_rows(face_union, face_dummy, IFA, IFC);
	ind2bfaces = IFC(seq(face_s.size() / 3, face_union.size() / 3 - 1));
	bdry5faces = faces2nodes(ind2bfaces, { 0, 1, 2 });

	return std::make_pair(ind2bfaces, bdry5faces);
}
std::pair<VectorXi, MatrixXi> BDRY_EDGE(MatrixXi bdry5faces, MatrixXi edges2nodes)
{
	int num5bface = bdry5faces.size() / 3;
	MatrixXi bedge1, bedge2, bedge3;
	bedge1 = bdry5faces(seq(0, num5bface - 1), { 0, 1 });
	bedge2 = bdry5faces(seq(0, num5bface - 1), { 1, 2 });
	bedge3 = bdry5faces(seq(0, num5bface - 1), { 2, 0 });

	MatrixXi bedges_gat(num5bface * 3, 2);
	bedges_gat(seq(0, num5bface * 3 - 1, 3), { 0, 1 }) = bedge1;
	bedges_gat(seq(1, num5bface * 3 - 1, 3), { 0, 1 }) = bedge2;
	bedges_gat(seq(2, num5bface * 3 - 1, 3), { 0, 1 }) = bedge3;

	MatrixXi bedges_sort, Jbe;
	igl::sort(bedges_gat, 2, true, bedges_sort, Jbe);

	MatrixXi bedges_unique;
	VectorXi IAbe, ICbe;
	igl::unique_rows(bedges_sort, bedges_unique, IAbe, ICbe);

	MatrixXi bdry5edges = bedges_gat(IAbe, { 0, 1 });

	MatrixXi edges_s, bedges_s;
	igl::sort(edges2nodes, 2, true, edges_s);
	igl::sort(bdry5edges, 2, true, bedges_s);

	MatrixXi edge_union(edges_s.size() / 2 + bedges_s.size() / 2, 2);
	edge_union(seq(0, edges_s.size() / 2 - 1), { 0, 1 }) = edges_s;
	edge_union(seq(edges_s.size() / 2, edges_s.size() / 2 + bedges_s.size() / 2 - 1), { 0, 1 }) = bedges_s;

	VectorXi IAebe, ICebe;
	MatrixXi edge_dummy;
	igl::unique_rows(edge_union, edge_dummy, IAebe, ICebe);
	VectorXi ind2bedges = ICebe(seq(edges_s.size() / 2, edge_union.size() / 2 - 1));

	return std::make_pair(ind2bedges, bdry5edges);
}

std::pair<VectorXi, VectorXi> ABSINC(MatrixXi faces2nodes, MatrixXi ABC, MatrixXi PEC)
{
	MatrixXi face_s, abc_s, pec_s;
	igl::sort(faces2nodes, 2, true, face_s);
	igl::sort(ABC, 2, true, abc_s);
	igl::sort(PEC, 2, true, pec_s);

	MatrixXi face_union_A(face_s.size() / 3 + abc_s.size() / 3, 3);
	face_union_A(seq(0, face_s.size() / 3 - 1), { 0, 1, 2 }) = face_s;
	face_union_A(seq(face_s.size() / 3, face_s.size() / 3 + abc_s.size() / 3 - 1), { 0, 1, 2 }) = abc_s;

	MatrixXi face_union_P(face_s.size() / 3 + pec_s.size() / 3, 3);
	face_union_P(seq(0, face_s.size() / 3 - 1), { 0, 1, 2 }) = face_s;
	face_union_P(seq(face_s.size() / 3, face_s.size() / 3 + pec_s.size() / 3 - 1), { 0, 1, 2 }) = pec_s;

	VectorXi IAA, IAC, IPA, IPC;
	MatrixXi face_dummy_a, face_dummy_p;

	igl::unique_rows(face_union_A, face_dummy_a, IAA, IAC);
	VectorXi ind5abs_tmp = IAC(seq(face_s.size() / 3, face_union_A.size() / 3 - 1));
	VectorXi ind5abs;
	igl::sort(ind5abs_tmp, 1, true, ind5abs);

	igl::unique_rows(face_union_P, face_dummy_p, IPA, IPC);
	VectorXi ind5inc_tmp = IPC(seq(face_s.size() / 3, face_union_P.size() / 3 - 1));
	VectorXi ind5inc;
	igl::sort(ind5inc_tmp, 1, true, ind5inc);

	return make_pair(ind5abs, ind5inc);
}

