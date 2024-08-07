# FEAL_KEM: CPU-based Parallel Library for Electromagnetic Finite Element Analysis

The finite element method (FEM) has the advantage of not being limited by the geometric shape of the analysis structure because it provides various types of elements, and can define material constants for each element. This is a discrete technique essential for numerical analysis of large complex shapes and composite material structures. Unlike the method of moments (MoM), a representative numerical analysis technique that uses only the surface information of the structure when analyzing electromagnetic fields, FEM discretizes the internal/external regions of the structure. In addition, the methodology of calculating the characteristics of unit elements and then combining the characteristics of all elements requires large-scale calculations in the global matrix assembly process involving repeated operations of similar processes. In this study, we developed a numerical library for the MPI-based domain partitioning technique by utilizing METIS, a graph partitioning-based parallel library that divides a large analysis domain into subdomains and assigns operations in the subdomains to each processor.

# Authors
+ Hyesun Na (hyesunna@yonsei.ac.kr), School of Mathematics and Computing(Computational Science and Engineering), Yonsei University (v1.0)

+ Youngmin Shin (youngmm@yonsei.ac.kr), School of Mathematics and Computing(Computational Science and Engineering), Yonsei University (v1.0)

+ Eunjung Lee (eunjunglee@yonsei.ac.kr), School of Mathematics and Computing(Computational Science and Engineering), Yonsei University (v1.0)


# Usage

## Downloading FEAL_KEM
The repository can be cloned as follows:

Alternatively, the source files can be downloaded through github menu 'Download ZIP'.

## Compile
### Prerequisites
Predrequisites to compile FEAL_KEM are as follows:
+ libigl
+ eigen-3.4.0
+ METIS
+ MUMPS
### Complie and build
+ Build FEAL_KEM
+ Build an example problem after guild FEAL_KEM
+ Build all
### Mores on compile option

## Running the example

# Folder structure
+ `src` : source files of FEAL_KEM 1.0
+ `example` : source files on example problem for 3D wave equation
+ `include` : header files are created after building
+ `lib` : static libraries of FEAL_KEM 1.0 are created after building
+ `doc` : documentation
+ `run` : an executable binary file for the example problem is created after building
+ `tool` : contains useful scripts and tools

## Cite
Please use the following bibtex, when you refer to this project

```
@hye{FEAL_KEM,
    title  = "CPU-based Parallel Library for Electromagnetic Finie Element Analysis",
    author = "Na, Hyesun and Shin, Youngmin and Lee, Eunjung",
    url    = "https://github.com/FEAL-Lab/FEAL_KEM",
    year   = "2024"
}
```

## References
For more information, please the reference paper and School of Mathematics and Computing (Computational Science and Engineering)
