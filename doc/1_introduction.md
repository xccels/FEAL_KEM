Introduction                       {#intro_page}
============

[TOC]
# Overview

Na Hyesun

Kang jihoon


This library is for both single and many tridiagonal systems of equations. The main algorithm for a tridiagonal matrix consists of the following five steps: 

- (1) Transform the partitioned submatrices in the tridiagonal systems into modified submatrices:
        Each computing core transforms the partitioned submatrices in the tridiagonal systems of equations into the modified forms by applying modified Thomas algorithm.
- (2) Construct reduced tridiagonal systems from the modified submatrices:
        The reduced tridiagonal systems are constructed by collecting the first and last rows of the modified submatrices from each core using MPI_Ialltoallw.


# Features

## Single and many tridiagonal system
### Single tridiagonal system
<div class="darkmode_inverted_image">\image html intro_2.png width=50%</div>
- A communication method using MPI_Gather for a single diagonal system.
- \\(\texttt{PaScaL_TDMA_plan_single_create}\\): A subroutine to create a type variable of \\(\texttt{plan_single}\\), for a single tridiagonal system

- \\(\texttt{PaScaL_TDMA_plan_single_destroy}\\): A subroutine to dellocate the \\(\texttt{plan_single}\\) and all associated data

- \\(\texttt{PaScaL_TDMA_plan_single_solve}\\): A subroutine for solving a single tridiagonal system with the \\(\texttt{MPI_gather}\\) and \\(\texttt{MPI_scatter}\\) fuctions, as shown in upper figure

### Many tridiagonal system
<div class="darkmode_inverted_image">\image html intro_1.png width=90%</div>
- The present method using MPI_Alltoall for many tridiagonal systems. 
- \\(\texttt{PaScaL_TDMA_plan_many_create}\\):
  A subroutine to create a type variable of \\(\texttt{plan_many}\\), for a many tridiagonal systems. The definition of the derived data type depends on the version of MPI library.

- \\(\texttt{PaScaL_TDMA_plan_many_destroy}\\): A subroutine to dellocate the \\(\texttt{plan_many}\\) and all associated data

- \\(\texttt{PaScaL_TDMA_plan_many_solve}\\): A subroutine for solving a many tridiagonal systems with the \\(\texttt{MPI_Alltoallw}\\) fuction, as shown in upper figure
  
# Authors

- ABC Kim

# Versions
<div class="tabbed">

- <b class="tab-title">1.0</b>
    - Version 1.0
    
- <b class="tab-title">2.0</b>
    - Version 2.0

</div>


# Citation

Please use the following bibtex, when you refer to this project.


## Credits

Thanks for writing this library and inspiring feedback on GitHub!

Special thanks to all the contributors:
<br><br>
<a href="https://github.com/xccels/PaScaL_TDMA/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=xccels/PaScaL_TDMA" />
</a>


<div class="section_buttons">

|                        Read Next |
|---------------------------------:|
| [Installation](install_page.html) |

</div>