Performance                         {#perf_page}
===========

[TOC]

# Vectorization_nurion


<!-- Vectorization vs. no vectorization -->
<div class="darkmode_inverted_image">\image html perf_vectorization.png width=50%</div>

- A multicore processor-level vectorization application reduces wall-clock time to process a \\(512^3\\) grid by 42% and improves performance by 72%.

- Tested in [CPU environment](perf_page.html#experimental-setup) proposed by [Kim et al. (2021)](reference_page.html)


# CPU Parallel benchmark

@pre 
- Tested in [CPU environment](perf_page.html#experimental-setup) proposed by [Kim et al. (2021)](reference_page.html)

## MPI and OpenMP parallelization

<!-- # MPI and OpenMP parallelization -->
<div class="darkmode_inverted_image">\image html perf_parallel_openmp.png width=50%</div>

- The scalability of parallelization, as exemplified by OpenMP and MPI, is optimal when the number of grids is high (\\(512^3\\)). However, when the grid number is low (\\(64^3\\)), the parallel performance is compromised.


# Experimental Setup {#experimental-setup}           
 <div class="tabbed">

- <b class="tab-title">CPU</b>
    @note
    **[CPU Environment]** <br>
    All the computations were executed on **the Nurion manycore cluster** at the Korea Institute of Science and Technology Information (KISTI). The Nurion consists of **8,305 Cray CS500 nodes** interconnected by the Intel Omni-Path Architecture. Each node has a **68-Core Intel Xeon Phi 7250 processor with 16GB** of high bandwidth on a chip and 96 GB of main memory. Each simulation is repeated for 10 time steps to average any small fluctuations in the execution time. An executable program was built using an Intel compiler (version 19.0.1.144) with the flags of full optimization (-O3) and automatic vectorization according to the hardware architecute (-xMIC-AVX512).<br>
    This environment proposed by [Kim et al. (2021)](reference_page.html)
    
- <b class="tab-title">GPU</b>
    @note
    **[GPU Environment]** <br>
    We evaluated the computational performance and energy efficiency of the GPU implementation of PaScaL_TDMA 2.0 on **the NEURON cluster** at the Korea Institute of Science and Technology Information (KISTI). The cluster consisted of **two AMD EPYC 7543 processors (hosts) and eight NVLnk-connected NVIDIA A100 GPUs (devices) per compute node.** The results were compared with those obtained on the NURION cluster at KISTI, which features an Intel Xeon Phi 7750 Knight Landing (KNL) processor per compute node. Intel OneAPI 22.2 and NVIDIA HPC SDK 22.7 were used to compile PaScaL_TDMA 2.0 on the NURION and NEURON clusters, respectively. <br>
    This environment proposed by [Yang et al. (2023)](reference_page.html)
</div>



<div class="section_buttons">

| Previous          |                              Next |
|:------------------|----------------------------------:|
| [Installation](install_page.html) | [Home](intro_page.html) |
</div>