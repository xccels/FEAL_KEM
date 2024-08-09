Installation                        {#install_page}
============

[TOC]
# Downloads
The repository can be cloned as follows:

# Compile
## Prerequisites
Prerequisites to compile PaScaL_TDMA are as follows:
* MPI (OpenMPI, IntelMPI, MPICH, .. and so on)
* Fortran compiler (NVIDIA HPC SDK, Intel fortran, GNU fortran, .. and so on )
  * ([`nvfortran`](https://developer.nvidia.com/hpc-sdk-downloads) for GPU runs, NVIDIA HPC SDK 21.1 or higher)

### Option1: NVIDIA HPC SDK + Open MPI
[![](https://img.shields.io/badge/HPC_SDK-21.1_or_higher-nvidia.svg?logo=nvidia)](https://developer.nvidia.com/hpc-sdk-downloads)
  ![](https://img.shields.io/badge/Tested-HPC_SDK_24.3_with_CUDA_12.3-silver.svg?logo=cachet)
* The NVIDIA HPC SDK includes a pre-compiled version of OpenMPI.
* Change `24.3` to `Your SDK version`
  
  ```shell
  cd <your-path>/PaScaL_TDMA
  export TDMA_PATH=$(pwd)
  ```

* Test examples
  ```shell 
  cd examples
  nvmpirun -np 2 ex1_single.out
  ```
___
### Compile and build
* Build PaScaL_TDMA
    ```
	make lib
	```
* Build an example problem after build PaScaL_TDMA

    ```
	make example
	```

* Build all

  ```
	make all
	```

* Clean build
  ```shell
  export TDMA_PATH=$(pwd)
  make clean
  ```

# Running the example
After building the example file, an executable binary, `*.out`, 
```
mpirun -np 8 ./a.out ./PARA_INPUT.inp
```

# Folder structure
* `src` : source files of PaScaL_TDMA 2.0.
* `doc` : documentation


<div class="section_buttons">

| Previous          |                              Next |
|:------------------|----------------------------------:|
| [Introduction](index.html) | [Performance](perf_page.html) |
</div>