# Fast Feature Search

This repository is part of the course project ES611 Algorithms on Advance Computer Architectures. In this project we have tried to paralyze the feature matching process via openMP and CUDA.

## Getting Started

All the program is written in C++. The serial.cpp file contains serial version of the code and parallel.cpp contains OpenMP implementation. Inside GPU folder, gpuVersion.cu contains CUDA implementation. 

### Prerequisites

Required libraries are following.

```
OpenCV version 2.4.9 or above
CUDA 6.0 or above
```

### Running the code

In the CMakeLists.txt file replace the name of the .cpp file or .cu file and do following steps.


```
cmake .
make
run executable 
```


## Contributing

The some functionality of the code is borrowed from C++ KDTree implementation on Rosetta Code (https://rosettacode.org/wiki/K-d_tree). 


## Authors

* **Ojas Joshi**
* **Nilay Thakor**
* **Rushi Jariwala**

