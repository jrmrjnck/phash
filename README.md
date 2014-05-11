phash - Parallel Hash Tables
============================

Compilation
-----------
The supplied Makefile can be used to build the software in both debug mode
(`make debug`) and optimized mode (`make release`). A symlink is created next
to the Makefile that points to the generated binary.

The CUDA hash table and C++ hash table are compiled into a single driver
executable.  Therefore, the Makefile has to know the location of nvcc and g++
(>= 4.7), so change the variables CXX and NVCC to the correct names if needed.
Also adjust the CUDA\_FLAGS variable to use the best compute capability for
your GPU.

Usage
-----
`./phash <2^N> <CUDA|ShMem> <Linear|Quad|Cuckoo> <iter> <queryIter> <shMemThreads>`

All arguments are integers.

1. log2 of the number of items to generate for testing.
2. 0 = CUDA, 1 = ShMem
3. 0 = Linear Probing, 1 = Quadratic Probing, 2 = Cuckoo Hashing
4. Num of test iterations to run. Results are averaged into a single insertion
   rate and query rate. (Report used 10)
5. Number of times to query for each item in each test. (Report used 250)
6. Number of threads to use for C++ hash table.
