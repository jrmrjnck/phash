#include <iostream>

using namespace std;

__global__ void kernel( int* a )
{
   int tid = threadIdx.x;
   a[tid] *= 2;
}

bool errorOccurred( cudaError_t err, std::string operation )
{
   if( err != cudaSuccess )
   {
      cout << "Error during " << operation << ": " << err << endl;
      return true;
   }
   return false;
}

#define CHECK_ERROR(err,op) do{if(errorOccurred(err,op)) return;}while(false)

void kernelWrapper( int* a, int N )
{
   int* deviceKeys;
   size_t size = N * sizeof(int);
   cudaError_t err = cudaMalloc( &deviceKeys, size );
   CHECK_ERROR(err,"malloc");
   err = cudaMemcpy( deviceKeys, a, size, cudaMemcpyHostToDevice );
   CHECK_ERROR(err,"memcpy to");

   kernel<<<1,N>>>( deviceKeys );

   err = cudaMemcpy( a, deviceKeys, size, cudaMemcpyDeviceToHost );
   CHECK_ERROR(err,"memcpy from");
   err = cudaFree( deviceKeys );
   CHECK_ERROR(err,"free");
}
