/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */

#include "kernel.h"
#include "Util.h"
#include "HashMap.h"

#include <iostream>
#include <cassert>

using namespace std;

namespace
{
// 
Key*   _keys;
Value* _values;
int _inputSize;
dim3 _grid;
dim3 _block;

__device__ HashMap<Key,Value> _table;

// Initialize table
__global__ void initTable( int inputSize )
{
   _table.init();
   _table.resize( inputSize );
}

// Kernel to insert items
__global__ void insertItem( Key* keys, Value* values )
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   _table.insert( keys[tid], values[tid] );
}

// Kernel to query an item
__global__ void queryItem( Key* keys )
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   Value v = _table.query( keys[tid] );
}
}

// Copy host data into device side arrays
void copyData( int N, Key* keys, Value* values )
{
   _inputSize = N;
   size_t keySize = N * sizeof(Key);
   size_t valSize = N * sizeof(Value);
   cudaMalloc( &_keys, keySize );
   cudaMalloc( &_values, valSize );
   cudaMemcpy( _keys, keys, keySize, cudaMemcpyHostToDevice );
   cudaMemcpy( _values, values, valSize, cudaMemcpyHostToDevice );

   initTable <<<1,1>>> ( _inputSize );

   cudaDeviceSynchronize();
}

// Allocate one thread to insert each input item
void constructTable()
{
   // Calculate reasonable grid/block dimensions
   const int maxBlockSize = 64;
   _grid  = dim3( 1, 1, 1 );
   _block = dim3( 1, 1, 1 );
   if( _inputSize < maxBlockSize )
   {
      _block.x = _inputSize;
   }
   else
   {
      _block.x = maxBlockSize;
      _grid.x  = _inputSize / maxBlockSize;
      assert( _inputSize % maxBlockSize == 0 );
   }

   insertItem <<<_grid,_block>>> ( _keys, _values );

   cudaDeviceSynchronize();
}

void queryTable()
{
   queryItem <<<_grid,_block>>> ( _keys );
}
