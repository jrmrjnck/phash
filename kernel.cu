/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */

#include "kernel.h"

#include <iostream>
#include <cassert>
#include <cstdio>

using namespace std;

namespace
{
const float LOAD_FACTOR = 0.8;
const Key NULL_KEY = 0;
const Slot NULL_SLOT = 0;
const int MAX_PROBES = 100;

// Device side key-value data
Key*      _keys;
Value*    _values;
uint32_t* _params;

Slot* _table;

int  _inputSize;
dim3 _grid;
dim3 _block;

struct TableState
{
   Slot* table;
   int capacity;
   uint32_t* params;
   int paramIdx;
};

TableState* _tableState;

// Slot utility functions
__device__
Slot makeSlot( Key k, Value v )
{
   return (static_cast<Slot>(k) << CHAR_BIT*sizeof(v)) | v;
}
__device__
Key slotKey( Slot s )
{
   return s >> CHAR_BIT*sizeof(Value);
}
__device__
Value slotValue( Slot s )
{
   return s & ((1ULL << CHAR_BIT*sizeof(Value))-1);
}

// Initialize table
__global__
void initTable( TableState* tableState, int capacity, Slot* tableSlots, uint32_t* params, int numParams )
{
   printf( "Initializing table\n" );
   tableState->table    = tableSlots;
   tableState->capacity = capacity;
   tableState->params   = params;
   tableState->paramIdx = numParams - 1;
}

__device__
uint32_t hash( const TableState* ts, Key key, int offset = 0 )
{
   offset *= 2;
   uint64_t a = ts->params[ts->paramIdx-offset];
   uint64_t b = ts->params[ts->paramIdx-offset-1];
   return (a*key + b) % 4294967291U;
}

// Kernel to insert items using quadratic probing
__global__
void quadInsert( TableState* ts, Key* keys, Value* values )
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;

   Slot newEntry = makeSlot( keys[tid], values[tid] );
   uint32_t index = hash( ts, keys[tid] ) % ts->capacity;

   for( int i = 1; i <= MAX_PROBES; ++i )
   {
      // nvcc requires these ridiculous casts even though the types are the same
      Slot oldEntry = atomicCAS( reinterpret_cast<unsigned long long*>(ts->table+index),
                                 static_cast<unsigned long long>(NULL_SLOT),
                                 static_cast<unsigned long long>(newEntry) );

      // The swap was successful
      if( oldEntry == NULL_SLOT )
         return;

      // The swap was unsuccessful
      index = (index + i*i) % ts->capacity;
   }

   // Couldn't find a spot - rehash
   printf( "Insert (%u,%u) failed\n", keys[tid], values[tid] );
}

// Kernel to query an item using quadratic probing
__global__
void quadQuery( TableState* ts, Key* keys, Value* values )
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   Key key = keys[tid];
   uint32_t index = hash( ts, key ) % ts->capacity;

   for( int i = 1; i <= MAX_PROBES; ++i )
   {
      Slot entry = ts->table[index];
      Key k = slotKey( entry );

      if( k == key )
         return;
      if( k == NULL_KEY )
         break;

      index = (index + i*i) % ts->capacity;
   }

   printf( "Query for %u failed\n", key );
}

__global__
void cuckooInsert( TableState* ts, Key* keys, Value* values )
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   Key key = keys[tid];
   Value value = values[tid];
   Slot entry = makeSlot( key, value );

   uint32_t idx = hash( ts, key, 0 ) % ts->capacity;

   for( int i = 0; i <= MAX_PROBES; ++i )
   {
      // Exchange items
      entry = atomicExch( reinterpret_cast<unsigned long long*>(&ts->table[idx]), entry );

      key = slotKey( entry );

      // If the displaced item is null, we're done
      if( key == NULL_KEY )
         return;

      // Otherwise find a new location for the displaced item
      uint32_t idx1 = hash( ts, key, 0 );
      uint32_t idx2 = hash( ts, key, 1 );
      uint32_t idx3 = hash( ts, key, 2 );
      uint32_t idx4 = hash( ts, key, 3 );
           if( idx == idx1 ) idx = idx2;
      else if( idx == idx2 ) idx = idx3;
      else if( idx == idx3 ) idx = idx4;
      else                   idx = idx1;
   }

   printf( "Insert (%u,%u) failed\n", key, value );
}

__global__
void cuckooQuery( TableState* ts, Key* keys, Value* values )
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   Key key = keys[tid];

   // Compute all possible locations
   uint32_t idx[4];
   idx[0] = hash( ts, key, 0 );
   idx[1] = hash( ts, key, 1 );
   idx[2] = hash( ts, key, 2 );
   idx[3] = hash( ts, key, 3 );

   Slot entry;
   for( int i = 0; i < 4; ++i )
   {
      entry = ts->table[idx[i] % ts->capacity];
      Key k = slotKey( entry );
      if( k == key )
         return;
      if( k == NULL_KEY )
         break;
   }

   printf( "Query for %u failed\n", key );
}
}

#define ERROR_CHECK(x,eType,eSuccess,eStrFn)\
   do\
   {\
      eType err = x;\
      if( err != eSuccess )\
      {\
         printf( "Error (%d:%s) at %s:%d\n", err, eStrFn, __FILE__, __LINE__ );\
         return;\
      }\
   } while( false )
#define CUDA_CALL(x) ERROR_CHECK(x,cudaError_t,cudaSuccess,cudaGetErrorString(err))

// Copy host data into device side arrays
void copyData( int N, Key* keys, Value* values, uint32_t* params, int numParams )
{
   _inputSize = N;
   int capacity = N / LOAD_FACTOR;
   size_t keySize = N * sizeof(Key);
   size_t valSize = N * sizeof(Value);
   size_t paramSize = numParams * sizeof(uint32_t);
   size_t tableSize = capacity * sizeof(Slot);
   
   CUDA_CALL(cudaMalloc( &_tableState, sizeof(TableState) ));
   CUDA_CALL(cudaMalloc( &_keys, keySize ));
   CUDA_CALL(cudaMalloc( &_values, valSize ));
   CUDA_CALL(cudaMalloc( &_params, paramSize ));
   CUDA_CALL(cudaMalloc( &_table, tableSize ));
   CUDA_CALL(cudaMemcpy( _keys, keys, keySize, cudaMemcpyHostToDevice ));
   CUDA_CALL(cudaMemcpy( _values, values, valSize, cudaMemcpyHostToDevice ));
   CUDA_CALL(cudaMemcpy( _params, params, paramSize, cudaMemcpyHostToDevice ));
   CUDA_CALL(cudaMemset( _table, 0, tableSize ));

   initTable <<<1,1>>> ( _tableState, capacity, _table, _params, numParams );

   // Synchronize to get accurate timing
   CUDA_CALL(cudaDeviceSynchronize());
}

// Allocate one thread to insert each input item
void constructTable( bool cuckoo )
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

   printf( "Launching %d x %d threads\n", _grid.x, _block.x );

   if( cuckoo )
      cuckooInsert <<<_grid,_block>>> ( _tableState, _keys, _values );
   else
      quadInsert <<<_grid,_block>>> ( _tableState, _keys, _values );

   CUDA_CALL(cudaDeviceSynchronize());
}

void queryTable( bool cuckoo, int times )
{
   if( cuckoo )
   {
      for( int i = 0; i < times; ++i )
         cuckooQuery <<<_grid,_block>>> ( _tableState, _keys, _values );
   }
   else
   {
      for( int i = 0; i < times; ++i )
         quadQuery <<<_grid,_block>>> ( _tableState, _keys, _values );
   }

   CUDA_CALL(cudaDeviceSynchronize());
}

void tearDown()
{
   CUDA_CALL(cudaFree( _keys ));
   CUDA_CALL(cudaFree( _values ));
   CUDA_CALL(cudaFree( _params ));
   CUDA_CALL(cudaFree( _table ));
   CUDA_CALL(cudaFree( _tableState ));
}

