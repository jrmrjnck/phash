#ifndef HASHMAP_H
#define HASHMAP_H

#define MAX_PROBES 100

#include "kernel.h"

#include <cstdio>

class HashMap
{
public:
   __device__ void init( int inputSize, Slot* table, uint32_t* params, int numParams );
   __device__ void deinit();
   __device__ bool insert( Key key, Value value, Value* oldVal = NULL );
   __device__ bool query( Key key, Value& value );

private:
   __device__ uint32_t _hash( Key key );

private:
   int   _size;
   int   _capacity;
   Slot* _table;

   uint32_t _a;
   uint32_t _b;
   uint32_t _p;
   uint32_t* _params;
   int _numParams;
};

#endif // !HASHMAP_H
