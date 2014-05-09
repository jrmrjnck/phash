/*
 * Jonathan Doman
 * jonathan.doman@gmail.cpm
 */

#include "HashMap.h"

#include <cuda.h>

#include <cassert>

__device__
void HashMap::init( int inputSize, Slot* table, uint32_t* params, int numParams )
{
   _size = 0;
   _capacity = inputSize / LOAD_FACTOR;
   _table = table;

   assert( numParams >= 2 );
   _params = params;
   _numParams = numParams - 1;
   _a = _params[_numParams--];
   _b = _params[_numParams--];
   _p = 4294967291U; // Largest unsigned 32-bit prime number
}

__device__
void HashMap::deinit()
{
   delete [] _table;
}

__device__ 
bool HashMap::insert( Key key, Value value, Value* oldVal )
{
   Slot newEntry = makeSlot( key, value );
   uint32_t index = _hash( key ) % _capacity;

   for( int i = 1; i <= MAX_PROBES; ++i )
   {
      Slot oldEntry = atomicCAS( reinterpret_cast<unsigned long long*>(_table+index), 
                                 static_cast<unsigned long long>(NULL_SLOT), 
                                 static_cast<unsigned long long>(newEntry) );

      if( oldEntry == NULL_SLOT )
      {
         return true;
      }

      // Quadratic jump
      index = (index + i*i) % _capacity;
   }

   return false;
}

__device__
bool HashMap::query( Key key, Value& value )
{
   uint32_t index = _hash( key ) % _capacity;

   for( int i = 1; i <= MAX_PROBES; ++i )
   {
      Slot entry = _table[index];
      Key k = slotKey( entry );
      if( k == key )
      {
         value = slotValue( entry );
         return true;
      }
      if( k == NULL_KEY )
         return false;

      index = (index + i*i) % _capacity;
   }

   return false;
}

__device__
uint32_t HashMap::_hash( Key key )
{
   return ((static_cast<uint64_t>(_a)*key + _b) % _p);
}
