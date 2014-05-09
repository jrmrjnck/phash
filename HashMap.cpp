/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */

#include "HashMap.h"

#include <cassert>
#include <climits>
#include <chrono>
#include <thread>
#include <type_traits>

using namespace std;

namespace
{
const int DEFAULT_TABLE_CAPACITY = 1 << 10;
const int MAX_PROBES = 16;
const uint32_t NULL_KEY = 0;
const uint32_t DEAD_KEY = numeric_limits<uint32_t>::max();
const uint32_t NULL_VALUE = NULL_KEY;
const uint32_t DEAD_VALUE = DEAD_KEY / 2;
const float GROWTH_FACTOR = 2.0;
const uint32_t PRIME_BIT = 1 << (CHAR_BIT*sizeof(uint32_t)-1);
const uint32_t DEADPRIME_VALUE = DEAD_VALUE | PRIME_BIT;
bool isPrime( uint32_t v )
{
   return (v & PRIME_BIT) != 0;
}
}

HashMap::HashMap( HashType type )
 : _table(new Table(DEFAULT_TABLE_CAPACITY)),
   _type(type)
{
   // Initialize hash function parameters
   random_device rd;
   _gen.seed( rd() );
   _a = _rand( _gen );
   _b = _rand( _gen );
   _p = 4294967291U; // Largest unsigned 32-bit prime number
}

HashMap::~HashMap()
{
   delete _table;
   _table = nullptr;
   for( Table* ot : _oldTables )
      delete ot;
   _oldTables.clear();
}

void HashMap::set( uint32_t key, uint32_t value )
{
   _set( *_table, key, value );
}

bool HashMap::get( uint32_t key, uint32_t* value )
{
   auto v = _get( *_table, key, _hash(key) );
   if( v == NULL_VALUE )
      return false;
   if( value != nullptr )
      *value = v;
   return true;
}

void HashMap::remove( uint32_t key )
{
   _set( *_table, key, DEAD_VALUE );
}

uint32_t HashMap::_set( Table& table, 
                        uint32_t key, 
                        uint32_t value, 
                        SetMode mode,
                        uint32_t expectedVal )
{
   assert( value != NULL_VALUE );
   assert( !isPrime(value) );
   assert( !isPrime(expectedVal) );

   auto index = _hash( key ) % table.capacity;

   uint32_t k, v;
   int probeCount = 0;
   Table* newTable = nullptr;

   // Spin until we can claim a key
   while( true )
   {
      // Load current slot
      v = table[index].value.load( memory_order_relaxed );
      k = table[index].key.load( memory_order_relaxed );

      // Try to claim an empty slot
      if( k == NULL_KEY )
      {
         if( value == DEAD_VALUE )
            return value;

         if( table[index].key.compare_exchange_strong(k,key) )
         {
            ++table.slots;
            break;
         }

         assert( k != NULL_KEY );
      }

      // Cliff Click requires a volatile read at this point in order to
      // ensure that the key-body is written before he tries to read it.
      // Since our keys are just a word, I don't think the fence here is
      // necessary, but I'll include it anyway.
      newTable = table.newTable.load( memory_order_seq_cst );

      // Requires a full key-object comparison in CC's implementation.
      if( k == key )
         break;

      // End of the road, go to new table
      if( ++probeCount > _probeLimit(table.capacity) 
          || key == DEAD_KEY ) // FIXME: k or key here?
      {
         newTable = _resize( table );
         if( mode != MatchVal ||  expectedVal != NULL_VALUE )
            _helpCopy();
         return _set( *newTable, key, value, mode, expectedVal );
      }

      if( _type == QuadraticProbe )
         index = (index + probeCount*probeCount) % table.capacity;
      else
         index = (index + 1) % table.capacity;
   }

   if( v == value )
      return v;

   // Test to see if we want to move to a new table
   if( newTable == nullptr
       && ((v == NULL_VALUE && probeCount > _probeLimit(table.capacity))
           || isPrime(v)) )
   {
      newTable = _resize( table );
   }

   // It we're moving to new table, copy our slot and retry
   if( newTable != nullptr )
   {
      _copySlotAndCheck( table, index, true );
      return _set( *newTable, key, value, mode, expectedVal );
   }

   // Update the existing table
   while( true )
   {
      assert( !isPrime(v) );

      // No idea, copied from CC's NBHM
      if( mode != MatchAll
          && v != expectedVal
          && (mode != MatchAnyNotNull || v == DEAD_VALUE || v == NULL_VALUE)
          && !(v == NULL_VALUE && expectedVal == DEAD_VALUE)
          && (expectedVal == NULL_VALUE || expectedVal != v) )
         return v;

      if( table[index].value.compare_exchange_strong(v,value) )
      {
            if( v == NULL_VALUE || v == DEAD_VALUE )
            {
               if( value != DEAD_VALUE )
                  ++table.size;
            }
            else if( value == DEAD_VALUE )
               --table.size;

         bool expNotNull = !(mode == MatchVal && expectedVal == NULL_VALUE);
         return (v == NULL_VALUE && expNotNull) ? DEAD_VALUE : v;
      }

      if( isPrime(v) )
      {
         _copySlotAndCheck( table, index, true );
         return _set( *newTable, key, value, mode, expectedVal );
      }
   }
}

uint32_t HashMap::_get( Table& table, uint32_t key, uint32_t hash )
{
   int index = hash % table.capacity;

   int probeCount = 0;
   while( true )
   {
      const uint32_t k = table[index].key.load( memory_order_relaxed );
      const uint32_t v = table[index].value.load( memory_order_relaxed );

      if( k == NULL_KEY )
         return NULL_VALUE;

      Table* const newTable = table.newTable.load( memory_order_seq_cst );

      if( k == key )
      {
         if( !isPrime(v) )
            return (v == DEAD_KEY) ? NULL_KEY : v;

         assert( newTable != nullptr );
         _copySlotAndCheck( table, index, true );
         return _get( *newTable, key, hash );
      }

      if( ++probeCount > _probeLimit(table.capacity) || key == DEAD_KEY )
      {
         if( newTable == nullptr )
            return NULL_KEY;
         
         _helpCopy();
         return _get( *newTable, key, hash );
      }

      if( _type == QuadraticProbe )
         index = (index + probeCount*probeCount) % table.capacity;
      else
         index = (index + 1) % table.capacity;
   }
   return false;
}

uint32_t HashMap::_hash( uint32_t key ) const
{
   return ((_a*key + _b) % _p);
}

HashMap::Table* HashMap::_resize( Table& table )
{
   // Check if another thread started resize
   Table* newTable = table.newTable.load( memory_order_seq_cst );
   if( newTable != nullptr )
      return newTable;

   // Start our own resize
   int oldLen  = table.capacity;
   int newLen  = oldLen * GROWTH_FACTOR;

   int r = table.resizers.load( memory_order_relaxed );
   while( !table.resizers.compare_exchange_weak(r,r+1) )
      r = table.resizers.load( memory_order_relaxed );

   // If there are already two resizers, sleep for a while to give them
   // a chance to finish before we try to allocate
   if( r >= 2 )
   {
      newTable = table.newTable.load( memory_order_seq_cst );
      if( newTable != nullptr )
         return newTable;

      int megs = (newLen*sizeof(Slot)) / (1024*1024);
      auto time = chrono::milliseconds(8*megs);
      this_thread::sleep_for( time );
   }

   // Check one more time to avoid the allocation
   newTable = table.newTable.load( memory_order_seq_cst );
   if( newTable != nullptr )
      return newTable;

   newTable = new Table( newLen );

   atomic_thread_fence( memory_order_release );

   Table* null = nullptr;
   if( !table.newTable.compare_exchange_strong(null,newTable) )
      delete newTable;

   return table.newTable.load( memory_order_relaxed );
}

// Help along an existing resize operation
void HashMap::_helpCopy()
{
   // No copy in progress if the next table is null
   Table& topTable = *_table.load( memory_order_relaxed );
   Table* newTable = topTable.newTable.load( memory_order_relaxed );
   if( newTable == nullptr )
      return;

   int oldLen = topTable.capacity;
   const int workToDo = min( oldLen, DEFAULT_TABLE_CAPACITY );

   int panicStart = -1;
   int copyIdx = 0;

   while( topTable.copyDone < oldLen )
   {
      // Panic if we've tried twice to copy every slot and it still has
      // not happened.
      if( panicStart == -1 )
      {
         copyIdx = topTable.copyIdx;
         while( copyIdx < oldLen*2
                && !topTable.copyIdx.compare_exchange_weak(copyIdx,copyIdx+workToDo) );

         // PANIC!
         if( copyIdx >= oldLen*2 )
            panicStart = copyIdx;
      }

      // Try to copy
      int workDone = 0;
      for( int i = 0; i < workToDo; ++i )
      {
         if( _copySlot((copyIdx+i)%oldLen,topTable,*newTable) )
            ++workDone;
      }

      if( workDone > 0 )
         _copyCheckAndPromote( topTable, workDone );

      copyIdx += workToDo;

      if( panicStart == -1 )
         return;
   }

   _copyCheckAndPromote( topTable, 0 );
}

void HashMap::_copySlotAndCheck( Table& table, int index, bool doExtraHelp )
{
   Table* newTable = table.newTable.load( memory_order_seq_cst );
   assert( newTable != nullptr );
   if( _copySlot(index,table,*newTable) )
      _copyCheckAndPromote( table, 1 );

   if( doExtraHelp )
      _helpCopy();
}

void HashMap::_copyCheckAndPromote( Table& table, int workDone )
{
   int oldLen = table.capacity;
   int copyDone = table.copyDone;
   assert( copyDone + workDone <= oldLen );

   // Record the work we did
   if( workDone > 0 )
   {
      while( !table.copyDone.compare_exchange_weak(copyDone,copyDone+workDone) )
      {
         copyDone = table.copyDone.load( memory_order_relaxed );
         assert( copyDone + workDone <= oldLen );
      }
   }

   // Check whether the top-level copy is finished
   // and swap the old and new tables
   if( copyDone+workDone == oldLen
       && _table.load(memory_order_relaxed) == &table )
   {
      Table* expected = &table;
      if( _table.compare_exchange_strong(expected,table.newTable) )
         _oldTables.push_back( expected );
   }
}

bool HashMap::_copySlot( int index, Table& oldTable, Table& newTable )
{
   uint32_t k = oldTable[index].key;
   while( k == NULL_KEY )
   {
      oldTable[index].key.compare_exchange_strong( k, DEAD_KEY );
   }

   uint32_t v = oldTable[index].value;
   while( !isPrime(v) )
   {
      uint32_t primeV = (v == NULL_VALUE) ? DEADPRIME_VALUE : (v | PRIME_BIT);
      if( oldTable[index].value.compare_exchange_strong(v,primeV) )
      {
         // If old value was dead, don't bother copying to new table
         if( primeV == DEADPRIME_VALUE )
            return true;

         v = primeV;
         break;
      }
      v = oldTable[index].value;
   }

   if( v == DEADPRIME_VALUE )
      return false;

   assert( isPrime(v) );
   uint32_t unPrimed = v & ~PRIME_BIT;
   assert( unPrimed != DEAD_VALUE );
   bool copiedToNew = (_set(newTable,k,unPrimed,MatchVal,NULL_VALUE) == NULL_VALUE);

   while( !oldTable[index].value.compare_exchange_weak(v,DEADPRIME_VALUE) );

   return copiedToNew;
}

int HashMap::_probeLimit( int capacity ) const
{
   // Cliff Click's resize trigger heuristic
   return MAX_PROBES + capacity / 4;
}
