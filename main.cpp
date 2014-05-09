/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */

#include "kernel.h"
#include "Timer.h"
#include "HashMap.h"

#include <random>
#include <limits>
#include <vector>
#include <algorithm>
#include <iostream>
#include <thread>
#include <stdexcept>
#include <cassert>
#include <iomanip>

using namespace std;

namespace
{
double lap( Timer& timer, const char* event = NULL )
{
   double time = timer.elapsed<chrono::milliseconds>();
   if( event != NULL )
      cout << event << ": " << time << " ms" << endl;
   timer.reset();
   timer.start();
   return time;
}

enum HashType
{
   LinearProbe,
   QuadraticProbe,
   Cuckoo
};

void runCuda( int N, 
              HashType type, 
              int nQuery, 
              double& constTime, 
              double& queryTime )
{
   if( type == LinearProbe )
      throw logic_error( "CUDA implementation does not support linear probing" );

   Timer timer;
   timer.start();

   // Uniform distribution with mersenne twister PRNG
   random_device rd;
   mt19937 gen( rd() );
   uniform_int_distribution<Key> keyGen( 1 );
   uniform_int_distribution<Value> valGen( 1 );
   uniform_int_distribution<uint32_t> hashFnGen( 1 );

   // Generate random key-value pairs
   vector<Key> keys(N);
   vector<Value> values(N);
   for( int i = 0; i < N; ++i )
   {
      keys[i] = keyGen( gen );
      values[i] = valGen( gen );
   }

   // Generate random numbers to use for the hash function
   const int numParams = 20;
   vector<uint32_t> hashParams( numParams );
   for( auto& a : hashParams )
   {
      a = hashFnGen( gen );
   }

   bool cuckoo = type == Cuckoo;
   copyData( cuckoo, N, keys.data(), values.data(), hashParams.data(), numParams );
   lap( timer, "Host to Device data copy" );

   constructTable( cuckoo );
   constTime += lap( timer );

   queryTable( cuckoo, nQuery );
   queryTime += lap( timer );

   tearDown();
   lap( timer, "Table tear down" );
}

void constructTable( HashMap& table,
                     std::vector<HashMap::Slot>::const_iterator begin,
                     std::vector<HashMap::Slot>::const_iterator end )
{
   auto it = begin;
   while( it < end )
   {
      table.set( it->key, it->value );
      
      ++it;
   }
}

void queryTable( HashMap& table,
                 std::vector<HashMap::Slot>::const_iterator begin,
                 std::vector<HashMap::Slot>::const_iterator end,
                 int times = 1 )
{
   while( times-- )
   {
      auto it = begin;
      while( it < end )
      {
         uint32_t v;
         bool result = table.get( it->key, &v );
         assert( result );
         ++it;
      }
   }
}

void runSharedMem( int N, 
                   HashType type, 
                   int nQuery, 
                   int nThreads, 
                   double& constTime, 
                   double& queryTime )
{
   if( type == Cuckoo )
      throw logic_error( "Shared memory implementation does not support cuckoo hashing" );

   // Set up PRNG (don't allow the special values max-{0,1})
   random_device rd;
   mt19937 gen( rd() );
   uniform_int_distribution<uint32_t> keyDist( 1, numeric_limits<uint32_t>::max()-1 );
   uniform_int_distribution<uint32_t> valDist( 1, numeric_limits<uint32_t>::max()/2-1 );

   // Generate random key-value pairs
   vector<HashMap::Slot> items( N );
   for( auto& s : items )
   {
      s.key = keyDist( gen );
      s.value = valDist( gen );
   }

   // Convert hash type
   HashMap::HashType hashType;
   if( type == LinearProbe )
      hashType = HashMap::LinearProbe;
   else
      hashType = HashMap::QuadraticProbe;

   HashMap table( hashType );
   Timer timer;
   timer.start();

   // Launch threads
   vector<thread> threads( nThreads );
   auto it = items.begin();
   unsigned int nPerThread = items.size() / nThreads;
   assert( nPerThread*nThreads == items.size() );

   for( auto& t : threads )
   {
      t = thread( constructTable, ref(table), it, it+nPerThread );
      it += nPerThread;
   }

   for( auto& t : threads )
      t.join();
   constTime += lap( timer );

   it = items.begin();
   for( auto& t : threads )
   {
      t = thread( queryTable, ref(table), it, it+nPerThread, nQuery );
      it += nPerThread;
   }

   for( auto& t : threads )
      t.join();
   queryTime += lap( timer );
}
}

int main( int argc, char* argv[] )
{
   if( argc == 1 )
   {
      cout << "Usage: <2^N> <CUDA|ShMem> <Linear|Quad|Cuckoo> <iter> <queryIter> <smMemThreads>" << endl;
      return 0;
   }

   int exp = 10;
   if( argc > 1 )
   {
      exp = atoi( argv[1] );
   }
   const int N = 1 << exp;

   bool shMem = false;
   if( argc > 2 )
   {
      shMem = atoi(argv[2]) != 0;
   }

   HashType type = LinearProbe;
   if( argc > 3 )
   {
      type = static_cast<HashType>(atoi(argv[3]));
   }

   int iterations = 1;
   if( argc > 4 )
   {
      iterations = atoi( argv[4] );
   }

   int nQuery = 20;
   if( argc > 5 )
   {
      nQuery = atoi( argv[5] );
   }

   int nThreads = thread::hardware_concurrency();
   if( argc > 6 )
   {
      if( !shMem )
         cout << "nThreads ignored for CUDA" << endl;
      nThreads = atoi( argv[6] );
   }

   if( shMem )
      cout << nThreads << "-thread ";

   cout << (shMem ? "Shared Memory" : "CUDA")
        << " Hash Table using "
        << (type == LinearProbe ? "Linear Probing" : (type == QuadraticProbe ? "Quadratic Probing" : "Cuckoo Hashing")) << endl
        << "2^" << exp << " Item Test" << endl
        << "Times averaged over " << iterations << " runs" << endl
        << "Each key queried " << nQuery << " times" << endl << endl;

   double constTime = 0.0;
   double queryTime = 0.0;

   for( int i = 0; i < iterations; ++i )
   {
      if( shMem )
         runSharedMem( N, type, nQuery, nThreads, constTime, queryTime );
      else
         runCuda( N, type, nQuery, constTime, queryTime );
   }

   constTime /= iterations;
   queryTime /= iterations;
   queryTime /= nQuery;

   cout << endl
        << "Average construction time: " << setw(10) << constTime << " ms, rate: " << setw(12) << N*1000LL/constTime << " insertion/s" << endl
        << "Average total query time:  " << setw(10) << queryTime << " ms, rate: " << setw(12) << N*1000LL/queryTime << " query/s" << endl;

   return 0;
}
