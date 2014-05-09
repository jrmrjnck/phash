/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */

#include "kernel.h"
#include "Timer.h"

#include <random>
#include <limits>
#include <vector>
#include <algorithm>
#include <iostream>
#include <thread>

using namespace std;

namespace
{
void lap( Timer& timer, const char* event )
{
   cout << event << ": " << timer.elapsed<chrono::microseconds>() << " us" << endl;
   timer.reset();
   timer.start();
}
}

int main( int argc, char* argv[] )
{
   Timer timer;
   timer.start();

   int exp = 10;
   if( argc > 1 )
   {
      exp = atoi( argv[1] );
   }

   bool cuckoo = false;
   if( argc > 2 )
   {
      cuckoo = atoi(argv[2]) != 0;
   }

   // Uniform distribution with mersenne twister PRNG
   random_device rd;
   mt19937 gen( rd() );
   uniform_int_distribution<Key> keyGen( 1 );
   uniform_int_distribution<Value> valGen( 1 );
   uniform_int_distribution<uint32_t> hashFnGen( 1 );

   int N = 1 << exp;

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
   lap( timer, "Preamble" );

   copyData( N, keys.data(), values.data(), hashParams.data(), numParams );
   lap( timer, "Host to Device data copy" );

   constructTable( false );
   auto constTime = timer.elapsed<chrono::microseconds>();
   lap( timer, "Table construction" );

   int nQuery = 20;
   queryTable( false, nQuery );
   auto queryTime = timer.elapsed<chrono::microseconds>();
   lap( timer, "Table query" );

   tearDown();
   lap( timer, "Table tear down" );
   timer.reset();

   cout << endl
        << "Average construction rate: " << N*1000000LL/constTime << " ins/s" << endl
        << "Average query rate: " << N*nQuery*1000000LL/queryTime << " query/s" << endl;

   return 0;
}
