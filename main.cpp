/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */

#include "kernel.h"

#include <random>
#include <limits>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

int main()
{
   // Uniform distribution with mersenne twister PRNG
   random_device rd;
   mt19937 gen( rd() );
   uniform_int_distribution<Key> keyGen( numeric_limits<Key>::min(), 
                                         numeric_limits<Key>::max() );
   uniform_int_distribution<Value> valGen( numeric_limits<Value>::min(), 
                                           numeric_limits<Value>::max() );

   const int N = 1 << 10;

   vector<Key> keys(N);
   vector<Value> values(N);
   for( int i = 0; i < N; ++i )
   {
      keys[i] = keyGen( gen );
      values[i] = valGen( gen );
   }

   copyData( N, keys.data(), values.data() );

   constructTable();

   return 0;
}
