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
   random_device rd;
   default_random_engine gen( rd() );
   uniform_int_distribution<int> dis( numeric_limits<int>::min(), numeric_limits<int>::max() );

   const int N = 1 << 4;

   vector<int> keys( N );
   for( int& k : keys )
   {
      k = dis( gen );
      cout << k << " ";
   }
   cout << endl << endl;

   kernelWrapper( keys.data(), N );

   for( int k : keys )
   {
      cout << k << " ";
   }
   cout << endl;

   return 0;
}
