/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */

#include "Util.h"

#include <iostream>

using namespace std;

bool errorOccurred( cudaError_t err, const char* operation )
{
   if( err != cudaSuccess )
   {
      cout << "Error during " << operation << ": " << err << endl;
      return true;
   }
   return false;
}
