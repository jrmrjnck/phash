/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */

#ifndef KERNEL_H
#define KERNEL_H

#include <stdint.h>

typedef uint32_t Key;
typedef uint32_t Value;
typedef uint64_t Slot;

void copyData( int N, Key* keys, Value* values, uint32_t* params, int numParams );
void constructTable( bool cuckoo );
void queryTable( bool cuckoo, int times );
void tearDown();

#endif // !KERNEL_H
