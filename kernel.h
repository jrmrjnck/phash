/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */

#ifndef KERNEL_H
#define KERNEL_H

typedef unsigned int Key;
typedef unsigned int Value;

void copyData( int N, Key* keys, Value* values );
void constructTable();

#endif // !KERNEL_H
