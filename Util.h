/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */
#ifndef UTIL_H
#define UTIL_H

#define CHECK_ERROR(err,op) do{if(errorOccurred(err,op)) return;}while(false)

bool errorOccurred( cudaError_t err, const char* operation );

#endif // !UTIL_H
