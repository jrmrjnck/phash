/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */
#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer
{
public:
   typedef std::chrono::high_resolution_clock ClockType;

private:
   std::chrono::time_point<ClockType> _start;
   ClockType::duration                _duration;
   
   bool _running;

public:
   Timer();

   void start();
   void stop();
   void reset();

   ClockType::duration duration();

   template<typename Rep = std::chrono::milliseconds>
   double elapsed()
   {
      return std::chrono::duration_cast<Rep>(duration()).count();
   }
};

#endif // !TIMER_H
