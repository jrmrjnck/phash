/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */
#include "Timer.h"

using namespace std;

Timer::Timer()
{
   reset();
}

void Timer::start()
{
   _running = true;
   _start = ClockType::now();
}

void Timer::stop()
{
   _running = false;
   auto diff = ClockType::now() - _start;
   _duration += diff;
}

void Timer::reset()
{
   _running = false;
   _duration = ClockType::duration::zero();
}

Timer::ClockType::duration Timer::duration()
{
   if( !_running )
      return _duration;

   stop();
   auto copy = _duration;
   start();
   return copy;
}
