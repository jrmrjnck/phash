#ifndef HASHMAP_H
#define HASHMAP_H

#define MAX_PROBES 10000

template<typename KeyT, typename ValueT>
class HashMap
{
public:
   struct Slot
   {
      KeyT   key;
      ValueT value;
   };

public:
   __device__ void init()
   {
      _size = 0;
      _table = NULL;
      _loadFactor = 0.9;
   }

   __device__ void deinit()
   {
      delete _table;
   }

   __device__ bool insert( KeyT key, ValueT value )
   {
      unsigned index = _hash( key );

      for( int i = 1; i <= MAX_PROBES; ++i )
      {
         KeyT oldKey = atomicCAS( reinterpret_cast<KeyT*>(_table+index), 0, key );

         if( oldKey == 0 || oldKey == key )
         {
            _table[index].value = value;
            return true;
         }

         // Quadratic jump
         index = (index + i*i) % _size;
      }

      return false;
   }

   __device__ ValueT query( KeyT key )
   {
      unsigned index = _hash( key );

      for( int i = 1; i <= MAX_PROBES; ++i )
      {
      }

      return 0;
   }

   __device__ void resize( int size )
   {
      _size = size / _loadFactor;
      delete _table;
      _table = new Slot[_size];
      memset( _table, 0, _size*sizeof(Slot) );
   }

private:
   __device__ unsigned int _hash( KeyT key )
   {
      return 0;
   }

private:
   int   _size;
   Slot* _table;

   float _loadFactor;
};

#endif // !HASHMAP_H
