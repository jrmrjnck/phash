/*
 * Jonathan Doman
 * jonathan.doman@gmail.com
 */

#ifndef HASHMAP_H
#define HASHMAP_H

#include <cstdint>
#include <atomic>
#include <random>
#include <memory>
#include <cstring>

class HashMap
{
public:
   struct Slot
   {
      std::atomic<uint32_t> key;
      std::atomic<uint32_t> value;
   };

public:
   HashMap();
   ~HashMap();

   void set( uint32_t key, uint32_t value );
   bool get( uint32_t key, uint32_t* value = nullptr );
   void remove( uint32_t key );

   int size() const { return _table.load()->size; }

private:
   class Table;
   enum SetMode
   {
      MatchVal,
      MatchAll,
      MatchAnyNotNull
   };
   uint32_t _set( Table& table, 
                  uint32_t key, 
                  uint32_t value, 
                  SetMode mode = MatchAll, 
                  uint32_t expectedVal = 0 );
   uint32_t _get( Table& table, uint32_t key, uint32_t hash );
   uint32_t _hash( uint32_t key ) const;
   Table*   _resize( Table& table );
   void     _helpCopy();
   void     _copySlotAndCheck( Table& table, int index, bool doExtraHelp = false );
   void     _copyCheckAndPromote( Table& table, int workDone );
   bool     _copySlot( int index, Table& oldTable, Table& newTable );

private:
   class Table
   {
   public:
      Table( int capacity )
       : capacity(capacity),
         size(0),
         slots(0),
         newTable(nullptr),
         copyIdx(0),
         copyDone(0),
         resizers(0),
         ref(0)
      {
         _arr = new Slot[capacity];
         memset( _arr, 0, capacity*sizeof(Slot) );
      }

      ~Table()
      {
         delete [] _arr;
      }

      const Slot& operator[]( int idx ) const { return _arr[idx]; }
            Slot& operator[]( int idx )       { return _arr[idx]; }

   public:
      const int           capacity;
      std::atomic<int>    size;
      std::atomic<int>    slots;

      std::atomic<Table*> newTable;
      std::atomic<int>    copyIdx;
      std::atomic<int>    copyDone;
      std::atomic<int>    resizers;

      std::atomic<int>    ref;

   private:
      Slot* _arr;
   };

   std::atomic<Table*> _table;
   std::vector<Table*> _oldTables;

   uint32_t _a;
   uint32_t _b;
   uint32_t _p;

   std::mt19937 _gen;
   std::uniform_int_distribution<uint32_t> _rand;
};

#endif // !HASHMAP_H
