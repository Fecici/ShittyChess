#ifndef ZOBRIST_HEADER
#define ZOBRIST_HEADER


#include "definitions.h"

typedef enum {

    square  // just so nothing complains for now. need to do this
    
} ZobristIndex;

uint64_t generateZobristHash(Board* b);

uint64_t generateRandom();  // each square and each piece, 1 to indicate colour to move, 4 for castleing rights, 8 for ep square: 12*64 + 1 + 4 + 8 = 781
void initZobristArray(uint64_t arr[781]);  // index by enum (multiplication). piece type gives 64*(pieceType - 1) + square. last 13 elements are reserved for the above.
                                           // the -1 comes from the fact that empty is defined as 0.

uint64_t __DEBUG_randomHash();  // reproducibility - still need to figure this out

#endif