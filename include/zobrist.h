#ifndef ZOBRIST_HEADER
#define ZOBRIST_HEADER


#include "definitions.h"
#include "bitUtils.h"

uint64_t generateZobristHash(Board* b);

uint64_t getZobristHash(Piece piece, Square square);
uint64_t getZobristCastleHash(uint8_t castlingRights);
uint64_t getZobristEnPassantHash(Square epFile);
uint64_t getZobristBlackToMoveHash();

// each square and each piece, 1 to indicate colour to move, 4 for castleing rights, 8 for ep square: 12*64 + 1 + 4*4 + 8 = 793
void initZobrist();  // index by enum (multiplication). piece type gives 64*(pieceType - 1) + square. last 13 elements are reserved for the above.
                     // the -1 comes from the fact that empty is defined as 0.



#endif