#ifndef BIT_UTIL_HEADER
#define BIT_UTIL_HEADER


#include "definitions.h"

#define PIECE_TYPE(p)   ((p) & 7)        // 0..7 (0 = EMPTY, 1..6 valid)
#define PIECE_COLOR(p)  ((p) >> 3)       // 0 = white/empty, 1 = black (EMPTY treated separately)
#define IS_EMPTY(p)     ((p) == 0)
#define IS_BLACK(p)     ((p) & 8)


// function signatures
uint8_t getBitboardIndex(uint8_t piece);
uint8_t getPieceType(uint8_t piece);
uint8_t getPiecesColour(uint8_t piece);
uint8_t getSrc(Move move);
uint8_t getDst(Move move);
uint8_t getEnPassant(Move move);
bool isCastled(Move move);
uint8_t getPromotion(Move move);
uint8_t getCapturedPieceCode(Move move);
uint8_t getCapturedType(Move move);
uint8_t getCapturedColour(Move move);
int     getMoveCount(uint64_t gameState);

#endif