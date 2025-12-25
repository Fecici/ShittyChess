#ifndef PRINT_UTILS_HEADER
#define PRINT_UTILS_HEADER

#include "definitions.h"
#include "bitUtil.h"

void printGameState(Board* b);
void printBoard(Board* b);
void printBitboards(Board* b);
void printBitBoard(uint64_t bitboard, char* name, bool makeSquare);
void printZobrist(Board* b);
void printColour(Board* b);



#endif