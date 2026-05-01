#ifndef PRINT_UTILS_HEADER
#define PRINT_UTILS_HEADER

#include "definitions.h"
#include "bitUtils.h"
#include "history.h"

void printGameState(Board* b);
void printBoard(Board* b);
void printBitboards(Board* b);
void printBitBoard(uint64_t bitboard, char* name, bool makeSquare);
void printZobrist(Board* b);
void printColour(Board* b);

// Command stuff
void printHelp();
void printLegalMoves(Board* b);
void printHistory(History* h);
void printEval(Board* b);  // eval will be written somewhere else, this is a printing wrapper
void printAttacksFromSquare(Board* b, Square sq);
void printPinsBitboards(Board* b);
void printCheckersBitboards(Board* b);

#endif
