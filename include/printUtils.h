#ifndef PRINT_UTILS_HEADER
#define PRINT_UTILS_HEADER

#include "definitions.h"
#include "bitUtils.h"
#include "history.h"
#include "eval.h"

void printGameState(Board* b, bool makeSquare);
void printBoard(Board* b);
void printBitboards(Board* b, bool makeSquare);
void printBitBoard(uint64_t bitboard, char* name, bool makeSquare);
void printZobrist(Board* b);
void printColour(Board* b);
void printBitboardHex(uint64_t bitboard, char* name);
char* getPieceNameFromIndex(uint8_t piece);
void printBitboardHexAll(Board* b);
char* getPieceNameFromPiece(Piece piece);


// Command stuff
void printHelp();
void printLegalMoves(Board* b);
void printHistory(Undo64* undoStack, unsigned int ply);
void printEval(Board* b);  // eval will be written somewhere else, this is a printing wrapper
void printAttacksFromSquare(Board* b, Square sq);
void printPinsBitboards(Board* b);
void printCheckersBitboards(Board* b);

#endif
