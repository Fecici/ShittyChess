#ifndef PRINT_UTILS_HEADER
#define PRINT_UTILS_HEADER

#include "definitions.h"
#include "bitUtils.h"
#include "history.h"
#include "eval.h"
#include "legal.h"

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


void printLegalMoves(Board* b, bool makeSquare);
void printPseudoLegalMoves(Board* b, bool printSquare);
void printLegalMovesFromSquare(Board* b, Square src, bool printSquare);
void printLegalMovesForColour(Board* b, Colour colour, bool printSquare);
void printLegalMovesForPiece(Board* b, Piece piece, bool printSquare);


// Command stuff
void printHelp();
void printHistory(Undo64* undoStack, unsigned int ply);
void printEval(Board* b);  // eval will be written somewhere else, this is a printing wrapper
void printAttacksFromSquare(Board* b, Square sq);
void printPinsBitboards(Board* b);
void printCheckersBitboards(Board* b);

static const char* const squareChar[64] = {
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"
};

void debug_kingGen();
void debug_knightGen();

#endif
