#ifndef MOVE_HEADER
#define MOVE_HEADER

#include "definitions.h"
#include "bitUtils.h"
#include "zobrist.h"
#include "movegen.h"

void makeMove(Board* b, Move move);
Move getMoveFromHex(char* hexStr);
bool isLegalMove(Board* b, Move move);
static inline Piece getPieceOnSquare(Board* b, Square sq) {
    return b->pieces[sq];
}

Move getMoveFromAlgebra(Board* b, char* moveStr);
Move getMoveFromNotation(Board* b, char* moveStr);
Move getMoveFromSquare(Board* b, Square src, Square dst, bool promo);
bool validMoveNotation(char* moveStr);
bool validAlgebraicNotation(char* moveStr);



#endif