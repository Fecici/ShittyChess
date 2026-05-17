#ifndef MOVE_HEADER
#define MOVE_HEADER

#include "definitions.h"
#include "bitUtils.h"
#include "zobrist.h"

void makeMove(Board* b, Move move);
Move getMoveFromHex(char* hexStr);
bool isLegalMove(Board* b, Move move);
Piece getPieceOnSquare(Board* b, Square sq);

Move getMoveFromAlgebra(Board* b, char* moveStr);
Move getMoveFromNotation(Board* b, char* moveStr);
bool validMoveNotation(char* moveStr);
bool validAlgebraicNotation(char* moveStr);



#endif