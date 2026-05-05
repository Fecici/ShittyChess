#ifndef MOVE_HEADER
#define MOVE_HEADER

#include "definitions.h"
#include "bitUtils.h"

void performMove(Board* b, Move move);

void handleMakeMove(Board* b, Move move);
Move getMoveFromHex(char* hexStr);
bool isLegalMove(Board* b, Move move);

Move getMoveFromAlgebra(Board* b, char* moveStr);
Move getMoveFromNotation(Board* b, char* moveStr);
bool validMoveNotation(char* moveStr);
bool validAlgebraicNotation(char* moveStr);



#endif