#ifndef MOVE_HEADER
#define MOVE_HEADER

#include "definitions.h"
#include "bitUtils.h"

void performMove(Board* b, Move move);

void handleMakeMove(Board* b, Move move);
Move getMoveFromHex(char* hexStr);
bool isValidMove(Board* b, Move move);


#endif