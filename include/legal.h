#ifndef LEGAL_HEADER
#define LEGAL_HEADER


#include "definitions.h"
#include "move.h"
#include "history.h"

bool isLegalMove(Board* b, Move move);

uint64_t getLegalFromPseudo(Board* b, uint64_t pesudoMoves, Square src);


#endif