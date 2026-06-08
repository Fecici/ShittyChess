#ifndef LEGAL_HEADER
#define LEGAL_HEADER


#include "definitions.h"
#include "move.h"
#include "history.h"

bool isLegalMove(Board* b, Move move);

uint64_t getLegalFromPseudo(Board* b, uint64_t pesudoMoves, Square src);

int handleMakeMove(Board* b, Move move);

void generate_moves(Board* b, Move* move_list);  // generates pseudo moves, then filters legal moves in search and perft. we can change this later if we want to generate legal moves directly, but this is easier for now and the performance difference should be negligible since we are already generating a lot of moves in perft and search and the legality check is not that expensive.

#endif