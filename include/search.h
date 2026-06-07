#ifndef SEARCH_HEADER
#define SEARCH_HEADER


#include "bitUtils.h"
#include "definitions.h"
#include "move.h"
#include "movegen.h"
#include "history.h"
#include "legal.h"

uint64_t perft(Board* b, int depth);
void perft_wrapper(Board* b, int depth);
void perft_iterativeDeepening(Board* b, int maxDepth);

// cmds
void handlePerft(Board* b);
void handleChildren(Board* b);

#endif