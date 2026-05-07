#ifndef SEARCH_HEADER
#define SEARCH_HEADER


#include "bitUtils.h"
#include "definitions.h"
#include "moves.h"

int perft(Board* b, int depth);

// cmds
void handlePerft(Board* b);
void handleChildren(Board* b);

#endif