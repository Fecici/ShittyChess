#ifndef ENGINE_HEADER
#define ENGINE_HEADER

#include "bitUtils.h"
#include "definitions.h"
#include "search.h"

typedef struct {

    // eval
    // metadata
    // precomp data
    // ill figure this format out later clearly

} Engine;


Move getEngineMove(Board* b, int depth);
int alphaBetaMax(int alpha, int beta, int depth, Board* b);
int alphaBetaMin(int alpha, int beta, int depth, Board* b);



#endif