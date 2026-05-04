#ifndef HISTORY_HEADER
#define HISTORY_HEADER

#include "definitions.h"
#include "move.h"

typedef struct {

    uint64_t hashHistory[MAX_PLY];
    Move moveHistory[MAX_PLY];
    Undo undoHistory[MAX_PLY];

    
} History;

Undo* getUndoFromMove(Board* b, Move move);

void performUndo(Board* b, Undo* undo);

// commands:
void handleUndo(Board* b, Undo* undo);

#endif
