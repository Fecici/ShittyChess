#ifndef HISTORY_HEADER
#define HISTORY_HEADER

#include "definitions.h"

typedef struct {

    uint64_t hashHistory[MAX_PLY];
    Move moveHistory[MAX_PLY];
    Undo undoHistory[MAX_PLY];

    
} History;

Undo* getUndoFromMove(Board* b, Move move);

void performUndo(Board* b, Undo* undo);
void performMove(Board* b, Move move);

// commands:
void handleUndo(Board* b, Undo* undo);
void handleMakeMove(Board* b, Move move);
void handlePerft(Board* b);
void handleChildren(Board* b);
void handleResign(Board* b);

#endif
