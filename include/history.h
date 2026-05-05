#ifndef HISTORY_HEADER
#define HISTORY_HEADER

#include "definitions.h"
#include "move.h"

Undo* getUndoFromMove(Board* b, Move move);

void performUndo(Board* b, Undo* undo);

// commands:
void handleUndo(Board* b, Undo* undo);

#endif
