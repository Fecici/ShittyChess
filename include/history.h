#ifndef HISTORY_HEADER
#define HISTORY_HEADER

#include "definitions.h"
#include "move.h"

Undo* getUndoFromMove(Board* b, Move move);

void performUndo(Board* b, Undo* undo);

bool pushUndoToStack(Board* b, Undo* undo);

// commands:
int handleUndo(Board* b, Undo* undo);
void handlePerft(Board* b);
void handleChildren(Board* b);
void handleResign(Board* b);
#endif
