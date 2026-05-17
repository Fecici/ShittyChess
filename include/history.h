#ifndef HISTORY_HEADER
#define HISTORY_HEADER

#include "definitions.h"
#include "move.h"

//Undo* getUndoFromMove(Board* b, Move move);

void performUndo(Board* b, Undo64 undo);

//bool pushUndoToStack(Board* b, Undo* undo);     // old
bool pushUndo64ToStack(Board* b, Undo64 undo);  // for speed

// commands:
int handleUndo(Board* b, Undo64 undo);
int handleMakeMove(Board* b, Move move);

void handlePerft(Board* b);
void handleChildren(Board* b);
#endif
