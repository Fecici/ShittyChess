#include "history.h"

Undo* getUndoFromMove(Board* b, Move move) {

    (void) b;
    (void) move;
    void* u;

    return (Undo*) u;
}


void performUndo(Board* b, Undo* undo) {
    (void) b; (void) undo;
}

void performMove(Board* b, Move move) {
    
    (void) b; 
    (void) move;

}

// commands:
void handleUndo(Board* b, Undo* undo);
void handleMakeMove(Board* b, Move move);
void handlePerft(Board* b);
void handleChildren(Board* b);
void handleResign(Board* b);
