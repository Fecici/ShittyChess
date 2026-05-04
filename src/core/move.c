#include "move.h"


void printMove(Move move) {
    // debug

    printf("Move: %x\n", move);
    printf("Src: %d\n", getSrc(move));
    printf("Dst: %d\n", getDst(move));
    printf("Promotion: %d\n", getPromotion(move));
    printf("En Passant: %d\n", getEnPassant(move));
    printf("Castled: %d\n", isCastled(move));
    printf("Captured Piece Code: %d\n", getCapturedPieceCode(move));
    printf("Captured Piece Type: %d\n", getCapturedType(move));
    printf("Captured Piece Colour: %d\n", getCapturedColour(move));

}

Move getMoveFromNotation(Board* b, char* moveStr) {

    // we assume valid notation at this point

    Move m;

    return m;
}

Move getMoveFromHex(char* hexStr) {


    Move m = (Move) strtol(hexStr, NULL, 0);
    Board b = {0};  // test board
    if (!isValidMove(&b, m)) {
        fprintf(stderr, "Invalid move hex: %s\n", hexStr);
        return 0;
    }
    return m;
}

void performMove(Board* b, Move move);
void handleMakeMove(Board* b, Move move);

// for now, play, check king, unmove
bool isValidMove(Board* b, Move move);

