#include "eval.h"


int evaluateBoard(Board* b) {
    // for now, just a simple material count. positive for white, negative for black. this is not a good eval, but it serves as a placeholder for now.

    int eval = 0;  // int for faster eval

    int black_flip = 1;
    int offset = 0;

    if (isBlackToMove(b->gamestate)) {
        black_flip = -1;
        offset = 6;
    }

    // piece values

    for (int i = 0; i < 6; i++) {
        uint64_t bitboard = b->bitboards[i + offset];
        long val = black_flip * __builtin_popcountll(bitboard) * victim_value[i];
        eval += val;
    }

    return eval;
}