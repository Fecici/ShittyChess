#include "eval.h"


int evaluateBoard(Board* b) {
    // for now, just a simple material count. positive for white, negative for black. this is not a good eval, but it serves as a placeholder for now.

    int eval = 0;  // int for faster eval

    // piece values

    for (int i = 0; i < 12; i++) {
        uint64_t bitboard = b->bitboards[i];
        while (bitboard) {
            //int square = __builtin_ctzll(bitboard);  // get index of least significant bit
            eval += (victim_value[i % 6] * (i < 6 ? 1 : -1));  // add value for white pieces, subtract for black pieces
            bitboard &= bitboard - 1;  // clear least significant bit

            // Using __builtin_popcountll(bitboard) * value is simpler and faster than clearing every bit individually
        }
    }

    return eval;
}