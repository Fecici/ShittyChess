#include "movegen.h"


uint64_t generateWhitePawnMoves(Board* b) {
    // this is so fucking ugly, but it works. maybe there is some beauty to be found in it that eludes me. maybe not.
    // also, this is really slow, but it works, and movegen is not the bottleneck right now, so im not going to optimize it until later when i know where the bottlenecks are. 
    // also also, this is really really really bad for move ordering, but it works, and move ordering is not the bottleneck right now, so im not going to optimize it until later when i know where the bottlenecks are. 

    uint64_t moves = 0;

    uint64_t whitePawns = b->bitboards[iWP];
    uint64_t emptySquares = ~b->boardUnions[2];  // all pieces

    // single pushes
    uint64_t singlePushes = (whitePawns << 8) & emptySquares;
    moves |= singlePushes;

    // double pushes
    uint64_t doublePushes = ((singlePushes & rank3) << 8) & emptySquares;
    moves |= doublePushes;

    // captures
    uint64_t leftCaptures = (whitePawns << 7) & b->boardUnions[1] & ~fileH;  // capture to the left (from white's perspective)
    uint64_t rightCaptures = (whitePawns << 9) & b->boardUnions[1] & ~fileA;  // capture to the right (from white's perspective)
    moves |= leftCaptures | rightCaptures;

    // en passant captures
    Square epSquare = getEnPassant(b->gamestate);
    if (epSquare) {
        uint64_t epMask = squareBitboards[epSquare];
        uint64_t leftEPCaptures = (whitePawns << 7) & epMask & ~fileH;  // en passant capture to the left (from white's perspective)
        uint64_t rightEPCaptures = (whitePawns << 9) & epMask & ~fileA;  // en passant capture to the right (from white's perspective)
        moves |= leftEPCaptures | rightEPCaptures;
    }

    return moves;
}


uint64_t generateWhiteKnightMoves(Board* b);
uint64_t generateWhiteBishopMoves(Board* b);
uint64_t generateWhiteRookMoves(Board* b);
uint64_t generateWhiteQueenMoves(Board* b);
uint64_t generateWhiteKingMoves(Board* b);

uint64_t generateBlackPawnMoves(Board* b);
uint64_t generateBlackKnightMoves(Board* b);
uint64_t generateBlackBishopMoves(Board* b);
uint64_t generateBlackRookMoves(Board* b);
uint64_t generateBlackQueenMoves(Board* b);
uint64_t generateBlackKingMoves(Board* b);