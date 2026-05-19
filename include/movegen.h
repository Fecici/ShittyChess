#ifndef MOVEGEN_HEADER
#define MOVEGEN_HEADER


#include "definitions.h"
#include "bitUtils.h"


// strictly movegen stuff for each piece. 
uint64_t generateWhitePawnMoves(Board* b);
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

// indexed by getBitboardIndex(piece) (eg, iWP)
(uint64_t*) pieceGenerator(Board* b)[12] = {
    generateWhitePawnMoves,
    generateWhiteKnightMoves,
    generateWhiteBishopMoves,
    generateWhiteRookMoves,
    generateWhiteQueenMoves,
    generateWhiteKingMoves,

    generateBlackPawnMoves,
    generateBlackKnightMoves,
    generateBlackBishopMoves,
    generateBlackRookMoves,
    generateBlackQueenMoves,
    generateBlackKingMoves
};

#endif