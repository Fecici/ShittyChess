#include "movegen.h"

uint64_t precomputedKnightmoves[64];  // indexed by source square, gives bitboard of knight moves from that square
uint64_t precomputedKingMoves[64];  // raw going 1 in each dir

/// TODO: many of these can likely become static inlines. maybe all except kings

uint64_t generateWhitePawnMoves(Board* b, Square src) {
    // assume white's perspective
    // promo will be detected later

    uint64_t moves = 0;

    uint64_t whitePawns = b->bitboards[iWP];
    uint64_t pawn = whitepawns & squareBitboards[src];
    uint64_t emptySquares = ~b->boardUnions[2];  // all pieces

    // single pushes
    uint64_t singlePushes = (pawn << 8) & emptySquares;  // toprank pawns fall off the board (they dont exist anyways)
    moves |= singlePushes;

    // double pushes
    uint64_t doublePushes = ((singlePushes & rank3) << 8) & emptySquares;
    moves |= doublePushes;

    // captures
    uint64_t leftCaptures = (pawn << 7) & b->boardUnions[1] & ~fileH;  // capture to the left (from white's perspective)
    uint64_t rightCaptures = (pawn << 9) & b->boardUnions[1] & ~fileA;  // capture to the right (from white's perspective)
    moves |= leftCaptures | rightCaptures;

    // en passant captures
    Square epSquare = getEnPassant(b->gamestate);
    if (epSquare) {
        uint64_t epMask = squareBitboards[epSquare];
        uint64_t leftEPCaptures = (pawn << 7) & epMask & ~fileH;  // en passant capture to the left (from white's perspective)
        uint64_t rightEPCaptures = (pawn << 9) & epMask & ~fileA;  // en passant capture to the right (from white's perspective)
        moves |= leftEPCaptures | rightEPCaptures;
    }

    return moves;
}


uint64_t generateWhiteKnightMoves(Board* b, Square src) {
    // apply blockers later
    uint64_t moves = 0;

    uint64_t whiteKnight = b->bitboards[iWN] & squareBitboards[src];
    
    moves |= precomputedKnightmoves[src];  // add precomputed moves
    moves &= ~b->boardUnions[0];  // block white pieces
    
    return moves;
}

uint64_t generateWhiteBishopMoves(Board* b, Square src) {
    
    uint64_t moves = 0;
    
    // for now before we add magics, we do this manually
    int i, j;
    getIJFromSquare(src, &i, &j);

    for (int k = 0; k < 4; k++) {
        int x = j;
        int y = i;
        int dx = directions[k][0];
        int dy = directions[k][1];

        while (x + dx >= 0 && x + dx < 8 && y + dy >= 0 && y + dy < 8) {
            Square targetSquare = getSquareIndex(y + dy, x + dx);
            moves |= squareBitboards[targetSquare];  // add move to bitboard

            if (b->boardUnions[0] & squareBitboards[targetSquare]) {  // if there is a piece on the target square, stop looking in this direction
                break;
            }

            x += dx;
            y += dy;
        }
    }
    
    return moves;
}

uint64_t generateWhiteRookMoves(Board* b, Square src) {
    
    uint64_t moves = 0;
    // for now before we add magics, we do this manually
    int i, j;
    getIJFromSquare(src, &i, &j);

    for (int k = 4; k < 8; k++) {  // orth directions
        int x = j;
        int y = i;
        int dx = directions[k][0];
        int dy = directions[k][1];

        while (x + dx >= 0 && x + dx < 8 && y + dy >= 0 && y + dy < 8) {
            Square targetSquare = getSquareIndex(y + dy, x + dx);
            moves |= squareBitboards[targetSquare];  // add move to bitboard

            if (b->boardUnions[0] & squareBitboards[targetSquare]) {  // if there is a piece on the target square, stop looking in this direction
                break;
            }

            x += dx;
            y += dy;
        }
    }
    
    return moves;
}

uint64_t generateWhiteQueenMoves(Board* b, Square src) {
    
    uint64_t moves = 0;

    // for now before we add magics, we do this manually
    int i, j;
    getIJFromSquare(src, &i, &j);

    for (int k = 0; k < 8; k++) {
        int x = j;
        int y = i;
        int dx = directions[k][0];
        int dy = directions[k][1];

        while (x + dx >= 0 && x + dx < 8 && y + dy >= 0 && y + dy < 8) {
            Square targetSquare = getSquareIndex(y + dy, x + dx);
            moves |= squareBitboards[targetSquare];  // add move to bitboard

            if (b->boardUnions[0] & squareBitboards[targetSquare]) {  // if there is a piece on the target square, stop looking in this direction
                break;
            }

            x += dx;
            y += dy;
        }
    }
    
    return moves;
}

uint64_t generateWhiteKingMoves(Board* b, Square src) {

    return precomputedKingMoves[src] & ~b->boardUnions[1];
}

uint64_t generateBlackPawnMoves(Board* b, Square src) {
    // blacks perspective

    uint64_t moves = 0;

    uint64_t blackPawns = b->bitboards[iBP];
    uint64_t pawn = blackPawns & squareBitboards[src];
    uint64_t emptySquares = ~b->boardUnions[2];  // all pieces

    // single pushes
    uint64_t singlePushes = (pawn >> 8) & emptySquares;
    moves |= singlePushes;

    // double pushes
    uint64_t doublePushes = ((singlePushes & rank6) >> 8) & emptySquares;
    moves |= doublePushes;

    // captures
    uint64_t leftCaptures = (pawn >> 9) & b->boardUnions[0] & ~fileH;  // index 0 are the white pieces
    uint64_t rightCaptures = (pawn >> 7) & b->boardUnions[0] & ~fileA;
    moves |= leftCaptures | rightCaptures;

    // en passant captures
    Square epSquare = getEnPassant(b->gamestate);
    if (epSquare) {
        uint64_t epMask = squareBitboards[epSquare];
        uint64_t leftEPCaptures = (pawn >> 9) & epMask & ~fileH;  // en passant capture to the left (from black's perspective)
        uint64_t rightEPCaptures = (pawn >> 7) & epMask & ~fileA;  // en passant capture to the right (from black's perspective)
        moves |= leftEPCaptures | rightEPCaptures;
    }

    return moves;
}

uint64_t generateBlackKnightMoves(Board* b, Square src) {
    // apply blockers later
    uint64_t moves = 0;

    uint64_t blackKnight = b->bitboards[iBN] & squareBitboards[src];
    
    moves |= precomputedKnightmoves[src];  // add precomputed moves
    moves &= ~b->boardUnions[1];  // block black pieces
    
    return moves;
}
uint64_t generateBlackBishopMoves(Board* b, Square src) {
    
    uint64_t moves = 0;

    // for now before we add magics, we do this manually
    int i, j;
    getIJFromSquare(src, &i, &j);

    for (int k = 0; k < 4; k++) {
        int x = j;
        int y = i;
        int dx = directions[k][0];
        int dy = directions[k][1];

        while (x + dx >= 0 && x + dx < 8 && y + dy >= 0 && y + dy < 8) {
            Square targetSquare = getSquareIndex(y + dy, x + dx);
            moves |= squareBitboards[targetSquare];  // add move to bitboard

            if (b->boardUnions[1] & squareBitboards[targetSquare]) {  // if there is a piece on the target square, stop looking in this direction
                break;
            }

            x += dx;
            y += dy;
        }
    }
    
    return moves;
}
uint64_t generateBlackRookMoves(Board* b, Square src) {
    
    uint64_t moves = 0;

    // for now before we add magics, we do this manually
    int i, j;
    getIJFromSquare(src, &i, &j);

    for (int k = 4; k < 8; k++) {
        int x = j;
        int y = i;
        int dx = directions[k][0];
        int dy = directions[k][1];

        while (x + dx >= 0 && x + dx < 8 && y + dy >= 0 && y + dy < 8) {
            Square targetSquare = getSquareIndex(y + dy, x + dx);
            moves |= squareBitboards[targetSquare];  // add move to bitboard

            if (b->boardUnions[1] & squareBitboards[targetSquare]) {  // if there is a piece on the target square, stop looking in this direction
                break;
            }

            x += dx;
            y += dy;
        }
    }
    
    return moves;
}
uint64_t generateBlackQueenMoves(Board* b, Square src) {
   
    uint64_t moves = 0;

    // for now before we add magics, we do this manually
    int i, j;
    getIJFromSquare(src, &i, &j);

    for (int k = 0; k < 8; k++) {
        int x = j;
        int y = i;
        int dx = directions[k][0];
        int dy = directions[k][1];

        while (x + dx >= 0 && x + dx < 8 && y + dy >= 0 && y + dy < 8) {
            Square targetSquare = getSquareIndex(y + dy, x + dx);
            moves |= squareBitboards[targetSquare];  // add move to bitboard

            if (b->boardUnions[1] & squareBitboards[targetSquare]) {  // if there is a piece on the target square, stop looking in this direction
                break;
            }

            x += dx;
            y += dy;
        }
    }
    
    return moves;
}

uint64_t generateBlackKingMoves(Board* b, Square src) {
    
    return precomputedKingMoves[src] & ~b->boardUnions[1];
}

void precomputeKnights() {

    // this really only needs to run once at the beginning
    // here we iterate through all src and make a lookup table for the knights

    for (int square = 0; square < 64; square++) {
        precomputedKnightmoves[square] = 0;
        uint64_t moves = 0;
        uint64_t initialBitboard = squareBitboards[square];

        // we need to account for the borders as well
        // we'll convert to a i, j situation first so that we can check bounds
        // square = 56 - i * 8 + j
        int j = square % 8;  // file is easy
        int i = 7 - square / 8;  // j / 8 = 0, 56 / 8 = 7

        // check all 8 l shapes

        if (i - 2 >= 0 && j - 1 >= 0) {
            moves |= initialBitboard << 15;  // up 2, left 1
        }
        if (i - 2 >= 0 && j + 1 < 8) {
            moves |= initialBitboard << 17;  // up 2, right 1
        }
        if (i - 1 >= 0 && j - 2 >= 0) {
            moves |= initialBitboard << 6;   // up 1, left 2
        }
        if (i - 1 >= 0 && j + 2 < 8) {
            moves |= initialBitboard << 10;  // up 1, right 2
        }
        if (i + 1 < 8 && j - 2 >= 0) {
            moves |= initialBitboard >> 10;  // down 1, left 2
        }
        if (i + 1 < 8 && j + 2 < 8) {
            moves |= initialBitboard >> 6;   // down 1, right 2
        }
        if (i + 2 < 8 && j - 1 >= 0) {
            moves |= initialBitboard >> 17;  // down 2, left 1
        }
        if (i + 2 < 8 && j + 1 < 8) {
            moves |= initialBitboard >> 15;  // down 2, right 1
        }

        precomputedKnightmoves[square] = moves;
    }
}

void precomputeKingMoves() {
    // similar to knights

    for (int square = 0; square < 64; square++) {
        precomputedKingMoves[square] = 0;
        uint64_t moves = 0;
        uint64_t initialBitboard = squareBitboards[square];

        // add all 8 surrounding squares
        moves |= initialBitboard << 8;   // up
        moves |= initialBitboard >> 8;   // down
        moves |= (initialBitboard << 1) & ~fileA;  // right
        moves |= (initialBitboard >> 1) & ~fileH;  // left
        moves |= (initialBitboard << 9) & ~fileA;  // up right
        moves |= (initialBitboard << 7) & ~fileH;  // up left
        moves |= (initialBitboard >> 7) & ~fileA;  // down right
        moves |= (initialBitboard >> 9) & ~fileH;  // down left

        precomputedKingMoves[square] = moves;
    }
}