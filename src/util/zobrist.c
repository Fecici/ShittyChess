#include "zobrist.h"


/*
clanker did this. i want to do it differently, so this serves as a reference
check spec in header for more insight into this. 

https://www.chessprogramming.org/Zobrist_Hashing
*/

uint64_t zobristTable[12][64];  // [piece][square]
uint64_t zobristCastle[16];  // 16 possible castling rights combinations
uint64_t zobristEnPassant[8];  // 8 possible en passant files
uint64_t zobristBlackToMove;  // for black to move


static uint64_t zobrist_seed = 0x9E3779B97F4A7C15ULL;

static uint64_t splitmix64(void) {
    uint64_t z;

    // tried and true prng
    zobrist_seed += 0x9E3779B97F4A7C15ULL;
    z = zobrist_seed;

    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);

    if (z == 0) {
        return splitmix64();;
    }

    // check that key has not yet appeared
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 64; j++) {
            if (zobristTable[i][j] == z) {
                return splitmix64();
            }
        }
    }

    for (int i = 0; i < 16; i++) {
        if (zobristCastle[i] == z) {
            return splitmix64();
        }
    }

    for (int i = 0; i < 8; i++) {
        if (zobristEnPassant[i] == z) {
            return splitmix64();
        }
    }

    if (zobristBlackToMove == z) {
        return splitmix64();
    }

    return z;
}

// check that table has rank 64 over the field F_2
static bool checkZobristTableRank() {

    uint64_t basis[64];
    int rank = 0;

    // it is enough to check the rank of the piece matrix; adding more vectors definitionally cannot increase the rank
    for (int piece = 0; piece < 12; piece++) {
        for (int square = 0; square < 64; square++) {
            uint64_t key = zobristTable[piece][square];
            for (int i = 0; i < rank; i++) {
                if ((key ^ basis[i]) < key) {  // if the xor is smaller, then the leading bit has been cancelled, so we can reduce key by xoring with basis[i]
                    key ^= basis[i];
                }
            }
            if (key) {  // i.e. non-zero (linearly independent). over F2, there are only 2 scalars. 
                        // wlog we check only c = 1. c = 0 is included as a subset automatically
                basis[rank++] = key;
            }
        }
    }

    return rank == 64;
}

void initZobrist() {

    for (int piece = 0; piece < 12; piece++) {
        for (int square = 0; square < 64; square++) {
            zobristTable[piece][square] = splitmix64();
        }
    }

    zobristBlackToMove = splitmix64();

    for (int i = 0; i < 16; i++) {
        zobristCastle[i] = splitmix64();
    }

    for (int file = 0; file < 8; file++) {
        zobristEnPassant[file] = splitmix64();
    }

    if (!checkZobristTableRank()) {
        fprintf(stderr, "Error: Zobrist table does not have full rank. Regenerating...\n");
        initZobrist();
    }

}

uint64_t generateZobristHash(Board* b) {
    
    uint64_t hash = 0;

    for (int i = 0; i < 12; i++) {
        uint64_t bitboard = b->bitboards[i];
        while (bitboard) {
            uint64_t k = bitboard & -bitboard;  // get least significant bit
            uint8_t squareIndex = __builtin_ctzll(k);  // get index of least significant bit
            hash ^= zobristTable[i][squareIndex];  // xor with corresponding zobrist value
            bitboard &= bitboard - 1;  // clear least significant bit
        }
    }

    // castling rights
    uint8_t castlingRights = getCastlingRights(b->gameState);
    hash ^= zobristCastle[castlingRights];

    // en passant
    uint8_t epSquare = getEnPassantSquare(b->gameState);
    if (epSquare) {
        hash ^= zobristEnPassant[epSquare % 8];  // only the file matters for en passant
    }

    // colour to move
    if (isBlackToMove(b->gameState)) {
        hash ^= zobristBlackToMove;
    }

    return hash;
}
