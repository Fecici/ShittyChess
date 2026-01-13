#include "printUtils.h"



static char* const pieceCodes[12] = {

        "♙",
        "♘",
        "♗",
        "♖",
        "♕",
        "♔",
        "♟",  // its black even though it looks white
        "♞",
        "♝",
        "♜",
        "♛",
        "♚"
    };




void printGameState(Board* b) {

    printf("Current game state - move %u\n", getMoveCount(b->ply));
    printf("gameState hex: %x\n", b->gameState);
    printBoard(b);
    printBitboards(b);
    printZobrist(b);

}

void printBoard(Board* b) {
// UNTESTED: need to do this
    Piece* arr = b->pieces;
    
    printf("---+---+---+---+---+---+---+---+---+\n");
    for (int i = 0; i < 8; i++) {
        printf(" %d ", 8 - i);
        for (int j = 8; j > 0; j--) {
            int k = 64 - i * 8 - j;  // index into arr

            Piece p = arr[k];

            char* pieceCode = " ";
            if (p != EMPTY) {
                pieceCode = pieceCodes[getBitboardIndex(p)];
            }

            printf("| %s ", pieceCode);

        }

        printf("|\n");
        printf("---+---+---+---+---+---+---+---+---+\n");
    }
    printf("   | a | b | c | d | e | f | g | h |\n");
}

void printBitboards(Board* b) {

    printBitBoard(b->bitboards[iWP], "White Pawns",   false);
    printBitBoard(b->bitboards[iWN], "White Knights", false);
    printBitBoard(b->bitboards[iWB], "White Bishops", false);
    printBitBoard(b->bitboards[iWR], "white Rooks",   false);
    printBitBoard(b->bitboards[iWQ], "White Queens",  false);
    printBitBoard(b->bitboards[iWK], "White King",    false);

    printBitBoard(b->bitboards[iBP], "Black Pawns",   false);
    printBitBoard(b->bitboards[iBN], "Black Knights", false);
    printBitBoard(b->bitboards[iBB], "Black Bishops", false);
    printBitBoard(b->bitboards[iBR], "Black Rooks",   false);
    printBitBoard(b->bitboards[iBQ], "Black Queens",  false);
    printBitBoard(b->bitboards[iBK], "Black King",    false);

}

void printBitBoard(uint64_t bitboard, char* name, bool makeSquare) {

    printf("%s: \n", name);
    
    if (makeSquare) {
        
        for (uint64_t i = 0; i < 8; i++) {
            for (uint64_t j = 8; j > 0; j--) {
                uint64_t k = ((uint64_t) 1 << ((64 - i * 8) - j));
                
                if (k & bitboard) printf("1");
                else              printf("0");
            }
            printf("\n");
        }
        
        printf("\n");
        return;
    }
    

    for (uint64_t i = 0; i < 64; i++) {
        uint64_t k = ((uint64_t)1 << (63 - i));
        
        
        if ( k & bitboard)  printf("1");
        else                printf("0");
    }
    printf("\n");
    return;
}



void printZobrist(Board* b) {

    printf("\nZobrist:\t%llx", b->zobrist);

}


