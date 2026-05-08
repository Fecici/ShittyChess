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


static char* const pieceNames[13] = {

        "White Pawn",
        "White Knight",
        "White Bishop",
        "White Rook",
        "White Queen",
        "White King",
        "Black Pawn",
        "Black Knight",
        "Black Bishop",
        "Black Rook",
        "Black Queen",
        "Black King",
        "EMPTY"
    };


char* getPieceNameFromPiece(Piece piece) {
    if (piece == EMPTY) return pieceNames[12];
    return pieceNames[getBitboardIndex(piece)];
}

char* getPieceNameFromIndex(uint8_t index) {
    if (index > 11) return pieceNames[12];
    return pieceNames[index];
}

void printHistory(History* h) {

    printf("Move history (most recent move last):\n");
    for (unsigned int i = 0; i < h->ply; i++) {
        Move move = h->moveHistory[i];
        Square src = getSrc(move);
        Square dst = getDst(move);
        printf("%u. %c%d -> %c%d\n", getMoveCount(i), 'a' + src % 8, 8 - src / 8, 'a' + dst % 8, 8 - dst / 8);
    }
}


void printGameState(Board* b, bool makeSquare) {

    printf("Current game state - move %u\n", getMoveCount(b->ply));
    printf("gameState hex: %x\n", (uint32_t) b->gamestate);
    printf("gameState binary: ");
    printf("Colour to move: %s\n", isBlackToMove(b->gamestate) ? "Black" : "White");
    printf("Halfmove clock: %u\n", getHalfmoveClock(b->gamestate));
    printf("En passant square: %u\n", getEnPassantSquare(b->gamestate));
    printf("Castling rights: %u\n", getCastlingRights(b->gamestate));
    printf("Evaluation: %d\n", evaluateBoard(b));
    for (int i = 31; i >= 0; i--) {
        uint32_t k = ((uint32_t)1 << i);
        if (k & (uint32_t) b->gamestate) printf("1");
        else                  printf("0");
    }
    printf("\n");
    printBoard(b);
    printBitboards(b, makeSquare);
    printZobrist(b);

}

void printBoard(Board* b) {
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

void printBitboardHex(uint64_t bitboard, char* name) {
    printf("%s: %llx\n", name, bitboard);
}

void printBitboardHexAll(Board* b) {

    printBitboardHex(b->bitboards[iWP], "White Pawns");
    printBitboardHex(b->bitboards[iWN], "White Knights");
    printBitboardHex(b->bitboards[iWB], "White Bishops");
    printBitboardHex(b->bitboards[iWR], "white Rooks");
    printBitboardHex(b->bitboards[iWQ], "White Queens");
    printBitboardHex(b->bitboards[iWK], "White King");

    printBitboardHex(b->bitboards[iBP], "Black Pawns");
    printBitboardHex(b->bitboards[iBN], "Black Knights");
    printBitboardHex(b->bitboards[iBB], "Black Bishops");
    printBitboardHex(b->bitboards[iBR], "Black Rooks");
    printBitboardHex(b->bitboards[iBQ], "Black Queens");
    printBitboardHex(b->bitboards[iBK], "Black King");

}

void printBitboards(Board* b, bool makeSquare) {

    printBitBoard(b->bitboards[iWP], "White Pawns",   makeSquare);
    printBitBoard(b->bitboards[iWN], "White Knights", makeSquare);
    printBitBoard(b->bitboards[iWB], "White Bishops", makeSquare);
    printBitBoard(b->bitboards[iWR], "white Rooks",   makeSquare);
    printBitBoard(b->bitboards[iWQ], "White Queens",  makeSquare);
    printBitBoard(b->bitboards[iWK], "White King",    makeSquare);

    printBitBoard(b->bitboards[iBP], "Black Pawns",   makeSquare);
    printBitBoard(b->bitboards[iBN], "Black Knights", makeSquare);
    printBitBoard(b->bitboards[iBB], "Black Bishops", makeSquare);
    printBitBoard(b->bitboards[iBR], "Black Rooks",   makeSquare);
    printBitBoard(b->bitboards[iBQ], "Black Queens",  makeSquare);
    printBitBoard(b->bitboards[iBK], "Black King",    makeSquare);

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

void printEval(Board* b) {
    int eval = evaluateBoard(b);
    printf("Evaluation: %d\n", eval);
}

void printZobrist(Board* b) {

    printf("\nZobrist:\t%llx\n", b->zobrist);

}


