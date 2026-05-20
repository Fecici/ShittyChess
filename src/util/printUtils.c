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

void printHistory(Undo64* undoStack, unsigned int ply) {

    printf("Move history (most recent move last):\n");
    for (unsigned int i = 0; i < ply; i++) {
        Move move = getMoveFromUndo(undoStack[i]);
        Square src = getSrc(move);
        Square dst = getDst(move);

        // rough convert to alg notation
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
    printf("Castling rights: %x | BLACK SHORT %x | BLACK LONG %x | WHITE SHORT %x | WHITE LONG %x\n", 
        getCastlingRights(b->gamestate), 
        canBlackCastleShort(b->gamestate) ? 1 : 0,
        canBlackCastleLong(b->gamestate) ? 1 : 0,
        canWhiteCastleShort(b->gamestate) ? 1 : 0,
        canWhiteCastleLong(b->gamestate) ? 1 : 0
    );

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

void printLegalMoves(Board* b, bool makeSquare) {
    // prints all legal moves on board by piece

    for (int i = 0; i < 12; i++) {
        uint64_t bitboard = b->bitboards[i];
        while (bitboard) {
            uint64_t k = bitboard & -bitboard;  // get least significant bit
            uint8_t squareIndex = __builtin_ctzll(k);  // get index of least significant bit
            uint64_t moves = pieceGenerator[i](b, squareIndex);  // get pseudo-legal moves for this piece on this square
            moves = getLegalFromPseudo(b, moves, squareIndex);  // filter pseudo-legal moves to legal moves
            printBitBoard(moves, getPieceNameFromIndex(i), makeSquare);
            bitboard &= bitboard - 1;  // clear least significant bit
        }
    }
}

void printPseudoLegalMoves(Board* b, bool printSquare) {
    // prints all pseudo-legal moves on board by piece

    for (int i = 0; i < 12; i++) {
        uint64_t bitboard = b->bitboards[i];
        while (bitboard) {
            uint64_t k = bitboard & -bitboard;  // get least significant bit
            uint8_t squareIndex = __builtin_ctzll(k);  // get index of least significant bit
            uint64_t moves = pieceGenerator[i](b, squareIndex);  // get pseudo-legal moves for this piece on this square
            printBitBoard(moves, getPieceNameFromIndex(i), printSquare);
            bitboard &= bitboard - 1;  // clear least significant bit
        }
    }

}

void printLegalMovesFromSquare(Board* b, Square src, bool printSquare) {
    // prints all legal moves from a specific square

    Piece piece = b->pieces[src];
    if (piece == EMPTY) {
        printf("No piece on the specified square.\n");
        return;
    }

    uint64_t moves = pieceGenerator[getBitboardIndex(piece)](b, src);

    moves = getLegalFromPseudo(b, moves, src);

    printBitBoard(moves, "Legal moves from square", printSquare);
}

void printLegalMovesForColour(Board* b, Colour colour, bool printSquare) {

    for (int i = colour * 6; i < 6 + colour * 6; i++) {
        uint64_t bitboard = b->bitboards[i];
        while (bitboard) {
            uint64_t k = bitboard & -bitboard;  // get least significant bit
            uint8_t squareIndex = __builtin_ctzll(k);  // get index of least significant bit
            uint64_t moves = pieceGenerator[i](b, squareIndex);  // get pseudo-legal moves for this piece on this square
            moves = getLegalFromPseudo(b, moves, squareIndex);  // filter pseudo-legal moves to legal moves
            printBitBoard(moves, getPieceNameFromIndex(i), printSquare);
            bitboard &= bitboard - 1;  // clear least significant bit
        }
    }

}

void printLegalMovesForPiece(Board* b, Piece piece, bool printSquare) {

    PieceIndex pieceIndex = getBitboardIndex(piece);
    uint64_t bitboard = b->bitboards[pieceIndex];

    while (bitboard) {
        uint64_t k = bitboard & -bitboard;  // get least significant bit
        uint8_t squareIndex = __builtin_ctzll(k);  // get index of least significant bit
        uint64_t moves = pieceGenerator[pieceIndex](b, squareIndex);  // get pseudo-legal moves for this piece on this square
        moves = getLegalFromPseudo(b, moves, squareIndex);  // filter pseudo-legal moves to legal moves
        printBitBoard(moves, getPieceNameFromPiece(piece), printSquare);
        bitboard &= bitboard - 1;  // clear least significant bit
    }

}
