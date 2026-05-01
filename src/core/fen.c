#include "fen.h"

bool isCharInt(const char c) {
    return '0' <= c && c <= '9';
}


Piece getPieceFromChar(const char c) {

    switch (c) {
        case 'P': return WP;
        case 'N': return WN;
        case 'B': return WB;
        case 'R': return WR;
        case 'K': return WK;
        case 'Q': return WQ;

        case 'p': return BP;
        case 'n': return BN;
        case 'b': return BB;
        case 'r': return BR;
        case 'k': return BK;
        case 'q': return BQ;
        default: return EMPTY;  // not valid piece
    }
}

static inline bool isValidPiece(const char c) {
    // basically one of these will kill c if its valid
    return !(
        (c ^ 'r') & (c ^ 'n') & (c ^ 'b') & (c ^ 'q') & (c ^ 'k') & (c ^ 'p') &
        (c ^ 'R') & (c ^ 'N') & (c ^ 'B') & (c ^ 'Q') & (c ^ 'K') & (c ^ 'P')
    );
}

static inline unsigned int getSquareIndex(const int i, const int j) {

    // i gives the chunk, j gives the index.
    // eg, 00001000 00000000 ...
    // is the 0th i and 3rd j, and the square is 59. so we need the conversion 64 - i*8 + j - 8 = 56 - i * 8 + j
    return (unsigned int) (56 - i * 8 + j);

}

// return the uint64_t with a 1 in the position of rank 8 - i and file j
static inline uint64_t getPieceBitboardSetter(const int i, const int j) {

    uint64_t k = 1;

    return k << getSquareIndex(i, j);
}

static inline uint8_t getValidCastlingFen(const char c) {
    switch (c) {
        case 'K': return whiteShortCastleMask;
        case 'Q': return whiteLongCastleMask;
        case 'k': return blackShortCastleMask;
        case 'q': return blackLongCastleMask;
        default:  return 0x0;
    }
}

static inline uint8_t convertSquareNotationToEP(const char file, const char rank) {

    if (rank != '3' || rank != '6' || rank < '1' || rank > '8') return 0;

    uint8_t k = 16;
    k += (uint8_t) (file - '0' - 1);
    if (rank == '6') k += 24;
    return k;

}

static inline unsigned int convertFullmoveStringToPly(const char* fullmoves, uint64_t blackToMove) {


    return ((((unsigned int) strtol(fullmoves, NULL, 0)) - 1) << 1) + ((unsigned int) blackToMove);

}

bool loadFromFen(Board* b, char* fen) {

    //printf("%s\n", fen);
    for (int i = 0; i < 8; i++) { 
        //printf("%d\n", i);
        for (int j = 0; j < 8; j++) {  // could try <= 8 to assume that we terminate with '/'. probably not necessary?

            char c = *fen;
            
            if (c == '/') {
                fen++;
                j--;
                continue;
            }
            
            if (isCharInt(c)) {
                j += (c - 0x30);  // add int to fen ptr
                fen++;
                continue;
            }
            fen++;
            //printf("%c %s\n", c, fen);

            if (!isValidPiece(c)) return false;
            Piece piece = getPieceFromChar(c);
            if (piece == EMPTY) return false;
            uint8_t pieceIndex = getBitboardIndex(piece);
            unsigned int squareIndex = getSquareIndex(i, j);

            b->pieces[squareIndex] = piece;
            b->bitboards[pieceIndex] |= getPieceBitboardSetter(i, j);
        }

    }

    //printf("AFTER LOOP 1: %s\n", fen);

    if (*fen != ' ') return false;
    fen++;

    uint8_t colourToMove = 0;
    if (*fen == 'b') colourToMove = 1;
    else if (*fen != 'w') return false;
    setColourToMove(&(b->gameState), colourToMove);
    fen += 2;
    if (*fen != '-') {
        
        while (*fen != ' ') {
            uint8_t castleState = getValidCastlingFen(*fen);
            if (!castleState) return false;

            orCastlingRights(&(b->gameState), castleState);
            fen++;
        }
    }
    else {
        fen++;
    }

    if (*fen != ' ') return false;
    fen++;
    if (*fen != '-') {

        char file = *fen;
        char rank = *(fen + 1);

        uint8_t epSquare = convertSquareNotationToEP(file, rank);
        if (!epSquare) return false;
        setEnPassantSquare(&(b->gameState), epSquare);
        fen += 2;
    }
    else fen++;
    if (*fen != ' ') return false;
    fen++; 
    
    char digit1 = *fen;
    char digit2 = *(fen + 1);
    fen += 2;

    uint8_t halfmove = 0;

    if (!isCharInt(digit1)) return false;

    if (digit1 == '0') {
        if (digit2 != ' ') return false;
    }
    else {

        if (digit2 == ' ') {
            halfmove = (uint8_t) (digit1 - 0x30);
        }

        else {
            if (!isCharInt(digit2)) return false;
            halfmove = (uint8_t) (((uint8_t) (digit1 - 0x30)) * 10 + ((uint8_t) (digit2 - 0x30)));
            if (*fen != ' ') return false;
            fen++;
        }
    }
    setHalfmoveClock(&(b->gameState), halfmove);

    char* fullmoves = fen;  // from here until \0
    unsigned int fenPly = convertFullmoveStringToPly(fullmoves, colourToMove);

    //if (!fenPly) return false;  // this should start at 0 when fullmove = 1 and white to move
    fenPly += colourToMove;
    b->ply = fenPly;

    b->zobrist = generateZobristHash(b);

    return true;
}

// convert position to fen (lets call this with a flag in the fen cmd)
char* convertToFen(Board* b);
