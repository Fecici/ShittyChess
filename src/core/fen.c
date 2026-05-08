#include "fen.h"

bool loadFromFen(Board* b, char* fen) {

    // clear board first
    memset(b, 0, sizeof(Board));

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
                j += (c - 0x30) - 1;  // add int to fen ptr. subtract 1 because we count from 0
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
    setColourToMove(&(b->gamestate), colourToMove);
    fen += 2;
    if (*fen != '-') {
        
        while (*fen != ' ') {
            uint8_t castleState = getValidCastlingFen(*fen);
            if (!castleState) return false;

            orCastlingRights(&(b->gamestate), castleState);
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
        setEnPassantSquare(&(b->gamestate), epSquare);
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
    setHalfmoveClock(&(b->gamestate), halfmove);

    char* fullmoves = fen;  // from here until \0
    unsigned int fenPly = convertFullmoveStringToPly(fullmoves, colourToMove);

    //if (!fenPly) return false;  // this should start at 0 when fullmove = 1 and white to move
    b->ply = fenPly;

    b->zobrist = generateZobristHash(b);

    return true;
}

// convert position to fen
char* convertToFen(Board* b) {
    char* fen = calloc(128, sizeof(char));  // this is definitely big enough for a fen string
    if (!fen) {
        fprintf(stderr, "Error: Memory allocation failed for FEN string\n");
        return NULL;
    }
    int fenIndex = 0;

    for (int i = 0; i < 8; i++) { 
        int emptyCount = 0;
        for (int j = 0; j < 8; j++) { 

            Piece piece = b->pieces[getSquareIndex(i, j)];
            if (piece == EMPTY) {
                emptyCount++;
            }
            else {
                if (emptyCount > 0) {
                    fen[fenIndex++] = (char) (emptyCount + 0x30);
                    emptyCount = 0;
                }
                char pieceChar = getCharFromPiece(piece);
                fen[fenIndex++] = pieceChar;
            }
        }
        if (emptyCount > 0) {
            fen[fenIndex++] = (char) (emptyCount + 0x30);
        }
        if (i != 7) {
            fen[fenIndex++] = '/';
        }
    }

    fen[fenIndex++] = ' ';

    uint8_t colourToMove = getColourToMove(b->gamestate);
    fen[fenIndex++] = colourToMove ? 'b' : 'w';

    fen[fenIndex++] = ' ';

    uint8_t castlingRights = getCastlingRights(b->gamestate);
    if (!castlingRights) {
        fen[fenIndex++] = '-';
    }
    else {
        if (castlingRights & whiteShortCastleMask) fen[fenIndex++] = 'K';
        if (castlingRights & whiteLongCastleMask) fen[fenIndex++]  = 'Q';
        if (castlingRights & blackShortCastleMask) fen[fenIndex++] = 'k';
        if (castlingRights & blackLongCastleMask) fen[fenIndex++]  = 'q';
    }

    fen[fenIndex++] = ' ';

    uint8_t epSquare = getEnPassantSquare(b->gamestate);
    if (!epSquare) {
        fen[fenIndex++] = '-';
    }
    else {

        char fileChar = (char)('a' + (epSquare % 8));
        char rankChar = (char)('1' + (epSquare / 8));   

        fen[fenIndex++] = fileChar;
        fen[fenIndex++] = rankChar;
    }
    fen[fenIndex] = '\0';  // null terminate the fen string

    uint8_t halfmoveClock = getHalfmoveClock(b->gamestate);
    char halfmoveStr[4];
    snprintf(halfmoveStr, sizeof(halfmoveStr), "%d", halfmoveClock);
    strcat(fen, " ");
    strcat(fen, halfmoveStr);

    char fullmoveStr[16];
    snprintf(fullmoveStr, sizeof(fullmoveStr), "%u", (b->ply >> 1) + 1);
    strcat(fen, " ");
    strcat(fen, fullmoveStr);

    return fen;
}

bool validFen(const char* fen) {

    // this is basically just a check that the fen is valid, so we can call loadFromFen with the guarantee that it will work. 

    // we can just try to load the fen into a dummy board and see if it works. 
    Board dummyBoard = {0};
    return loadFromFen(&dummyBoard, (char*) fen);
}