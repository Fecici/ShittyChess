#include "move.h"



// promo needs 3 bits instead of 2, since the third will represent whether or not a promo actually happened
//                                        v----- double pawn push (need more bits? no we have what we need)
// Move specification:      R R R R | R K D C | C C C E | E E E E | E P P P | T T T T | T T S S | S S S S
//                                      ^
//                         reserved castled capturedPiece enpassant promo     target        source


// debug stuff

// must pass
const char *sanTests[30] = {
    "e4",          // pawn push
    "d5",          // pawn push
    "Nf3",         // knight move
    "Nc6",         // knight move
    "Bb5",         // bishop move
    "a6",          // pawn push
    "O-O",         // kingside castling
    "O-O-O",       // queenside castling

    "exd5",        // pawn capture
    "axb8=Q",      // capture promotion
    "axb8=Q+",     // capture promotion check
    "axb8=Q#",     // capture promotion mate
    "e8=Q",        // quiet promotion
    "e8=N+",       // underpromotion with check
    "gxh8=R",      // capture underpromotion
    "gxh8=B#",     // capture underpromotion mate

    "Nbd2",        // file disambiguation
    "N5f7",        // rank disambiguation
    "Ngf3",        // file disambiguation
    "R1e2",        // rank disambiguation
    "Rad1",        // rook file disambiguation
    "Raxd1",       // rook capture with file disambiguation
    "R1xd1",       // rook capture with rank disambiguation
    "Qh4xe1#",     // full-square disambiguation capture mate

    "Bxh7+",       // bishop capture check
    "Nxe5",        // knight capture
    "Qxf7#",       // queen capture mate
    "Kxf2",        // king capture
    "O-O+",        // castling check
    "O-O-O#"       // castling mate
};


// must fail
const char *invalidSanTests[40] = {
    "",             // empty
    " ",            // whitespace only
    "e",            // missing rank
    "4",            // missing file
    "e9",           // invalid rank
    "i4",           // invalid file
    "E4",           // pawn files must be lowercase
    "ee4",          // invalid pawn move shape
    "e44",          // extra rank/character

    "Pxe4",         // SAN does not use P for pawns
    "Pe4",          // SAN does not use P for pawns
    "pxe4",         // invalid pawn piece marker
    "ex4",          // pawn capture missing destination file
    "exd",          // pawn capture missing destination rank
    "exd9",         // invalid destination rank
    "ixd4",         // invalid source file
    "exi4",         // invalid destination file

    "e8Q",          // promotion missing '='
    "e8=",          // promotion missing piece
    "e8=K",         // illegal promotion piece
    "e8=P",         // illegal promotion piece
    "e8=q",         // promotion piece must be uppercase
    "e4=Q",         // promotion on non-final rank
    "axb8=",        // capture promotion missing piece
    "axb8=K",       // illegal capture promotion piece

    "O",            // incomplete castle
    "O-O-O-O",      // too long
    "0-0",          // zeros instead of letter O, if strict SAN
    "0-0-0",        // zeros instead of letter O, if strict SAN
    "O-o",          // lowercase o
    "o-o",          // lowercase o

    "N",            // missing destination
    "Ne",           // missing rank
    "Nf9",          // invalid rank
    "Ni4",          // invalid file
    "Nff3",         // ambiguous/invalid duplicate file form
    "N55f3",        // invalid duplicate rank form
    "Nbd",          // missing destination rank
    "Nb9",          // invalid destination rank

    "Qh4xe1++",     // double check suffix not strict SAN
    "Qh4xe1##",     // double mate suffix
    "Qh4xe1+!",     // annotation not strict SAN
    "Qh4xe1??",     // annotation not strict SAN
    "Raxd1=Q",      // non-pawn promotion
    "Kxe1=Q"        // non-pawn promotion
};


bool DEBUG_algebraicNotationTests() {

    bool allPassed = true;
    for (int i = 0; i < 30; i++) {
        if (!validAlgebraicNotation(sanTests[i])) {
            printf("%d: Test failed: %s\n", i, sanTests[i]);
            allPassed = false;
        }
    }

    for (int i = 0; i < 40; i++) {
        if (validAlgebraicNotation(invalidSanTests[i])) {
            printf("%d: Test failed: %s\n", i, invalidSanTests[i]);
            allPassed = false;
        }
    }

    printf("All algebraic notation tests passed!\n");
    return allPassed;
}

void printMove(Move move) {
    // debug

    printf("Move: %x\n", move);
    printf("Src: %d\n", getSrc(move));
    printf("Dst: %d\n", getDst(move));
    printf("Promotion: %d\n", getPromotion(move));
    printf("En Passant: %d\n", getEnPassant(move));
    printf("Castled: %d\n", isCastled(move));
    printf("Captured Piece: %d\n", getCapturedPiece(move));
    printf("Captured Piece Type: %d\n", getCapturedType(move));
    printf("Captured Piece Colour: %d\n", getCapturedColour(move));

}

Move getMoveFromNotation(Board* b, char* moveStr) {

    // <move descriptor> ::= <from square><to square>[<promoted to>]
    // <square>        ::= <file letter><rank number>
    // <file letter>   ::= 'a'|'b'|'c'|'d'|'e'|'f'|'g'|'h'
    // <rank number>   ::= '1'|'2'|'3'|'4'|'5'|'6'|'7'|'8'
    // <promoted to>   ::= 'q'|'r'|'b'|'n'

    // we assume valid notation at this point
    // get move from something like e2e4, e7e8Q, e1g1, e5d6 etc

    Move m = NULL_MOVE;

    int len = strlen(moveStr);
    char srcFile, srcRank, dstFile, dstRank;
    srcFile = moveStr[0];
    srcRank = moveStr[1];
    dstFile = moveStr[2];
    dstRank = moveStr[3];

    Square src = (srcRank - 0x30 - 1) * 8 + (srcFile - 'a');
    Square dst = (dstRank - 0x30 - 1) * 8 + (dstFile - 'a');

    setSrc(&m, src);
    setDst(&m, dst);

    // get piecetype from bitboard, check captured piece, set enpassant, castling, and double
    // this is a cli program. speed is not an issue here.
    Gamestate gamestate = b->gamestate; 
    bool blackToMove = isBlackToMove(gamestate);
    Piece piece         = getPieceOnSquare(b, src);
    Piece capturedPiece = getPieceOnSquare(b, dst);


    // we can do castling here. check that type is king and that we move 2 or 3 squares just look at square enum

    if (piece == WK && src == e1) {
        if (dst == g1 || dst == h1 || dst == f1 || dst == a1) {
            setCastled(&m, true);
            return m;
        }
    } else if (piece == BK && src == e8) {
        if (dst == g8 || dst == h8 || dst == f8 || dst == a8) {
            setCastled(&m, true);
            return m;
        }
    }
    

    setCapturedPiece(&m, capturedPiece);
    if (piece == EMPTY) {
        // this should not happen in a legal move, but we will allow it for now and let the application deal with it
        return NULL_MOVE;
    }

    // check promo and colour
    if (len == 5) {
        char promo = moveStr[4];
        uint8_t promoType = getPieceType(getPieceFromChar(promo));

        if (promoType == 0) {
            // invalid promotion piece, should not happen if we assume valid notation
            return NULL_MOVE;
        }
        
        if (blackToMove) {
            promoType += 6;  // convert to black piece
        }

        switch (promoType) {
            case WN: case BN: promoType = promoKnight; break;
            case WB: case BB: promoType = promoBishop; break;
            case WR: case BR: promoType = promoRook; break;
            case WQ: case BQ: promoType = promoQueen; break;
            default:
                // invalid promotion piece type, should not happen if we assume valid notation
                return NULL_MOVE;
        }

        setPromotion(&m, promoType);
    }

    // check for double pawn push, goes hand in hand with en passant
    if (getPieceType(piece) == WP && dstRank == '4' && srcRank == '2') {
        setEnPassant(&m, dst - 8);  // set en passant square to the square behind the pawn
    } else if (getPieceType(piece) == BP && dstRank == '5' && srcRank == '7') {
        setEnPassant(&m, dst + 8);
    }

    return m;
}

bool validMoveNotation(char* moveStr) {

    // for now we just check if the moveStr is 4 or 5 chars (for promotion) and if the first 4 chars are in the correct format. this is not a full validation of the move, just a check of the notation. we will check if the move is legal later in move.c

    int len = strlen(moveStr);
    if (len != 4 && len != 5) return false;

    // check first 4 chars
    for (int i = 0; i < 4; i++) {
        char c = moveStr[i];
        if (i % 2 == 0) {
            // should be a-h
            if (c < 'a' || c > 'h') return false;
        } else {
            // should be 1-8
            if (c < '1' || c > '8') return false;
        }
    }

    // check 5th char (promotion)
    if (len == 5) {
        char c = moveStr[4];
        if (c != 'q' && c != 'r' && c != 'b' && c != 'n') return false;
    }

    return true;
}

bool validAlgebraicNotation(char* moveStr) {

    /*
    If the piece is sufficient to unambiguously determine the origin square, the whole from square is omitted. Otherwise, if two (or more) pieces of the same kind can move to the same square, the piece's initial is followed by (in descending order of preference)

    file of departure if different
    rank of departure if the files are the same but the ranks differ
    the complete origin square coordinate otherwise
    */

    // for now we just check if the moveStr is 2-5 chars (for promotion), 
    // if it starts with a piece char (or not for pawn moves), and if the 
    // rest of the chars are in the correct format
    // we may use this once we are able to generate the family of legal moves

    /*
    <SAN move descriptor piece moves>   ::= <Piece symbol>[<from file>|<from rank>|<from square>]['x']<to square>
    <SAN move descriptor pawn captures> ::= <from file>[<from rank>] 'x' <to square>[<promoted to>]
    <SAN move descriptor pawn push>     ::= <to square>[<promoted to>]*/

    // [piece?][disambiguation?][x?][destination][promotion?][check/mate?]

    int len = strlen(moveStr);
    if (len < 2 || len > 7) return false;  // minimum is e4, maximum is axb8=Q+

    // check for castling
    if (strcmp(moveStr, "O-O") == 0 || strcmp(moveStr, "O-O-O") == 0) {
        return true;
    }

    int i = 0;

    // check for piece char
    char c = moveStr[i];
    // only capital letters work here
    if (c == 'N' || c == 'B' || c == 'R' || c == 'Q' || c == 'K') {
        i++;
    }

    // check pawn
    if (c != 'N' && c != 'B' && c != 'R' && c != 'Q' && c != 'K') {
        // if the first char is not a piece char, then it must be a file char for a pawn move
        if (c < 'a' || c > 'h') return false;
        i++;
    }

    // check for disambiguation char (file, rank, or square)
    if (i < len && ((moveStr[i] >= 'a' && moveStr[i] <= 'h') || (moveStr[i] >= '1' && moveStr[i] <= '8'))) {
        i++;
    }

    // check for second disambiguation char (if the first was a file char, then this must be a rank char, and vice versa)
    if (i < len && ((moveStr[i] >= 'a' && moveStr[i] <= 'h') || (moveStr[i] >= '1' && moveStr[i] <= '8'))) {
        i++;
    }

    // check for capture char
    if (moveStr[i] == 'x') {
        i++;
    }

    // check for destination square
    if (i + 1 >= len) return false;
    char dstFile = moveStr[i];
    char dstRank = moveStr[i + 1];
    if (dstFile < 'a' || dstFile > 'h') return false;
    if (dstRank < '1' || dstRank > '8') return false;
    i += 2;

    // check for promotion
    if (i < len) {
        char promo = moveStr[i];
        if (promo != 'q' && promo != 'r' && promo != 'b' && promo != 'n') return false;
        i++;
    }

    if (i < len) {
        // check for check or mate char
        char checkMate = moveStr[i];
        if (checkMate != '+' && checkMate != '#') return false;
        i++;
    }

    return i == len;
}

Move getMoveFromAlgebra(Board* b, char* moveStr) {

    // we need the board because we need to be able to find src pieces from bitboards

    // we assume valid algebraic notation at this point, eg Nf3, e4, exd5, O-O, ...

    // promo needs 3 bits instead of 2, since the third will represent whether or not a promo actually happened
//                                        v----- double pawn push (need more bits? no we have what we need)
// Move specification:      R R R R | R K D C | C C C E | E E E E | E P P P | T T T T | T T S S | S S S S
//                                      ^
//                         reserved castled capturedPiece enpassant promo     target        source

    bool blackToMove = isBlackToMove(b->gamestate);

    Move m = NULL_MOVE;

    int len = strlen(moveStr);
    int i = 0;

    // check for castling
    if (strcmp(moveStr, "O-O") == 0) {  // kingside
        
        setCastled(&m, true);

        if (blackToMove) {
            setSrc(&m, e8);
            setDst(&m, g8);
        } else {
            setSrc(&m, e1);
            setDst(&m, g1);
        }

        return m;
    }

    if (strcmp(moveStr, "O-O-O") == 0) {  // queenside
        
        setCastled(&m, true);  

        if (blackToMove) {
            setSrc(&m, e8);
            setDst(&m, c8);
        } else {
            setSrc(&m, e1);
            setDst(&m, c1);
        }

        return m;
    }

    // check for piece char
    ///TODO: implement legal moves first then finish this

    return m;
}

Move getMoveFromHex(char* hexStr) {


    Move m = (Move) strtol(hexStr, NULL, 0);
    Board b = {0};  // test board
    if (!isLegalMove(&b, m)) {
        fprintf(stderr, "Invalid move hex: %s\n", hexStr);
        return 0;
    }
    return m;
}

void makeMove(Board* b, Move move) {

    // in here, the actual changes to the board struct are made
    // assumed pseudolegal

    Square src = getSrc(move);
    Square dst = getDst(move);

    Piece srcPiece = getPieceOnSquare(b, src);
    Piece capturedPiece = getCapturedPiece(move);
    Piece promoPiece = EMPTY_TYPE;
    bool isCapture = capturedPiece != EMPTY;
    uint8_t promo = getPromotion(move);
    Square epSquare = getEnPassant(move);
    bool isPromotion = promo != 0;
    bool isEnPassant = epSquare != 0;
    bool isDoublePawnPush = getDoublePush(move) != 0;  // for now, we will just set double pawn push to be the same as en passant, since they are closely related and we can always change this later if we need to
    ///TODO: castling MUST be normalized before this function, so that checks for certain squares are not done (they are unnecessary)
    bool isCastling = isCastled(move); 
    bool blackToMove = isBlackToMove(b->gamestate);

    // find corresponding bitboard
    uint64_t* dstBitboard = NULL;
    uint64_t* srcBitboard = &b->bitboards[getBitboardIndex(srcPiece)];
    if (isCapture) {
        dstBitboard = &b->bitboards[getBitboardIndex(capturedPiece)];
    }

    uint64_t srcMask = squareBitboards[src];
    uint64_t dstMask = squareBitboards[dst];

    // toggle bits xor
    *srcBitboard ^= srcMask;  // remove piece from source square
    *srcBitboard ^= dstMask;   // add piece to destination square
    b->pieces[src] = EMPTY;  // update piece array
    b->pieces[dst] = srcPiece;

    if (isCapture) {
        *dstBitboard ^= dstMask;  // remove captured piece from destination square
    }

    if (isPromotion) {
        // toggle promotion piece on destination square
        uint8_t promoColour = getPiecesColour(srcPiece);
        promoPiece = (promoColour << 3) | promo;
        uint64_t* promoBitboard = &b->bitboards[getBitboardIndex(promoPiece)];
        *promoBitboard ^= dstMask;  // add promotion piece to destination square
    }

    if (isEnPassant) {
        // remove captured pawn
        uint64_t epMask = 1ULL << epSquare;
        Piece epCapturedPiece = blackToMove ? WP : BP;
        uint64_t* epBitboard = &b->bitboards[getBitboardIndex(epCapturedPiece)];
        *epBitboard ^= epMask;  // remove captured pawn from en passant square
    }

    if (isCastling) {
        // move rook
        // perhaps we can reuse bits or reserved to specify which rook (determines the whole castle) instead of this thing
        // but it MUST be consistent with the undo assumptions
        if (dst == g1) {  // white kingside
            b->bitboards[getBitboardIndex(WR)] ^= (squareBitboards[h1]) | (squareBitboards[f1]);
            removeCastlingRights(&b->gamestate, whiteLongCastleMask);  // remove white kingside castling right
            b->zobrist ^= getZobristHash(WR, h1) ^ getZobristHash(WR, f1);  // move rook from h1 to f1
            b->zobrist ^= getZobristHash(WK, e1) ^ getZobristHash(WK, g1);  // move king from e1 to g1
            b->zobrist ^= getZobristCastleHash(whiteLongCastleMask);  // update castling hash for white kingside

        } else if (dst == c1) {  // white queenside
            b->bitboards[getBitboardIndex(WR)] ^= (squareBitboards[a1]) | (squareBitboards[d1]);
            removeCastlingRights(&b->gamestate, whiteShortCastleMask);  // remove white queenside castling right
            b->zobrist ^= getZobristHash(WR, a1) ^ getZobristHash(WR, d1);  // move rook from a1 to d1
            b->zobrist ^= getZobristHash(WK, e1) ^ getZobristHash(WK, c1);  // move king from e1 to c1
            b->zobrist ^= getZobristCastleHash(whiteShortCastleMask);  // update castling hash for white queenside
        } else if (dst == g8) {  // black kingside
            b->bitboards[getBitboardIndex(BR)] ^= (squareBitboards[h8]) | (squareBitboards[f8]);
            removeCastlingRights(&b->gamestate, blackLongCastleMask);  // remove black kingside castling right
            b->zobrist ^= getZobristHash(BR, h8) ^ getZobristHash(BR, f8);  // move rook from h8 to f8
            b->zobrist ^= getZobristHash(BK, e8) ^ getZobristHash(BK, g8);  // move king from e8 to g8
            b->zobrist ^= getZobristCastleHash(blackLongCastleMask);  // update castling hash for black kingside
        } else if (dst == c8) {  // black queenside
            b->bitboards[getBitboardIndex(BR)] ^= (squareBitboards[a8]) | (squareBitboards[d8]);
            removeCastlingRights(&b->gamestate, blackShortCastleMask);  // remove black queenside castling right
            b->zobrist ^= getZobristHash(BR, a8) ^ getZobristHash(BR, d8);  // move rook from a8 to d8
            b->zobrist ^= getZobristHash(BK, e8) ^ getZobristHash(BK, c8);  // move king from e8 to c8
            b->zobrist ^= getZobristCastleHash(blackShortCastleMask);  // update castling hash for black queenside
        }
    }

    if (isDoublePawnPush) {
        // set en passant square
        Square epSquare = blackToMove ? dst + 8 : dst - 8;
        setEnPassant(&move, epSquare);
        setEnPassantSquare(&b->gamestate, epSquare);
    } else {
        // clear en passant square
        setEnPassantSquare(&b->gamestate, 0);
    }

    // update zobrist hash
    b->zobrist ^= getZobristHash(srcPiece, src);  // remove piece from source square
    b->zobrist ^= getZobristHash(srcPiece, dst);  // add piece to destination square
    if (isCapture) {
        b->zobrist ^= getZobristHash(capturedPiece, dst);  // remove captured piece from destination square
    }
    if (isPromotion) {
        b->zobrist ^= getZobristHash(promoPiece, dst);  // add promotion piece to destination square
    }
    if (isEnPassant) {
        Piece epCapturedPiece = blackToMove ? WP : BP;
        b->zobrist ^= getZobristHash(epCapturedPiece, epSquare);  // remove captured pawn from en passant square
        b->zobrist ^= getZobristEnPassantHash(epSquare & 7);  // update en passant hash, & 7 is to get file of epSquare

    }

    // update game state
    b->gamestate ^= GS_colourtoMoveMask;  // toggle side to move
    if (isCapture || getPieceType(srcPiece) == PAWN) {
        b->gamestate &= ~GS_halfmoveClockMask;  // reset halfmove clock
    } else {
        b->gamestate = (b->gamestate & ~GS_halfmoveClockMask) | (((b->gamestate & GS_halfmoveClockMask) >> 8) + 1);  // increment halfmove clock
    }

    ///TODO: handle pushing move and undo to stack (in a wrapper maybe?)

}

int handleMakeMove(Board* b, Move move) {
    // idk what this will do, maybe it is a wrapper.
    // check colour to move, check legality unless forced (we just set the pieces there then and go around this function entirely)
    bool colourToMove = isBlackToMove(b->gamestate);
    Piece piece = getPieceOnSquare(b, getSrc(move));
    if (piece == EMPTY || getPiecesColour(piece) != colourToMove) {
        fprintf(stderr, "Illegal move: piece on source square does not match colour to move\n");
        return 1;
    }

    if (!isLegalMove(b, move)) {
        fprintf(stderr, "Illegal move: move is not legal in the current position\n");
        return 1;
    }

    makeMove(b, move);
    return 0;
}

Piece getPieceOnSquare(Board* b, Square sq) {
    return b->pieces[sq];
}

// for now, play, check king, unmove
bool isLegalMove(Board* b, Move move) {

    (void) b;
    (void) move;
    return true;
}

