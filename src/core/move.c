#include "move.h"



// promo needs 3 bits instead of 2, since the third will represent whether or not a promo actually happened
//                                        v----- double pawn push (need more bits? no we have what we need)
// Move specification:      R R R R | R K D C | C C C E | E E E E | E P P P | T T T T | T T S S | S S S S
//                                      ^
//                         reserved castled capturedPiece enpassant promo     target        source


// debug stuff

// must pass
const char *sanTests[] = {
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
    "O-O-O#",      // castling mate
    NULL
};


// must fail
const char *invalidSanTests[] = {
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
    "Kxe1=Q",       // non-pawn promotion
    NULL  // terminator
};


bool DEBUG_algebraicNotationTests() {

    bool allPassed = true;
    for (int i = 0; sanTests[i] != NULL; i++) {
        if (!validAlgebraicNotation(sanTests[i])) {
            printf("%d: Test failed: %s\n", i, sanTests[i]);
            allPassed = false;
        }
    }

    for (int i = 0; invalidSanTests[i] != NULL; i++) {
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

Move getMoveFromSquare(Board* b, Square src, Square dst, bool promo) {
    Move m = NULL_MOVE;
    setSrc(&m, src);
    setDst(&m, dst);
    Piece piece = getPieceOnSquare(b, src);
    uint64_t srcMask = squareBitboards[src];
    uint64_t dstMask = squareBitboards[dst];

    if (piece == WK && src == e1) {
        if (dst == g1 || dst == h1) {
            setDst(&m, g1);
            setCastled(&m, true);
            return m;
        }
        else if (dst == c1 || dst == a1) {
            setDst(&m, c1);
            setCastled(&m, true);
            return m;
        }

    } else if (piece == BK && src == e8) {
        if (dst == g8 || dst == h8) {
            setCastled(&m, true);
            setDst(&m, g8);
            return m;
        }
        else if (dst == c8 || dst == a8) {
            setCastled(&m, true);
            setDst(&m, c8);
            return m;
        }
    }

    Piece capturedPiece = getPieceOnSquare(b, dst);
    setCapturedPiece(&m, capturedPiece);

    if (promo) {
        // the type of promo does not matter. We use queen because we still must undo later,
        // but we only check at most 3 promo squares.
        setPromotion(&m, promoQueen);
        return m;
    }

    // check for double pawn push
    if (piece == WP && (dstMask & rank4) && (srcMask & rank2)) {

        setDoublePush(&m, true);
        return m;
    
    } 

    else if (piece == BP && (dstMask & rank5) && (srcMask & rank7)) {

        setDoublePush(&m, true);
        return m;
    }

    Square ep = getEnPassantSquare(b->gamestate);
    if (piece == WP && dst == ep && (srcMask & rank5) && (dstMask & rank6)) {
        setEnPassant(&m, dst);
        setCapturedPiece(&m, BP);
    }

    else if (piece == BP && dst == ep && (srcMask & rank4) && (dstMask & rank3)) {
        setEnPassant(&m, dst);
        setCapturedPiece(&m, WP);
    }

    return m;
}

Move getMoveFromNotation(Board* b, char* moveStr) {

    // <move descriptor> ::= <from square><to square>[<promoted to>]
    // <square>        ::= <file letter><rank number>
    // <file letter>   ::= 'a'|'b'|'c'|'d'|'e'|'f'|'g'|'h'
    // <rank number>   ::= '1'|'2'|'3'|'4'|'5'|'6'|'7'|'8'
    // <promoted to>   ::= 'q'|'r'|'b'|'n'

    // we assume valid notation at this point
    // get move from something like e2e4, e7e8Q, e1g1, e5d6 etc
    // this function runs under the cli, speed is not of concern.
    // getMoveFromSquare will be the function called by the engine.

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

    if (piece == EMPTY) {
        // this should not happen in a legal move, but we will allow it for now and let the application deal with it
        return NULL_MOVE;
    }

    Piece capturedPiece = getPieceOnSquare(b, dst);

    // we can do castling here. check that type is king and that we move 2 or 3 squares just look at square enum

    if (piece == WK && src == e1) {
        if (dst == g1 || dst == h1) {
            setDst(&m, g1);
            setCastled(&m, true);
            return m;
        }
        else if (dst == c1 || dst == a1) {
            setDst(&m, c1);
            setCastled(&m, true);
            return m;
        }

    } else if (piece == BK && src == e8) {
        if (dst == g8 || dst == h8) {
            setCastled(&m, true);
            setDst(&m, g8);
            return m;
        }
        else if (dst == c8 || dst == a8) {
            setCastled(&m, true);
            setDst(&m, c8);
            return m;
        }
    }
    

    setCapturedPiece(&m, capturedPiece);  // after castling because castling doesnt capture anything
    
    // check promo and colour
    if (len == 5) {
        char promo = moveStr[4];
        uint8_t promoType = getPieceType(getPieceFromChar(promo));

        if (promoType == 0) {
            // should not happen if we assume valid notation
            return NULL_MOVE;
        }
        
        // not needed, right?
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

    // check for double pawn push
    if (piece == WP && dstRank == '4' && srcRank == '2') {

        setDoublePush(&m, true);
    
    } 

    else if (piece == BP && dstRank == '5' && srcRank == '7') {

        setDoublePush(&m, true);
    }

    if (dst == getEnPassantSquare(gamestate) && piece == WP && srcRank == '5' && dstRank == '6') {
        setEnPassant(&m, dst);
        setCapturedPiece(&m, BP);
    }

    else if (dst == getEnPassantSquare(gamestate) && piece == BP && srcRank == '4' && dstRank == '3') {
        setEnPassant(&m, dst);
        setCapturedPiece(&m, WP);
    }

    return m;
}

bool validMoveNotation(char* moveStr) {

    // for now we just check if the moveStr is 4 or 5 chars (for promotion) and if the first 4 chars are in the correct format. this is not a full validation of the move, just a check of the notation. we will check if the move is legal later in move.c

    int len = (int) strlen(moveStr);
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

    if (moveStr[0] == moveStr[2] && moveStr[1] == moveStr[3]) {
        // source and destination squares cannot be the same
        return false;
    }

    // check 5th char (promotion)
    if (len == 5) {
        if (moveStr[3] != '8' && moveStr[3] != '1') {
            // promotion can only happen on the last rank
            return false;
        }

        if (moveStr[1] != '7' && moveStr[1] != '2') {
            // promotion can only happen from the 7th rank for white and 2nd rank for black
            return false;
        }

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

    int len = (int) strlen(moveStr);
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

    //int len = (int) strlen(moveStr);
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

    ///TODO: when i have the chance, init 1 ptr for bitboards and for gamestate
    // in here, the actual changes to the board struct are made
    // many optimizations could be made, like using only one access per thing, stuff like that
    // but it works right now and these things need to be thought through in more rigour later
    // ie, im lazy
    // this looks so fucking ugly, and slow. but maybe its not. maybe there is beauty that eludes me.
    // assumed pseudolegal

    uint64_t* bitboards = b->bitboards;
    Piece* pieces = b->pieces;

    Square src = getSrc(move);
    Square dst = getDst(move);

    Piece srcPiece = pieces[src];
    Piece capturedPiece = getCapturedPiece(move);
    Piece promoPiece = EMPTY_TYPE;
    bool isCapture = capturedPiece != EMPTY;
    Piece promo = (Piece) getPromotion(move);
    Square epSquare = getEnPassant(move);
    bool isPromotion = promo != 0;
    bool isEnPassant = epSquare != 0;
    bool isDoublePawnPush = getDoublePush(move) != 0;  // for now, just set double pawn push to be the same as en passant, since they are closely related and we can always change this later if we need to
    ///TODO: castling MUST be normalized before this function, so that checks for certain squares are not done (they are unnecessary)
    bool isCastling = isCastled(move) != 0; 
    bool blackToMove = isBlackToMove(b->gamestate);

    uint64_t srcMask = squareBitboards[src];
    uint64_t dstMask = squareBitboards[dst];

    // toggle bits xor
    bitboards[getBitboardIndex(srcPiece)] ^= srcMask | dstMask;  // remove/add piece from src/dst square
    pieces[src] = EMPTY;  // update piece array
    pieces[dst] = srcPiece;

    // update zobrist hash
    b->zobrist ^= getZobristHash(srcPiece, src);  // remove piece from source square
    b->zobrist ^= getZobristHash(srcPiece, dst);  // add piece to destination square

    // update game state
    b->gamestate ^= GS_colourtoMoveMask;  // toggle side to move

    // now we want to accurately track anything that affects castling, like rook/king, capturing rook.
    if (srcPiece == WK) {
        removeCastlingRights(&b->gamestate, whiteLongCastleMask | whiteShortCastleMask);  // remove white castling right
        b->zobrist ^= getZobristCastleHash(whiteLongCastleMask) ^ getZobristCastleHash(whiteShortCastleMask);  // update castling hash for white
    } else if (srcPiece == BK) {
        removeCastlingRights(&b->gamestate, blackLongCastleMask | blackShortCastleMask);  // remove black castling right
        b->zobrist ^= getZobristCastleHash(blackLongCastleMask) ^ getZobristCastleHash(blackShortCastleMask);  // update castling hash for black
    }

    if (srcPiece == WR && src == h1) {
        removeCastlingRights(&b->gamestate, whiteShortCastleMask);  // remove white kingside castling right
        b->zobrist ^= getZobristCastleHash(whiteShortCastleMask);  // update castling hash for white kingside
    } else if (srcPiece == WR && src == a1) {
        removeCastlingRights(&b->gamestate, whiteLongCastleMask);  // remove white queenside castling right
        b->zobrist ^= getZobristCastleHash(whiteLongCastleMask);  // update castling hash for white queenside
    } else if (srcPiece == BR && src == h8) {
        removeCastlingRights(&b->gamestate, blackShortCastleMask);  // remove black kingside castling right
        b->zobrist ^= getZobristCastleHash(blackShortCastleMask);  // update castling hash for black kingside
    } else if (srcPiece == BR && src == a8) {
        removeCastlingRights(&b->gamestate, blackLongCastleMask);  // remove black queenside castling right
        b->zobrist ^= getZobristCastleHash(blackLongCastleMask);  // update castling hash for black queenside
    }

    if (isCapture) {
        if (capturedPiece == WR && dst == h1) {
            removeCastlingRights(&b->gamestate, whiteShortCastleMask);  // remove white kingside castling right
            b->zobrist ^= getZobristCastleHash(whiteShortCastleMask);  // update castling hash for white kingside
        } else if (capturedPiece == WR && dst == a1) {
            removeCastlingRights(&b->gamestate, whiteLongCastleMask);  // remove white queenside castling right
            b->zobrist ^= getZobristCastleHash(whiteLongCastleMask);  // update castling hash for white queenside
        } else if (capturedPiece == BR && dst == h8) {
            removeCastlingRights(&b->gamestate, blackShortCastleMask);  // remove black kingside castling right
            b->zobrist ^= getZobristCastleHash(blackShortCastleMask);  // update castling hash for black kingside
        } else if (capturedPiece == BR && dst == a8) {
            removeCastlingRights(&b->gamestate, blackLongCastleMask);  // remove black queenside castling right
            b->zobrist ^= getZobristCastleHash(blackLongCastleMask);  // update castling hash for black queenside
        }
    }

    if (isCapture || getPieceType(srcPiece) == PAWN) {
        setHalfmoveClock(&b->gamestate, 0);  // reset halfmove clock
    } else {
        incrHalfmoveClock(&b->gamestate);  // increment halfmove clock
    }

    // clear enpassant square
    setEnPassantSquare(&b->gamestate, 0);

    ///TODO: the order in which we check these conditions might be optimizable. castling happens rarely,
    // so what is the true tradeoff between checking it now or checking it later, if we may possibly return early?
    // etc. 

    if (isCastling) {
        // move rook
        // perhaps we can reuse bits or reserved to specify which rook (determines the whole castle) instead of this thing
        // but it MUST be consistent with the undo assumptions
        ///TODO: use 1 zobrist ptr (google if worth it or not)
        if (dst == g1) {  // white kingside

            bitboards[iWR] ^= (squareBitboards[h1] | squareBitboards[f1]);  // move rook
            pieces[f1] = WR;
            pieces[h1] = EMPTY;
            removeCastlingRights(&b->gamestate, whiteLongCastleMask | whiteShortCastleMask);  // remove white castling right
            b->zobrist ^= getZobristHash(WR, h1) ^ getZobristHash(WR, f1);  // move rook from h1 to f1
            b->zobrist ^= getZobristHash(WK, e1) ^ getZobristHash(WK, g1);  // move king from e1 to g1
            b->zobrist ^= getZobristCastleHash(whiteLongCastleMask);  // update castling hash for white kingside

        } else if (dst == c1) {  // white queenside

            bitboards[iWR] ^= (squareBitboards[a1] | squareBitboards[d1]);
            pieces[d1] = WR;
            pieces[a1] = EMPTY;
            removeCastlingRights(&b->gamestate, whiteShortCastleMask | whiteLongCastleMask);  // remove white castling right
            b->zobrist ^= getZobristHash(WR, a1) ^ getZobristHash(WR, d1);  // move rook from a1 to d1
            b->zobrist ^= getZobristHash(WK, e1) ^ getZobristHash(WK, c1);  // move king from e1 to c1
            b->zobrist ^= getZobristCastleHash(whiteShortCastleMask);  // update castling hash for white queenside

        } else if (dst == g8) {  // black kingside

            bitboards[iBR] ^= (squareBitboards[h8] | squareBitboards[f8]);
            pieces[f8] = BR;
            pieces[h8] = EMPTY;
            removeCastlingRights(&b->gamestate, blackLongCastleMask | blackShortCastleMask);  // remove black castling right
            b->zobrist ^= getZobristHash(BR, h8) ^ getZobristHash(BR, f8);  // move rook from h8 to f8
            b->zobrist ^= getZobristHash(BK, e8) ^ getZobristHash(BK, g8);  // move king from e8 to g8
            b->zobrist ^= getZobristCastleHash(blackLongCastleMask);  // update castling hash for black kingside

        } else if (dst == c8) {  // black queenside

            bitboards[iBR] ^= (squareBitboards[a8] | squareBitboards[d8]);
            pieces[d8] = BR;
            pieces[a8] = EMPTY;
            removeCastlingRights(&b->gamestate, blackShortCastleMask | blackLongCastleMask);  // remove black castling right
            b->zobrist ^= getZobristHash(BR, a8) ^ getZobristHash(BR, d8);  // move rook from a8 to d8
            b->zobrist ^= getZobristHash(BK, e8) ^ getZobristHash(BK, c8);  // move king from e8 to c8
            b->zobrist ^= getZobristCastleHash(blackShortCastleMask);  // update castling hash for black queenside
        }

        goto finishMove;
    }

    if (isDoublePawnPush) {
        epSquare = (blackToMove) ? dst + 8 : dst - 8;  // set en passant square to the square behind the pawn
        setEnPassantSquare(&b->gamestate, epSquare);  // set en passant square in gamestate
        b->zobrist ^= getZobristEnPassantHash(epSquare & 7);  // update en passant hash, & 7 is to get file of epSquare
        goto finishMove;
    }

    if (isCapture) {
        bitboards[getBitboardIndex(capturedPiece)] ^= dstMask;  // remove captured piece from destination square
        b->zobrist ^= getZobristHash(capturedPiece, dst);  // remove captured piece from destination square
    }

    if (isPromotion) {
        // toggle promotion piece on destination square
        uint8_t promoColour = getPiecesColour(srcPiece);
        promoPiece = (promoColour << 3) | promo - 2;  // -2 to convert to piece
        bitboards[getBitboardIndex(promoPiece)] ^= dstMask;  // add promotion piece to destination square
        bitboards[getBitboardIndex(srcPiece)] ^= dstMask;  // remove pawn from destination square
        pieces[dst] = promoPiece;  // update piece array with promotion piece
        b->zobrist ^= getZobristHash(promoPiece, dst);  // add promotion piece to destination square

        goto finishMove;  // capture already checked
    }

    // pawn CAPTURES enpassant
    if (isEnPassant) {
        // remove captured pawn
        bitboards[getBitboardIndex(capturedPiece)] ^= squareBitboards[epSquare];  // remove captured pawn from en passant square
        pieces[epSquare + (blackToMove ? 8 : -8)] = EMPTY;  // remove captured pawn
        b->zobrist ^= getZobristHash(capturedPiece, epSquare);  // remove captured pawn from en passant zobrist
        // src and dst toggle already happens
    }

    finishMove:
    updateBoardUnions(b);
    return;

}

