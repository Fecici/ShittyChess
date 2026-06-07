#include "history.h"

// keep ply in data here?
// RETIRED
Undo* getUndoFromMove(Board* b, Move move) {

    Undo* undo = calloc(1, sizeof(Undo));
    undo->zobrist = b->zobrist;
    undo->captured = getPieceOnSquare(b, getDst(move));
    undo->enpassant = getEnPassant(move);
    undo->castling_rights = getCastlingRights(b->gamestate);
    undo->halftime = getHalfmoveClock(b->gamestate);

    return undo;
}


void performUndo(Board* b, Undo64 undo) {
    
    // null check in wrapper; assumed not null
    // heres an idea, see if it helps. load zobrist, update zobrist, store zobrist,
    // instead of constantly l/s-ing it

    b->gamestate = getGamestateFromUndo(undo);
    uint64_t* bitboards = b->bitboards;  // for convenience
    Piece* pieces = b->pieces;

    Move move = getMoveFromUndo(undo);

    // all this can be so far with replacing gs from gs in the undo64

    // using history struct, reconscruct based on move, hash, gamestate, undo
    // bitboards
    Square src = getSrc(move);
    Square dst = getDst(move);
    Piece capturedPiece = getCapturedPiece(move);  // can get from Move
    Piece srcPiece = getPieceOnSquare(b, dst);  // since we are undoing, the piece on the destination square is the piece that moved unless its promo
    Square epSquare = getEnPassant(move);
    uint64_t srcMask = squareBitboards[src];
    uint64_t dstMask = squareBitboards[dst];

    bool isCapture = capturedPiece != EMPTY;

    // toggle bits xor
    bitboards[getBitboardIndex(srcPiece)] ^= (srcMask | dstMask);  // toggle piece src and dst
    pieces[src] = srcPiece;  // update piece array
    if (!epSquare) pieces[dst] = capturedPiece;
    else pieces[dst] = EMPTY;

    // restore src dst zobrist
    b->zobrist ^= getZobristHash(srcPiece, src);  // add piece to source square
    b->zobrist ^= getZobristHash(srcPiece, dst);  // remove piece from destination square


    ///TODO: the order in which we check these conditions might be optimizable. castling happens rarely,
    // so what is the true tradeoff between checking it now or checking it later, if we may possibly return early?
    // etc. 

    // handle castling undo
    if (isCastled(move)) {
        
        if (dst == g1) {  // white kingside

            bitboards[iWR] ^= (squareBitboards[h1] | squareBitboards[f1]);
            pieces[f1] = EMPTY;
            pieces[h1] = WR;
            //setCastlingRights(&b->gamestate, getCastlingRights(b->gamestate) | whiteLongCastleMask);  // restore white kingside castling right
            b->zobrist ^= getZobristHash(WR, h1) ^ getZobristHash(WR, f1);  // move rook from h1 to f1
            b->zobrist ^= getZobristHash(WK, e1) ^ getZobristHash(WK, g1);  // move king from e1 to g1  
            b->zobrist ^= getZobristCastleHash(whiteLongCastleMask);  // update castling hash for white kingside

        } else if (dst == c1) {  // white queenside

            bitboards[iWR] ^= (squareBitboards[a1] | squareBitboards[d1]);
            pieces[d1] = EMPTY;
            pieces[a1] = WR;
            //setCastlingRights(&b->gamestate, getCastlingRights(b->gamestate) | whiteShortCastleMask);  // restore white queenside castling right
            b->zobrist ^= getZobristHash(WR, a1) ^ getZobristHash(WR, d1);  // move rook from a1 to d1
            b->zobrist ^= getZobristHash(WK, e1) ^ getZobristHash(WK, c1);  // move king from e1 to c1
            b->zobrist ^= getZobristCastleHash(whiteShortCastleMask);  // update castling hash for white queenside

        } else if (dst == g8) {  // black kingside

            bitboards[iBR] ^= (squareBitboards[h8] | squareBitboards[f8]);
            pieces[f8] = EMPTY;
            pieces[h8] = BR;
            //setCastlingRights(&b->gamestate, getCastlingRights(b->gamestate) | blackLongCastleMask);  // restore black kingside castling right
            b->zobrist ^= getZobristHash(BR, h8) ^ getZobristHash(BR, f8);  // move rook from h8 to f8
            b->zobrist ^= getZobristHash(BK, e8) ^ getZobristHash(BK, g8);  // move king from e8 to g8
            b->zobrist ^= getZobristCastleHash(blackLongCastleMask);  // update castling hash for black kingside

        } else if (dst == c8) {  // black queenside

            bitboards[iBR] ^= (squareBitboards[a8] | squareBitboards[d8]);
            pieces[d8] = EMPTY;
            pieces[a8] = BR;
            //setCastlingRights(&b->gamestate, getCastlingRights(b->gamestate) | blackShortCastleMask);  // restore black queenside castling right
            b->zobrist ^= getZobristHash(BR, a8) ^ getZobristHash(BR, d8);  // move rook from a8 to d8
            b->zobrist ^= getZobristHash(BK, e8) ^ getZobristHash(BK, c8);  // move king from e8 to c8
            b->zobrist ^= getZobristCastleHash(blackShortCastleMask);  // update castling hash for black queenside
        }

        goto finishUndo;  // no more zobrist updates needed
    }

    // restore zobrist
    // we can literally copy and paste this from the makeMove function because g^2 = 0

    if (isCapture) {
        bitboards[getBitboardIndex(capturedPiece)] ^= dstMask;
        b->zobrist ^= getZobristHash(capturedPiece, dst);  // add captured piece to destination square
    }

    uint8_t promo = getPromotion(move);
    if (promo) {
        Piece promoPiece = srcPiece;  // this is what was on dst at first
        uint8_t black = getPiecesColour(srcPiece) << 3;
        srcPiece = (Piece) PAWN | black;
        bitboards[getBitboardIndex(promoPiece)] ^= srcMask;  // remove promotion piece from src square (or'd above)
        bitboards[getBitboardIndex(srcPiece)] ^= srcMask;    // pawn is restored, does not happen above because srcPiece is only known here
        pieces[dst] = capturedPiece;  // restore captured piece to destination square
        pieces[src] = srcPiece;  // program only looks at dst square in general, which is not a pawn ever
        b->zobrist ^= getZobristHash(promoPiece, dst);  // remove promotion piece from destination square
        // does more zobrist need to happen? idk
        goto finishUndo;  // enpassant never happens
    }

    
    if (epSquare) {
        bool blackToMove = isBlackToMove(b->gamestate);
        Piece epCapturedPiece = blackToMove ? WP : BP;  // because if the piece that moved is black, the captured piece must be a white pawn
        bitboards[getBitboardIndex(epCapturedPiece)] ^= (squareBitboards[epSquare + (blackToMove ? 8 : -8)] | dstMask);  // add captured pawn to en passant square and remove restore from normal capture block
        pieces[epSquare + (blackToMove ? 8 : -8)] = epCapturedPiece;  // place the captured pawn back
        b->zobrist ^= getZobristHash(epCapturedPiece, epSquare);  // add captured pawn to en passant square
        b->zobrist ^= getZobristEnPassantHash(epSquare & 7);  // update en passant hash, & 7 is to get file of epSquare

    }

    finishUndo:
    updateBoardUnions(b);
    return;

}

void unmove(Board* b, Move move) {
    Undo64 undo = createUndo64(move, b->gamestate);
    performUndo(b, undo);
}

// commands:
int handleUndo(Board* b, Undo64 undo) {

    // if undo is null, use stack undo
    if (undo == NULL) {
        if (b->ply <= 0) {
            fprintf(stderr, "Error: No moves to undo\n");
            return 1;
        }
        else {
            // pop
            undo = b->undoStack[--b->ply];
        }
    }

    performUndo(b, undo);
    return 0;
}


void handlePerft(Board* b);
void handleChildren(Board* b);

// bool pushUndoToStack(Board* b, Undo* undo) {
//     if (b->history->ply >= MAX_PLY) {
//         fprintf(stderr, "Error: Maximum undo stack size reached\n");
//         return false;
//     }

//     b->history->undoHistory[b->history->ply] = *undo;
//     b->history->ply++;
//     return true;
// }

bool pushUndo64ToStack(Board* b, Undo64 undo) {
    if (b->ply >= MAX_PLY) {
        fprintf(stderr, "Error: Maximum undo stack size reached\n");
        return false;
    }

    b->undoStack[b->ply++] = undo;
    return true;
}

