#include "legal.h"

uint64_t opponentAttacks(Board* b, Colour blackToMove) {
    uint64_t enemyTargets = 0;
    Colour enemyColour = BLACK - blackToMove;

    // or all attacks
    // need the opposite side of the king's colour
    for (int i = 6 * enemyColour; i < 6 + 6 * enemyColour; i++) {
        uint64_t pieceBitboard = b->bitboards[i];
        while (pieceBitboard) {
            uint64_t moveMask = pieceBitboard & -pieceBitboard;  // get least significant bit
            pieceBitboard &= pieceBitboard - 1;  // clear least significant bit

            Square src = (Square) __builtin_ctzll(moveMask);  // get source square

            enemyTargets |= attackGenerator[i](b, src);  // add attacks from this piece to enemyTargets
        }
    }

    return enemyTargets;
}

bool kingInCheck(Board* b, Colour blackToMove, uint64_t enemyTargets) {
    uint64_t kingBitboard = b->bitboards[(blackToMove == WHITE) ? iWK : iBK];

    return kingBitboard & enemyTargets;
}

bool castleShortInCheck(Board* b, Colour blackToMove, uint64_t enemyTargets) {
    if (blackToMove == WHITE) {
        return (squareBitboards[e1] | squareBitboards[f1] | squareBitboards[g1]) & enemyTargets;
    } else {
        return (squareBitboards[e8] | squareBitboards[f8] | squareBitboards[g8]) & enemyTargets;
    }
}

bool castleLongInCheck(Board* b, Colour blackToMove, uint64_t enemyTargets) {
    if (blackToMove == WHITE) {
        return (squareBitboards[e1] | squareBitboards[d1] | squareBitboards[c1]) & enemyTargets;
    } else {
        return (squareBitboards[e8] | squareBitboards[d8] | squareBitboards[c8]) & enemyTargets;
    }
}

bool isLegalMove(Board* b, Move move) {
    // for now, play, check king, unmove

    // better: make an isSquareAttacked function that can return early

    Gamestate gamestate = b->gamestate;

    bool blackToMove = isBlackToMove(gamestate);

    Undo64 undo = createUndo64(move, gamestate);
    
    uint64_t enemyTargets;
    if (isCastled(move)) {
        enemyTargets = opponentAttacks(b, blackToMove);
        if (blackToMove) {
            if (getDst(move) == g8 && castleShortInCheck(b, blackToMove, enemyTargets)) {
                return false;
            }
            if (getDst(move) == c8 && castleLongInCheck(b, blackToMove, enemyTargets)) {
                return false;
            }
        } 
        else {
            if (getDst(move) == g1 && castleShortInCheck(b, blackToMove, enemyTargets)) {
                return false;
            }
            
            if (getDst(move) == c1 && castleLongInCheck(b, blackToMove, enemyTargets)) {
                return false;
            }
        }
    }

    makeMove(b, move);  // now white to move after this if black before

    // check if the king is in check after the move
    // needs to be done again
    enemyTargets = opponentAttacks(b, blackToMove);
    if (kingInCheck(b, blackToMove ? BLACK : WHITE, enemyTargets)) {
        // revert the move
        performUndo(b, undo);
        return false;
    }

    // revert the move
    performUndo(b, undo);
    return true;
}

uint64_t getLegalFromPseudo(Board* b, uint64_t pseudoMoves, Square src) {

    uint64_t legalMoves = 0;
    Piece piece = b->pieces[src];
    uint8_t promo = ((piece == WP && squareBitboards[src] & rank7) || (piece == BP && squareBitboards[src] & rank2)) ? promoQueen : 0;  // for now, we will not generate promo moves, so this is just a placeholder. we can set this to the correct promo type when we implement promo move generation

    // play each move (each bit) and or it to 0 if legal
    while (pseudoMoves) {
        uint64_t moveMask = pseudoMoves & -pseudoMoves;  // get least significant bit
        pseudoMoves &= pseudoMoves - 1;  // clear least significant bit

        Square dst = (Square) __builtin_ctzll(moveMask);  // get dst

        // promo decision does not happen here. Decision does not affect legality
        Move move = getMoveFromSquare(b, src, dst, promo);
        if (isLegalMove(b, move)) {
            legalMoves |= moveMask;  // add move to legal moves
        }   
    }

    return legalMoves;
}

int handleMakeMove(Board* b, Move move) {
    
    // check colour to move, check legality unless forced (we just set the pieces there then and go around this function entirely)
    bool colourToMove = isBlackToMove(b->gamestate);
    Piece piece = getPieceOnSquare(b, getSrc(move));
    if (piece == EMPTY) {
        fprintf(stderr, "Illegal move: no piece on source square\n");
        return 1;
    }

    if (getPiecesColour(piece) != colourToMove) {
        fprintf(stderr, "Illegal move: piece on source square does not match colour to move\n");
        return 1;
    }

    uint64_t moveMask = squareBitboards[getDst(move)];
    uint64_t legal = pieceGenerator[getBitboardIndex(piece)](b, getSrc(move));
    legal = getLegalFromPseudo(b, legal, getSrc(move));

    if (!(legal & moveMask)) {
        fprintf(stderr, "Illegal move: move is not legal in the current position\n");
        return 1;
    }

    Undo64 undo = createUndo64(move, b->gamestate);
    pushUndo64ToStack(b, undo);

    makeMove(b, move);
    return 0;
}


void generate_moves(Board* b, Move* move_list) {
    // this is the main movegen function that will be called by the search and perft functions. it will call the piece specific movegen functions and return a list of moves.

    // for (int i = 0; i < MAX_MOVES; i++) {
    //     move_list[i] = NULL_MOVE;
    // }  // already done
    // loop through pieces of the active colour, generate moves for each piece, add to move list
    
    int move_count = 0;

    Colour us = getColourToMove(b->gamestate);
    Colour them = (us == WHITE) ? BLACK : WHITE;

    // loop through pieces of the active colour, generate moves for each piece, add to move list
    for (int piece_index = 0; piece_index < 6; piece_index++) {
        uint64_t pieces = b->bitboards[(us == WHITE) ? piece_index : piece_index + 6];
        while (pieces) {
            Square src = (Square) __builtin_ctzll(pieces);
            uint64_t moves = pieceGenerator[(us == WHITE) ? piece_index : piece_index + 6](b, src);

            bool promo = ((b->pieces[src] == WP && squareBitboards[src] & rank7) || (b->pieces[src] == BP && squareBitboards[src] & rank2)) ? true : false;
            // sanitize legal
            moves = getLegalFromPseudo(b, moves, src);
            // convert moves bitboard to move list entries
            while (moves) {
                Square dst = (Square) __builtin_ctzll(moves);
                if (promo) {
                    move_list[move_count++] = getMoveFromSquare(b, src, dst, promoQueen);
                    move_list[move_count++] = getMoveFromSquare(b, src, dst, promoKnight);
                    move_list[move_count++] = getMoveFromSquare(b, src, dst, promoBishop);
                    move_list[move_count++] = getMoveFromSquare(b, src, dst, promoRook);


                    moves &= moves - 1;
                    continue;
                }
                move_list[move_count++] = getMoveFromSquare(b, src, dst, 0);
                moves &= moves - 1;
            }

            pieces &= pieces - 1;
        }
    }

    //return move_list;
}