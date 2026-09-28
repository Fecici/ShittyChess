#include "engine.h"


int alphaBetaMax(int alpha, int beta, int depth, Board* b) {

    if (depth == 0) {
        return evaluateBoard(b);
    }

    Move move_list[MAX_MOVES] = {0};
    generate_moves(b, move_list);
    int current_best = INT_MIN;

    int i = 0;
    while (move_list[i] != NULL_MOVE) {
        Move move = move_list[i++];

        Undo64 undo = createUndo64(move, b->gamestate);

        makeMove(b, move);
        int score = alphaBetaMin(alpha, beta, depth - 1, b);
        performUndo(b, undo);

        if (score > current_best) {
            current_best = score;

            if (score > alpha) {
                alpha = score;
            }
        }

        if (score >= beta) {
            return score;
        }
    
    }

    return current_best;

}

int alphaBetaMin(int alpha, int beta, int depth, Board* b) {

    if (depth == 0) {
        return -evaluateBoard(b);
    }

    Move move_list[MAX_MOVES] = {0};
    generate_moves(b, move_list);
    int current_best = INT_MAX;

    int i = 0;
    while (move_list[i] != NULL_MOVE) {
        Move move = move_list[i++];
        Undo64 undo = createUndo64(move, b->gamestate);
        makeMove(b, move);
        int score = alphaBetaMax(alpha, beta, depth - 1, b);
        performUndo(b, undo);
    
        if (score < current_best) {

            current_best = score;

            if (score < beta) {
                beta = score;
            }
        }

        if (score <= alpha) {
            return score;
        }
    }

    return current_best;

}


Move getEngineMove(Board* b, int depth) {
    
    int current_best = INT_MIN;

    Move move_list[MAX_MOVES] = {0};

    generate_moves(b, move_list);

    Move bestMove = move_list[0];

    int i = 0;

    while (move_list[i] != NULL_MOVE) {

        Move move = move_list[i++];
        Undo64 undo = createUndo64(move, b->gamestate);

        makeMove(b, move);
        int score = alphaBetaMax(INT_MIN, INT_MAX, depth, b);
        performUndo(b, undo);

        if (score > current_best) {
            current_best = score;
            bestMove = move;
        }

    }

    return bestMove;
}
