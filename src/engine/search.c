#include "search.h"

uint64_t perft(Board* b, int depth) {

    // get all moves from each piece based on colour, play, unmove, increment counter
    if (depth == 0) {
        return 1;
    }

    uint64_t count = 0;

    Move* move_list = generate_moves(b);
    int i = 0;
    while (move_list[i] != NULL_MOVE) {
        Move move = move_list[i++];
        Undo64 undo = createUndo64(move, b->gamestate);
        makeMove(b, move);
        count += perft(b, depth - 1);
        performUndo(b, undo);
    }

    free(move_list);

    return count;
}

void perft_wrapper(Board* b, int depth) {
    clock_t start = clock();
    uint64_t nodes = perft(b, depth);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Perft to depth %d: %llu nodes in %f seconds\n", depth, (unsigned long long)nodes, time_spent);
}

void perft_iterativeDeepening(Board* b, int maxDepth) {
    for (int depth = 1; depth <= maxDepth; depth++) {
        perft_wrapper(b, depth);
    }
}