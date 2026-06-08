#include "search.h"

uint64_t perft(Board* b, int depth) {

    // get all moves from each piece based on colour, play, unmove, increment counter
    if (depth == 0) {
        return 1;
    }

    uint64_t count = 0;
    Move move_list[MAX_MOVES] = {0};
    generate_moves(b, move_list);
    // better method:
    // generate pseudo
    // save undo
    // play pseudo
    // if pseudo was legal, recurse.
    // undo.
    int i = 0;
    while (move_list[i] != NULL_MOVE) {
        Move move = move_list[i++];
        Undo64 undo = createUndo64(move, b->gamestate);
        makeMove(b, move);
        count += perft(b, depth - 1);
        performUndo(b, undo);
    }

    return count;
}

void perft_wrapper(Board* b, int depth) {

    // lets also print divides for each child of the pos, eg e2e4: 20, ...

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