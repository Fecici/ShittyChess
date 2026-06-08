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
void perft_root(Board* b, int depth) {
    Move moves[MAX_MOVES] = {0};
    uint64_t total = 0;

    generate_moves(b, moves);

    clock_t totalStart = clock();

    for (int i = 0; moves[i] != NULL_MOVE; i++) {
        Move move = moves[i];
        Undo64 undo = createUndo64(move, b->gamestate);

        clock_t moveStart = clock();

        makeMove(b, move);
        uint64_t nodes = perft(b, depth - 1);
        performUndo(b, undo);

        double moveSeconds = (double) (clock() - moveStart) / CLOCKS_PER_SEC;
        total += nodes;

        printf("%s%s: %llu | %.7fs | total: %llu\n",
              squareChar[getSrc(move)],
              squareChar[getDst(move)],
              (unsigned long long)nodes,
              moveSeconds,
              (unsigned long long)total);
        fflush(stdout);
    }

    double totalSeconds = (double)(clock() - totalStart) / CLOCKS_PER_SEC;
    printf("Perft depth %d: %llu nodes in %.7fs\n",
        depth,
        (unsigned long long)total,
        totalSeconds);
}

static void perft_collect(
    Board* b,
    int currentDepth,
    int maxDepth,
    uint64_t* nodes
    ) {
    if (currentDepth == maxDepth) {
        return;
    }

    Move moves[MAX_MOVES] = {0};
    generate_moves(b, moves);

    for (int i = 0; moves[i] != NULL_MOVE; i++) {
        Move move = moves[i];
        Undo64 undo = createUndo64(move, b->gamestate);

        nodes[currentDepth + 1]++;
        if (currentDepth == maxDepth - 1 && nodes[maxDepth] % 1000000 == 0) {
            printf("\rDepth %d: %llu nodes searched",
                   maxDepth,
                   (unsigned long long)nodes[maxDepth]);
            fflush(stdout);
        }

        makeMove(b, move);
        perft_collect(b, currentDepth + 1, maxDepth, nodes);
        performUndo(b, undo);
    }
}

void perft_iterativeDeepening2(Board* b, int maxDepth) {
    uint64_t nodes[MAX_DEPTH + 1] = {0};

    clock_t start = clock();

    perft_collect(b, 0, maxDepth, nodes);

    double seconds =
        (double)(clock() - start) / CLOCKS_PER_SEC;

    for (int depth = 1; depth <= maxDepth; depth++) {
        printf(
            "Perft to depth %d: %llu nodes\n",
            depth,
            (unsigned long long)nodes[depth]
        );
    }

    printf("Completed through depth %d in %.7f seconds\n",
            maxDepth, seconds);
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

    //perft_iterativeDeepening2(b, maxDepth);
    //perft_root(b, maxDepth);
    for (int depth = 1; depth <= maxDepth; depth++) {
        perft_wrapper(b, depth);
    }
}