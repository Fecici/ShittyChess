#include "main.h"

#ifdef _WIN32
#include <windows.h>
#endif

char* const startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/*https://gist.github.com/peterellisjones/8c46c28141c162d1d8a0f0badbc9cff9*/
const char* testFens[] = {
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1",
    "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1",
    "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1",
    "5k2/8/8/8/8/8/8/4K2R w K - 0 1",
    "3k4/8/8/8/8/8/8/R3K3 w Q - 0 1",
    "r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1",
    "r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1",
    "2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1",
    "8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1",
    "4k3/1P6/8/8/8/8/K7/8 w - - 0 1",
    "8/P1k5/K7/8/8/8/8/8 w - - 0 1",
    "K1k5/8/P7/8/8/8/8/8 w - - 0 1",
    "8/k1P5/8/1K6/8/8/8/8 w - - 0 1",
    "8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1"
};

int main(int argc, char** argv) {

    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    char* fen;
    fen = startFen;  // default

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --help, -h       Show this help message and exit\n");
            printf("  --fen <FEN>     Start a game with the given FEN string\n");
            return 0;
        }

        if (strcmp(argv[i], "--fen") == 0) {
            if (i + 1 < argc) {

                if (validFen(argv[i + 1])) {

                    fen = argv[i + 1];

                } else if (isdigit(argv[i + 1][0])) {

                    int fenIndex = strtol(argv[i + 1], NULL, 10);
                    
                    if (fenIndex >= 0 && fenIndex < sizeof(testFens) / sizeof(testFens[0])) {

                        fen = (char*)testFens[fenIndex];

                    } else {
                        fprintf(stderr, "Error: Invalid FEN index\n");
                        return 1;
                    }
                } else {
                    fprintf(stderr, "Error: Invalid FEN string\n");
                    return 1;
                }
                i++; // Skip the FEN argument

            } else {
                fprintf(stderr, "Error: --fen option requires a FEN string argument or number\n");
                return 1;
            }
        }
    }

    // init board

    // init game state

    // init pieces

    // init "clock"

    // init "players"

    // print board

    // enter game loop


    // on exit, print exit information

    Player white = {HUMAN, WHITE, NULL};
    Player black = {HUMAN, BLACK, NULL};

    Game* game = initGame(fen, white, black, HvH);
    cliMainLoop(game, NULL);
    

    return 0;
}
