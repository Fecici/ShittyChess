#include "cli.h"

Game* game;  // this will hold the globals we need

 // init all, setup history, ui, etc.
Game* initGame(char* fen, Player white, Player black, GameType gt) {

    /*
    typedef struct {

    UI ui;
    Board* board;
    Player white, black;
    unsigned int moves;  // 2ply = 1 move
    unsigned int ply;
    /// TODO: time control eventually
    unsigned int whiteTime, blackTime;
    uint8_t gameResult;  // format to be defined, but basically its a flag that describes how the game ended
    History history;
} Game;*/

    initZobrist();

    if (game == NULL) {
        game = (Game*) calloc(1, sizeof(Game));
    }

    if (game == NULL) {fprintf(stderr, "Failed to allocate memory for game.\n"); exit(1); }

    Board* b = (Board*) calloc(1, sizeof(Board));

    if (b == NULL) {fprintf(stderr, "Failed to allocate memory for board.\n"); exit(1); }

    game->board = b;

    game->white = white;
    game->black = black;
    //printf(fen);
    if (fen != NULL && !loadFromFen(b, fen)) {
        fprintf(stderr, "Failed to parse fen string: %s\n", fen);
        exit(1);
    }

    unsigned int ply = b->ply;
    game->moves = (ply >> 1) + 1;

    History* h = calloc(1, sizeof(History));
    b->history = h;

    game->whiteTime = -1;
    game->blackTime = -1;

    game->gameResult = 0;

    UI ui;
    initUI(&ui, "CLI", gt, ascii_render, stdout_messager);

    game->ui = ui;

    return game;
}

// terminal functions
bool checkTermination(Board* b) {

    (void) b;

    return false;
}

void handleStalemate(Board* b) {
    printf("\n\n\n");
    printBoard(b);
    printf("Stalemate: 0.5 -- 0.5.\n");
    exit(0);  // new game maybe another time (make this function return a bool i guess)
}
void handleCheckmate(Board* b) {

    int white = 0;
    int black = 0;

    // get gamestate then print w or b

    printf("\n\n\n");
    printBoard(b);
    printf("Checkmate! %d -- %d.\n", white, black);
    exit(0);  // new game maybe another time (make this function return a bool i guess)

}

void cliMainLoop(Game* g, void (*performCommand)(Board* b)) {

    game = g;  // set our global game ptr to the one passed in
    setCommandGame(g);
    (void) performCommand;  // for now

    int nCmds = getCommandCount();

    
    char input[MAX_STDIN];
    int argc = 0;
    char* argv[MAX_ARG];

    bool terminationDebug = false;

    printBoard(game->board);

    while (true) {

        if (terminationDebug) (checkTermination(game->board));

        getInput(input, (size_t) MAX_STDIN);

        argc = tokenize(input, argv);
        if (!argc) continue;

        CommandAbstract* cmd = getCommand(input, nCmds);

        if (!cmd) {fprintf(stderr, "Command not found: \"%s\"\n", input); continue;}  // since we tokenized input, this only prints the name

        // perform cmd
        if (cmd->cmd(argc, argv) < 0) fprintf(stderr, "Something went wrong...\n");

        // Player player = (isBlackToMove(board->gameState)) ? black : white;

        // Move move = getMove(board, player);

        // performCommand(board);  // i am not sure how to implement this still

        // performMove(board, move);

    }


}
void handleQuit() {exit(0);}
