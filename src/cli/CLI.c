#include "CLI.h"

/*
CMDS: 
help
fen <string> / startpos
d (display board)
moves (print legal moves in UCI)
perft <depth>
go <depth> or go movetime <ms>
undo / redo
history
eval - likely a 0 for now, or greedy. doesnt matter atm
hash (print zobrist)
att <sq> (print attacks to/from a square)
pins (print pinned pieces mask)
checkers (print checkers mask)
quit



SPEC:

History should store:
Move
Undo
zobrist after move (or before, either is fine if consistent)

make_move should:
update board state, bitboards, piece[64], castling/ep/halfmove, zobrist
return false if move leaves own king in check
undo_last_move pops history and calls unmake_move

*/



/*
PSEUDO:

function run_game(game):
  ui.render(game.board)

  while true:
    // check terminal conditions BEFORE move
    if is_draw(game):  // repetition, 50-move, insufficient material, stalemate
        game.result = DRAW
        break
    if is_checkmate(game.board):
        game.result = WIN(opposite(side_to_move(game.board)))
        break

    // determine whose turn it is
    stm = side_to_move(game.board)
    player = (stm == WHITE) ? game.white : game.black

    // get a move from that player
    if player.type == HUMAN:
        move = get_human_move(game)
        // includes: parse input, validate legal, allow commands (undo, resign, etc.)
    else:
        move = get_engine_move(game, player.engine)

    // if command was issued, handle it
    if move == CMD_UNDO:
        if game.history.size >= 1:
           undo_last_move(game)
           ui.render(game.board)
        continue

    if move == CMD_RESIGN:
        game.result = WIN(opposite(stm))
        break

    if move == CMD_QUIT:
        game.result = ABORTED
        break

    // Apply the move (must be legal)
    ok = make_move(game.board, move, &undo)
    if not ok:
        // if human: tell them invalid and retry
        // if engine: it's a bug (fuck)
        ui.message("Illegal move.")
        continue

    // Record history (for undo, repetition, etc.)
    history_push(game.history, move, undo, game.board.zobrist)

    // render / notify
    ui.render(game.board)

*/

void getCommand();

///TODO: make commands interface