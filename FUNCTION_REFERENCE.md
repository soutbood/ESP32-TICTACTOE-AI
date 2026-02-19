# Tic Tac Toe AI - Complete Function Reference

## Table of Contents
1. [Setup & Main Loop](#setup--main-loop)
2. [Memory Management](#memory-management)
3. [Neural Network Functions](#neural-network-functions)
4. [Game Logic Functions](#game-logic-functions)
5. [Training Functions](#training-functions)
6. [Web Server Functions](#web-server-functions)
7. [Serial Command Functions](#serial-command-functions)
8. [Weight Export/Import Functions](#weight-exportimport-functions)
9. [Function Call Flow Diagrams](#function-call-flow-diagrams)

---

## Setup & Main Loop

### `setup()`

**Purpose:** Initializes the entire system when the ESP32 boots up.

**What it does:**
1. Starts Serial communication at 115200 baud
2. Prints startup banner with configuration info:
   - Max players count
   - Embedded weights status (ENABLED/DISABLED)
3. Sets up WiFi Access Point via `setup_wifi_ap()`
4. Initializes PSRAM (if available)
5. Allocates memory for neural network and player data via `allocate_psram_memory()`
6. Mounts LittleFS filesystem
7. Loads weights in priority order:
   - Try filesystem first (`load_weights_from_flash()`)
   - Try embedded weights (`load_embedded_weights()`)
   - Fall back to random initialization (`initialize_neural_network()`)
8. Seeds random number generator with analog read
9. Sets up web server via `setup_web_server()`
10. Starts initial game via `initialize_game(false)`

**Called:** Once at boot

**Key variables set:**
- `games_played = 0`
- `ai_wins = 0`, `player_wins = 0`, `random_wins = 0`
- `fast_training_mode = false`
- `selfplay_training_mode = false`

---

### `loop()`

**Purpose:** Main execution loop that runs continuously.

**What it does:**

1. **Player Timeout Check:**
   - Iterates through all player slots
   - If player inactive for 60 seconds, clears their slot

2. **Serial Command Processing:**
   - Only processes when `game_over` or `player_turn`
   - Flushes serial buffer during AI turn

3. **Training Mode Handling:**
   - If `fast_training_mode == true`: calls `fast_train_session()`
   - If `selfplay_training_mode == true`: calls `fast_train_self_play_session()`

4. **Normal Game Handling:**
   - If `!game_over` and `!player_turn`: calls `ai_turn_logic()`

5. **Delay:** 50ms between iterations

**Called:** Continuously after `setup()`

**State machine:**
```
┌─────────────────┐
│   Training?     │
└────────┬────────┘
    Yes  │  No
    ┌────┴─────┐
    │          │
    v          v
Training   Normal Game
  Loop        Loop
```

---

## Memory Management

### `allocate_psram_memory()`

**Purpose:** Allocates all dynamic memory for the neural network and player data.

**What it does:**

1. **Player Data Allocation (MAX_PLAYERS × 20 bytes):**
   - Tries PSRAM first via `ps_malloc()`
   - Falls back to regular `malloc()` if PSRAM unavailable
   - Initializes all player slots:
     - `active = 0` (inactive)
     - `ip = 0`
     - `player_sym = PLAYER_X`
     - Clears board array
     - `last_seen = 0`

2. **Neural Network Allocation:**
   - Tries PSRAM first
   - Falls back to regular RAM
   - Allocates:
     | Array | Size | Bytes |
     |-------|------|-------|
     | `input_weights` | 27 × 54 | 5,832 |
     | `hidden_weights` | 54 × 9 | 1,944 |
     | `input_bias` | 54 | 216 |
     | `hidden_bias` | 9 | 36 |
     | `input_vector` | 27 | 108 |
     | `hidden_output` | 54 | 216 |
     | `output_vector` | 9 | 36 |
     | `game_history` | 20 moves | 2,240 |

3. **Serial Output:** Reports allocation location (PSRAM or RAM)

**Total Memory:** ~11 KB

**Called:** Once during `setup()`

---

## Neural Network Functions

### `initialize_neural_network()`

**Purpose:** Randomly initializes all neural network weights.

**What it does:**
- Sets input weights to random values between -0.1 and 0.1:
  ```cpp
  input_weights[i] = (random(200) - 100) / 1000.0f;
  ```
- Sets hidden weights to random values between -0.1 and 0.1
- Sets all biases to 0.0
- Prints confirmation message

**Used when:** No saved weights exist (first run or reset)

**Weight ranges:**
- Input weights: [-0.1, 0.1]
- Hidden weights: [-0.1, 0.1]
- Input bias: 0.0
- Hidden bias: 0.0

---

### `load_weights_from_flash()`

**Purpose:** Loads trained weights from LittleFS filesystem.

**What it does:**
1. Opens `/tictactoe_weights.dat` for reading
2. Reads 4 binary blobs sequentially:
   - Input weights: 1,458 floats (5,832 bytes)
   - Hidden weights: 486 floats (1,944 bytes)
   - Input bias: 54 floats (216 bytes)
   - Hidden bias: 9 floats (36 bytes)
3. Closes file
4. Validates total bytes match expected (8,028 bytes)
5. Prints bytes loaded vs expected
6. Returns `true` if all bytes read successfully

**Returns:** `bool` - success/failure

**Called by:** `setup()`

**File format:** Raw binary (little-endian floats)

---

### `save_weights_to_flash()`

**Purpose:** Saves current weights to LittleFS for persistence across resets.

**What it does:**
1. Opens `/tictactoe_weights.dat` for writing (creates/overwrites)
2. Writes 4 binary blobs:
   - Input weights
   - Hidden weights
   - Input bias
   - Hidden bias
3. Closes file
4. Prints "Weights saved to flash"
5. Returns `true` if successful

**Called:**
- Every 500 games during normal play
- When training stops
- When `stop` command issued

**File size:** 8,028 bytes

---

### `load_embedded_weights()`

**Purpose:** Loads pre-trained weights from `ai_weights.h` (stored in flash/PROGMEM).

**What it does:**
1. Prints "Loading embedded weights from flash..."
2. Verifies array size:
   ```cpp
   expected = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE;
   // = 1458 + 486 + 54 + 9 = 2,007 values
   ```
3. Uses `pgm_read_float()` to read from program memory (flash)
4. Copies to RAM arrays in order:
   - Input weights (1,458 values, indices 0-1457)
   - Hidden weights (486 values, indices 1458-1943)
   - Input bias (54 values, indices 1944-1997)
   - Hidden bias (9 values, indices 1998-2006)
5. Prints confirmation with total values loaded

**Returns:** `bool` - success/failure

**Conditional:** Only compiled when `USE_EMBEDDED_WEIGHTS == 1`

---

### `sigmoid(float x)`

**Purpose:** Activation function for neural network neurons.

**Formula:** 
```
σ(x) = 1 / (1 + e^(-x))
```

**What it does:**
- Clamps extreme values to prevent floating-point overflow:
  - If `x > 10.0`: returns 1.0
  - If `x < -10.0`: returns 0.0
- Otherwise computes standard sigmoid

**Returns:** Value in range [0, 1]

**Used in:** `run_neural_network()` (both hidden and output layers)

**Properties:**
- `sigmoid(0) = 0.5`
- `sigmoid(positive) > 0.5`
- `sigmoid(negative) < 0.5`

---

### `run_neural_network()`

**Purpose:** Executes forward pass through the neural network.

**What it does:**

**Layer 1: Input → Hidden (27 → 54)**
```cpp
for(int i = 0; i < HIDDEN_SIZE; i++) {
  float sum = input_bias[i];
  for(int j = 0; j < INPUT_SIZE; j++) {
    sum += input_vector[j] * input_weights[j * HIDDEN_SIZE + i];
  }
  hidden_output[i] = sigmoid(sum);
}
```

**Layer 2: Hidden → Output (54 → 9)**
```cpp
for(int i = 0; i < OUTPUT_SIZE; i++) {
  float sum = hidden_bias[i];
  for(int j = 0; j < HIDDEN_SIZE; j++) {
    sum += hidden_output[j] * hidden_weights[j * OUTPUT_SIZE + i];
  }
  output_vector[i] = sigmoid(sum);
}
```

**Result:** `output_vector[0-8]` contains scores for each board position (0-8)

**Called by:**
- `ai_select_move()`
- `ai_select_move_player()`
- `record_ai_move()`
- `record_selfplay_move()`
- `update_network_for_move()`

---

### `update_weights_td_learning(bool game_won)`

**Purpose:** Updates neural network weights using Temporal Difference (TD) learning after a game ends.

**What it does:**

1. **Set Final Reward:**
   - Win: `final_reward = 1.0`
   - Draw: `final_reward = 0.5`
   - Loss: `final_reward = 0.0`

2. **Count Recorded Moves:**
   - Iterates through `game_history` until `move_made == -1`

3. **Backpropagate TD Error:**
   - For each move (backwards from last):
     ```cpp
     target = (i == num_moves - 1) ? final_reward : game_history[i + 1].move_value_before;
     td_error = target - game_history[i].move_value_before;
     update_network_for_move(game_history[i].board_state, game_history[i].move_made, td_error);
     ```

**Learning principle:**
- Last move's target = game outcome
- Earlier moves' target = value of next state
- TD error = prediction error to correct

**Called by:** `end_game()`, `fast_train_session()`

---

### `update_network_for_move(float* board_state, int move_idx, float td_error)`

**Purpose:** Applies TD error to update weights for a specific move.

**What it does:**

1. **Set Input Vector:**
   ```cpp
   for(int i = 0; i < INPUT_SIZE; i++) input_vector[i] = board_state[i];
   ```

2. **Run Forward Pass:**
   ```cpp
   run_neural_network();
   ```

3. **Update Hidden→Output Weights:**
   ```cpp
   for(int i = 0; i < HIDDEN_SIZE; i++) {
     float grad = LEARNING_RATE * td_error * hidden_output[i];
     hidden_weights[i * OUTPUT_SIZE + move_idx] += grad;
   }
   ```

4. **Update Input→Hidden Weights (Backpropagation):**
   ```cpp
   for(int i = 0; i < INPUT_SIZE; i++) {
     for(int j = 0; j < HIDDEN_SIZE; j++) {
       float out_err = hidden_weights[j * OUTPUT_SIZE + move_idx] * td_error;
       float grad = LEARNING_RATE * out_err * input_vector[i] * 
                    hidden_output[j] * (1.0f - hidden_output[j]);
       input_weights[i * HIDDEN_SIZE + j] += grad;
     }
   }
   ```

**Parameters:**
- `board_state`: 27-element input vector
- `move_idx`: Position (0-8) the AI chose
- `td_error`: Prediction error to correct

**Learning rate:** 0.01 (defined by `LEARNING_RATE`)

---

### `board_to_input_vector()`

**Purpose:** Converts current global `board[]` state to neural network input format.

**What it does:**

For each board position (0-8):
```cpp
if(board[i] == EMPTY) {
  input_vector[i*3] = 1.0f; input_vector[i*3+1] = 0.0f; input_vector[i*3+2] = 0.0f;
} else if(board[i] == PLAYER_X) {
  input_vector[i*3] = 0.0f; input_vector[i*3+1] = 1.0f; input_vector[i*3+2] = 0.0f;
} else {
  input_vector[i*3] = 0.0f; input_vector[i*3+1] = 0.0f; input_vector[i*3+2] = 1.0f;
}
```

**One-hot encoding:**
- Empty: `[1, 0, 0]`
- Player (X): `[0, 1, 0]`
- AI (O): `[0, 0, 1]`

**Result:** `input_vector[27]` ready for `run_neural_network()`

**Called by:** `ai_select_move()`, `record_ai_move()`, `record_selfplay_move()`, `update_network_for_move()`

---

### `record_ai_move(int move)`

**Purpose:** Records an AI move in game history for later training.

**What it does:**
1. Finds empty slot in `game_history`:
   ```cpp
   for(slot = 0; slot < MAX_GAME_MOVES && game_history[slot].move_made != -1; slot++);
   ```
2. Converts current board to input vector
3. Copies board state to history:
   ```cpp
   for(int i = 0; i < INPUT_SIZE; i++) {
     game_history[slot].board_state[i] = input_vector[i];
   }
   ```
4. Saves move position: `game_history[slot].move_made = move;`
5. Runs neural network
6. Saves the output value for that move: `game_history[slot].move_value_before = output_vector[move];`

**Why:** Training needs to know what the AI expected before seeing the outcome

**Called by:** `ai_turn_logic()`, `fast_train_session()`

---

## Game Logic Functions

### `initialize_game(bool silent_mode)`

**Purpose:** Resets the game state for a new game.

**What it does:**
1. Clears board:
   ```cpp
   for(int i = 0; i < BOARD_SIZE; i++) board[i] = EMPTY;
   ```
2. Clears game history:
   ```cpp
   for(int i = 0; i < MAX_GAME_MOVES; i++) game_history[i].move_made = -1;
   ```
3. Resets game state:
   - `game_over = false`
   - `winner = EMPTY`
   - `player_turn = true`
4. If `!silent_mode`:
   - Prints "New game started!"
   - Calls `print_board()`

**Parameters:**
- `silent_mode`: If `true`, skip serial output (used in training)

**Called by:** `setup()`, `end_game()`, `process_serial_command()`, training functions

---

### `is_valid_move(int position)`

**Purpose:** Checks if a move is legal.

**What it does:**
```cpp
return (position >= 0 && position < BOARD_SIZE && board[position] == EMPTY);
```

**Returns:** `true` if position is 0-8 and empty

**Called by:** `make_move()`, `random_move()`

---

### `make_move(int position, int player)`

**Purpose:** Places a mark on the board.

**What it does:**
```cpp
if(is_valid_move(position)) {
  board[position] = player;
  return true;
}
return false;
```

**Parameters:**
- `position`: Board position (0-8)
- `player`: `PLAYER_X` (1) or `AI_O` (2)

**Returns:** `true` if successful

**Called by:** `ai_turn_logic()`, `fast_train_session()`, `fast_train_self_play_session()`, `process_serial_command()`

---

### `print_board()`

**Purpose:** Displays current board state on Serial Monitor.

**Output format:**
```
Current Board:
 - | X | O 
-----------
 X | O | X 
-----------
 O | X | - 
```

**What it does:**
- Iterates through board in rows of 3
- Prints each position as `-`, `X`, or `O`
- Adds separator lines between rows

**Called by:** `initialize_game()`, `ai_turn_logic()`, `process_serial_command()`

---

### `check_winner()`

**Purpose:** Determines if the game has ended and who won.

**What it does:**

1. **Check 8 Winning Combinations:**
   ```cpp
   int wins[8][3] = {
     {0,1,2}, {3,4,5}, {6,7,8},  // Rows
     {0,3,6}, {1,4,7}, {2,5,8},  // Columns
     {0,4,8}, {2,4,6}            // Diagonals
   };
   ```

2. **Check for Winner:**
   ```cpp
   for(int i = 0; i < 8; i++) {
     if(board[wins[i][0]] != EMPTY && 
        board[wins[i][0]] == board[wins[i][1]] &&
        board[wins[i][0]] == board[wins[i][2]]) {
       return board[wins[i][0]];
     }
   }
   ```

3. **Check for Draw:**
   ```cpp
   for(int i = 0; i < BOARD_SIZE; i++) {
     if(board[i] == EMPTY) return 0;  // Game continues
   }
   return -1;  // Draw
   ```

**Returns:**
- `PLAYER_X` (1): Player won
- `AI_O` (2): AI won
- `-1`: Draw
- `0`: Game continues

**Called by:** `is_game_over()`, `end_game()`, `fast_train_self_play_session()`

---

### `is_game_over()`

**Purpose:** Checks and updates game state.

**What it does:**
```cpp
int result = check_winner();
if(result != EMPTY) {
  game_over = true;
  winner = result;
  return true;
}
return false;
```

**Returns:** `true` if game ended

**Called by:** `ai_turn_logic()`, `fast_train_session()`, `process_serial_command()`

---

### `ai_turn_logic()`

**Purpose:** Handles the AI's turn in normal gameplay.

**What it does:**
1. Prints "AI is thinking..."
2. Calls `ai_select_move()` to get best move
3. If valid move:
   - Calls `record_ai_move()` to save for training
   - Calls `make_move()` to place mark
   - Prints move position
   - Calls `print_board()`
   - Calls `is_game_over()`:
     - If true: calls `end_game()`
     - If false: sets `player_turn = true`
4. If no move available: prints error, calls `end_game()`

**Called by:** `loop()`

---

### `ai_select_move()`

**Purpose:** Selects the best move using the neural network.

**What it does:**
1. Converts board to input vector: `board_to_input_vector()`
2. Runs neural network: `run_neural_network()`
3. Collects all empty positions:
   ```cpp
   for(int i = 0; i < BOARD_SIZE; i++) {
     if(board[i] == EMPTY) moves[count++] = i;
   }
   ```
4. Finds position with highest output score:
   ```cpp
   for(int i = 0; i < count; i++) {
     if(output_vector[moves[i]] > best_score) {
       best_score = output_vector[moves[i]];
       best = moves[i];
     }
   }
   ```

**Returns:** Best move (0-8) or -1 if no moves

**Called by:** `ai_turn_logic()`, `fast_train_session()`, `fast_train_self_play_session()`

---

### `end_game()`

**Purpose:** Handles game conclusion.

**What it does:**
1. Prints "Game Over!"
2. Announces result:
   - If `winner == PLAYER_X`: "Player wins!", `player_wins++`
   - If `winner == AI_O`: "AI wins!", `ai_wins++`
   - Otherwise: "Draw!"
3. If not in fast training mode:
   - Calls `update_weights_td_learning(winner == AI_O)`
4. Increments `games_played++`
5. Every 500 games:
   - Calls `save_weights_to_flash()`
   - Waits 1 second
6. Prints statistics
7. Waits 3 seconds
8. Calls `initialize_game(false)`

**Called by:** `ai_turn_logic()`, `fast_train_session()`

---

### `random_move()`

**Purpose:** Generates a random valid move (for training opponent).

**What it does:**
1. Tries 5 random positions:
   ```cpp
   for(int attempt = 0; attempt < 5; attempt++) {
     int pos = random(BOARD_SIZE);
     if(is_valid_move(pos)) return pos;
   }
   ```
2. If all taken, collects all empty positions:
   ```cpp
   for(int i = 0; i < BOARD_SIZE; i++) {
     if(is_valid_move(i)) moves[count++] = i;
   }
   ```
3. Returns random empty position or -1

**Called by:** `fast_train_session()`

---

## Training Functions

### `fast_train_session()`

**Purpose:** Trains AI by playing against random opponent.

**State variables (static):**
- `train_count`: Games trained in current session
- `ai_turn`: Whose turn it is
- `first`: First run flag

**What it does:**

1. **First Run:**
   - Calls `initialize_game(true)`
   - Resets counters
   - Prints "Training: Random(X) vs AI(O)"

2. **Game Over Handling:**
   - Updates statistics (random_wins, ai_wins)
   - Prints result
   - Calls `update_weights_td_learning()`
   - Increments counters
   - Saves weights every 500 games
   - Checks for stop request or target reached
   - If done: saves weights, exits training mode
   - Otherwise: starts new game

3. **Random's Turn:**
   - Calls `random_move()`
   - Calls `make_move()` for PLAYER_X
   - Checks if game over
   - Passes turn to AI

4. **AI's Turn:**
   - Calls `ai_select_move()`
   - Calls `record_ai_move()`
   - Calls `make_move()` for AI_O

**Delay:** 10ms between games (`FAST_TRAIN_DELAY`)

**Called by:** `loop()` when `fast_training_mode == true`

---

### `fast_train_self_play_session()`

**Purpose:** Trains AI by playing against itself (X vs O).

**State variables:**
- `selfplay_first_run`: First run flag
- `selfplay_train_count`: Games trained
- `selfplay_x_turn`: Whose turn (X goes first)

**What it does:**

1. **First Run:**
   - Calls `initialize_game(true)`
   - Calls `clear_selfplay_history()`
   - Resets counters
   - Prints "Self-Play Training: AI(X) vs AI(O)"

2. **Game Over Handling:**
   - Updates statistics (selfplay_wins_x, selfplay_wins_o, selfplay_draws)
   - Calls `update_selfplay_weights(winner)`
   - Increments counters
   - Saves weights every 500 games
   - Checks for stop request or target reached
   - If done: saves weights, exits training mode
   - Otherwise: starts new game, clears history

3. **Each Turn:**
   - Calls `ai_select_move()`
   - Calls `record_selfplay_move(move, current_player)`
   - Calls `make_move()`
   - Checks for win/draw
   - Toggles turn

**Delay:** 10ms between games

**Called by:** `loop()` when `selfplay_training_mode == true`

---

### `record_selfplay_move(int move, int player_sym)`

**Purpose:** Records a move in self-play training.

**What it does:**
1. Selects correct history buffer:
   ```cpp
   struct GameMove* history = (player_sym == PLAYER_X) ? selfplay_history_x : selfplay_history_o;
   ```
2. Finds empty slot in history
3. Converts board to input vector
4. Copies board state to history
5. Saves move position
6. Runs neural network
7. Saves output value for that move

**Called by:** `fast_train_self_play_session()`

---

### `clear_selfplay_history()`

**Purpose:** Resets both X and O history buffers.

**What it does:**
```cpp
for(int i = 0; i < MAX_GAME_MOVES; i++) {
  selfplay_history_x[i].move_made = -1;
  selfplay_history_o[i].move_made = -1;
}
```

**Called by:** `fast_train_self_play_session()`

---

### `update_selfplay_weights(int game_winner)`

**Purpose:** Updates weights from both X and O perspectives.

**What it does:**
1. Calculates X's reward:
   ```cpp
   float x_won = (game_winner == PLAYER_X) ? 1.0f : 
                 ((game_winner == -1) ? 0.25f : 0.0f);
   ```
2. Calculates O's reward:
   ```cpp
   float o_won = (game_winner == AI_O) ? 1.0f : 
                 ((game_winner == -1) ? 0.25f : 0.0f);
   ```
3. Calls `update_selfplay_network()` for each history

**Note:** Draw reward is 0.25 (not 0.5) to encourage winning

**Called by:** `fast_train_self_play_session()`

---

### `update_selfplay_network(struct GameMove* history, float final_reward)`

**Purpose:** Updates weights using TD learning on a single history.

**What it does:**
1. Counts moves in history
2. If no moves, returns early
3. For each move (backwards):
   ```cpp
   target = (i == num_moves - 1) ? final_reward : history[i + 1].move_value_before;
   td_error = target - history[i].move_value_before;
   update_network_for_move(history[i].board_state, history[i].move_made, td_error);
   ```

**Called by:** `update_selfplay_weights()`

---

## Web Server Functions

### `setup_wifi_ap()`

**Purpose:** Creates WiFi Access Point.

**What it does:**
1. Prints "Setting up AP..."
2. Calls `WiFi.softAP(AP_SSID, AP_PASSWORD)`
3. Prints configuration:
   - IP address
   - SSID: `TicTacToe_AI`
   - Password: `12345678`

**Called by:** `setup()`

---

### `setup_web_server()`

**Purpose:** Configures HTTP endpoints.

**What it does:**
```cpp
server.on("/", HTTP_GET, handle_web_root);
server.on("/m", HTTP_GET, handle_web_move);
server.on("/ng", HTTP_GET, handle_web_newgame);
server.on("/st", HTTP_GET, handle_web_status);
server.on("/sy", HTTP_GET, handle_web_choose_symbol);
server.begin();
```

**Endpoints:**
| Endpoint | Method | Handler |
|----------|--------|---------|
| `/` | GET | `handle_web_root` |
| `/m?p=N` | GET | `handle_web_move` |
| `/ng` | GET | `handle_web_newgame` |
| `/st` | GET | `handle_web_status` |
| `/sy?s=X\|O` | GET | `handle_web_choose_symbol` |

**Called by:** `setup()`

---

### `find_player_slot(uint32_t ip)`

**Purpose:** Manages player sessions.

**What it does:**
1. Searches for existing player with matching IP:
   ```cpp
   for(int i = 0; i < MAX_PLAYERS; i++) {
     if(players[i].active && players[i].ip == ip) return i;
   }
   ```
2. If not found, finds empty slot:
   ```cpp
   for(int i = 0; i < MAX_PLAYERS; i++) {
     if(!players[i].active) {
       // Initialize new player
       players[i].ip = ip;
       players[i].active = 1;
       players[i].player_sym = PLAYER_X;
       // Clear board, set timestamp
       return i;
     }
   }
   ```
3. Returns -1 if server full

**Returns:** Slot index (0-3) or -1

**Called by:** `handle_web_root()`, `handle_web_move()`, `handle_web_newgame()`, `handle_web_status()`, `handle_web_choose_symbol()`

---

### `update_player_timeout(uint32_t ip)`

**Purpose:** Refreshes player's last-seen timestamp.

**What it does:**
```cpp
for(int i = 0; i < MAX_PLAYERS; i++) {
  if(players[i].active && players[i].ip == ip) {
    players[i].last_seen = millis();
    break;
  }
}
```

**Called by:** All web handlers

---

### `handle_web_root(AsyncWebServerRequest *request)`

**Purpose:** Serves the main game HTML page.

**What it does:**
1. Gets client IP, finds/creates player slot
2. If server full: sends 503 error
3. Updates player timeout
4. Generates board HTML:
   - Iterates through player's board
   - For each cell:
     - Determines symbol (`-`, `X`, `O`)
     - Sets CSS class
     - Determines if clickable (player's turn)
     - Generates HTML div with onclick handler
5. Replaces placeholders in `HTML_PAGE` template:
   - `%BD%` → Board cells HTML
   - `%AW%`, `%PW%`, `%DR%`, `%RT%` → Statistics
   - `%RS%` → Status message
   - `%XS%`, `%OS%` → Symbol selection highlight
6. Sends HTML response

**Called by:** Web server when client visits `/`

---

### `handle_web_move(AsyncWebServerRequest *request)`

**Purpose:** Processes a player's move from web interface.

**What it does:**
1. Updates player timeout
2. Validates player is active and game in progress
3. Validates it's player's turn
4. Gets position from `?p=N` parameter
5. Validates move is legal (empty position)
6. Makes player's move:
   ```cpp
   players[slot].board[pos] = players[slot].player_sym;
   players[slot].player_turn = 0;
   ```
7. Checks for win/draw via `check_winner_player()`
8. If game continues:
   - AI makes counter-move via `ai_select_move_player()`
   - Checks for win/draw
   - If still continuing: sets `player_turn = 1`
9. Sends "OK" response

**Called by:** Web server when player clicks cell

---

### `handle_web_newgame(AsyncWebServerRequest *request)`

**Purpose:** Starts a new game for the requesting player.

**What it does:**
1. Updates player timeout
2. Finds/creates player slot
3. Clears player's board
4. Sets `active = 2` (playing)
5. Sets `player_turn` based on symbol (X goes first)
6. If player chose O:
   - AI (X) makes first move
   - Sets `player_turn = 1` (player's turn next)
7. Sends "OK" response

**Called by:** Web server when "New Game" button clicked

---

### `handle_web_status(AsyncWebServerRequest *request)`

**Purpose:** Returns game statistics as plain text.

**What it does:**
```cpp
snprintf(status, sizeof(status), 
  "Games:%d AI:%d Player:%d Draws:%d",
  games_played, ai_wins, player_wins, 
  games_played - ai_wins - player_wins);
```

**Response format:**
```
Games:100 AI:45 Player:40 Draws:15
```

**Called by:** Web server when "Status" button clicked

---

### `handle_web_choose_symbol(AsyncWebServerRequest *request)`

**Purpose:** Lets player choose X or O.

**What it does:**
1. Updates player timeout
2. Finds/creates player slot
3. Gets symbol from `?s=X|O` parameter
4. Sets player's symbol:
   - X: `player_sym = PLAYER_X`, `player_turn = 1`
   - O: `player_sym = AI_O`, `player_turn = 0`
5. Clears board
6. Sets `active = 1` (ready)
7. Sends "OK" response

**Called by:** Web server when symbol button clicked

---

### `check_winner_player(uint8_t* pboard)`

**Purpose:** Checks winner for a specific player's board (web games).

**What it does:**
- Same logic as `check_winner()` but takes board as parameter
- Checks 8 winning combinations
- Checks for draw (no empty positions)

**Returns:**
- `PLAYER_X` (1): Player won
- `AI_O` (2): AI won
- `-1`: Draw
- `0`: Game continues

**Called by:** `handle_web_move()`

---

### `ai_select_move_player(uint8_t* pboard, int ai_sym, int player_sym)`

**Purpose:** Selects AI move for web-based games.

**What it does:**
1. Converts player's board to input vector:
   ```cpp
   for(int i = 0; i < BOARD_SIZE; i++) {
     if(pboard[i] == 0) {
       input_vector[i*3] = 1.0f; input_vector[i*3+1] = 0.0f; input_vector[i*3+2] = 0.0f;
     } else if(pboard[i] == player_sym) {
       input_vector[i*3] = 0.0f; input_vector[i*3+1] = 1.0f; input_vector[i*3+2] = 0.0f;
     } else {
       input_vector[i*3] = 0.0f; input_vector[i*3+1] = 0.0f; input_vector[i*3+2] = 1.0f;
     }
   }
   ```
2. Runs neural network
3. Finds empty position with highest score
4. Returns best move

**Called by:** `handle_web_move()`, `handle_web_newgame()`

---

## Serial Command Functions

### `process_serial_command()`

**Purpose:** Parses and executes serial commands.

**What it does:**
1. Reads line from Serial, trims whitespace
2. Parses command:

| Command | Action |
|---------|--------|
| `train [N]` | Sets `fast_training_mode = true`, `fast_train_target = N` |
| `trainself [N]` | Sets `selfplay_training_mode = true`, `fast_train_target = N` |
| `embed` | Calls `export_weights_as_header()` |
| `newgame` | Calls `initialize_game(false)` |
| `status` | Prints game statistics |
| `stop` | Calls `stop_training()` |
| `help` | Calls `print_help()` |
| `0-8` | Makes move if valid and player's turn |

3. Invalid commands: prints error

**Called by:** `loop()` when serial data available

---

### `print_help()`

**Purpose:** Displays available commands.

**Output:**
```
=== Commands ===
train [N]     - Train N games AI vs Random (default 45)
trainself [N] - Train N games AI vs AI self-play (default 45)
embed         - Export weights as C header (ai_weights.h)
newgame       - Start new game
status        - Show stats
stop          - Stop training
help          - This message
0-8           - Make move
```

**Called by:** `setup()`, `process_serial_command()`

---

### `stop_training()`

**Purpose:** Requests training to stop gracefully.

**What it does:**
```cpp
if(fast_training_mode || selfplay_training_mode) {
  stop_training_requested = true;
  Serial.println("Stopping training...");
}
```

**Called by:** `process_serial_command()`

---

## Weight Export/Import Functions

### `export_weights()`

**Purpose:** Outputs weights as comma-separated values.

**What it does:**
1. Prints header: `=== EXPORT ===`
2. Prints `AIWEIGHTS:` prefix
3. Outputs all weights in order:
   - Input weights (1,458 values)
   - Hidden weights (486 values)
   - Input bias (54 values)
   - Hidden bias (9 values)
4. Prints footer: `=== END ===`

**Format:**
```
=== EXPORT ===
AIWEIGHTS:0.123456,-0.234567,0.345678,...
=== END ===
```

**Called by:** `process_serial_command()` (not directly exposed)

---

### `export_weights_as_header()`

**Purpose:** Outputs weights as C array for `ai_weights.h`.

**What it does:**
1. Prints header comments:
   ```cpp
   // ============================================
   // ai_weights.h - Copy this to ai_weights.h file
   // ============================================
   // Generated: 12345ms after boot
   // Games played: 1000
   // AI Wins: 450
   ```
2. Iterates through all 2,007 weights:
   - Calculates which array each index belongs to
   - Prints value with 6 decimal places
   - Newline every 10 values
3. Prints footer with total count

**Output format:**
```cpp
0.027885,-0.094998,-0.044994,-0.055358,0.047294,0.035340,0.078436,-0.082612,-0.015616,-0.094041,
-0.056272,0.001071,-0.094693,...
// (10 values per line)

// ============================================
// End of weights (2007 values)
// ============================================
```

**Called by:** `process_serial_command()` via `embed` command

---

### `import_weights(const char* data)`

**Purpose:** Loads weights from AIWEIGHTS: format string.

**What it does:**
1. Validates `AIWEIGHTS:` prefix
2. Strips prefix
3. Parses comma-separated values:
   - Accumulates characters into buffer
   - On comma/newline: converts buffer to float
   - Distributes to correct array based on count
4. Validates total count (2,007 values)
5. Prints result

**Returns:** `true` if all 2,007 values loaded

**Called by:** (not directly exposed, used for paste import)

---

### `save_weights_to_file()`

**Purpose:** Saves weights to `/ai_weights.txt` on filesystem.

**What it does:**
1. Opens `/ai_weights.txt` for writing
2. Writes `AIWEIGHTS:` header
3. Writes all weights comma-separated
4. Closes file
5. Prints confirmation

**Called by:** (not directly exposed)

---

### `load_weights_from_file()`

**Purpose:** Loads weights from `/ai_weights.txt`.

**What it does:**
1. Opens `/ai_weights.txt` for reading
2. Parses comma-separated values
3. Distributes to weight arrays
4. Reports how many values loaded
5. Closes file

**Called by:** (not directly exposed)

---

## Function Call Flow Diagrams

### Normal Game Flow
```
setup()
  │
  ├─→ setup_wifi_ap()
  ├─→ allocate_psram_memory()
  ├─→ load_weights_from_flash() / load_embedded_weights() / initialize_neural_network()
  ├─→ setup_web_server()
  └─→ initialize_game(false)
        │
        └─→ print_board()
              │
              v
loop() ───────┘
  │
  ├─→ Check player timeouts
  ├─→ process_serial_command() ──→ make_move() ──→ is_game_over() ──→ end_game()
  │         │                                      │
  │         └─→ (during AI turn)                   └─→ update_weights_td_learning()
  │                   │                                    │
  │                   v                                    └─→ save_weights_to_flash()
  └─→ ai_turn_logic()                                      │
            │                                              └─→ initialize_game()
            ├─→ ai_select_move()
            │       │
            │       ├─→ board_to_input_vector()
            │       ├─→ run_neural_network()
            │       └─→ (return best move)
            │
            ├─→ record_ai_move()
            │       │
            │       ├─→ board_to_input_vector()
            │       ├─→ run_neural_network()
            │       └─→ (save to game_history)
            │
            ├─→ make_move()
            ├─→ print_board()
            └─→ is_game_over() ──→ end_game()
```

### Training Flow (AI vs Random)
```
loop()
  │
  └─→ fast_training_mode == true
        │
        v
  fast_train_session()
    │
    ├─→ (first run) initialize_game(true)
    │
    ├─→ (game over)
    │     ├─→ update_weights_td_learning()
    │     │       │
    │     │       └─→ update_network_for_move() ──→ run_neural_network()
    │     │
    │     ├─→ save_weights_to_flash() (every 500 games)
    │     └─→ initialize_game(true)
    │
    ├─→ (random's turn)
    │     ├─→ random_move()
    │     └─→ make_move(PLAYER_X)
    │
    └─→ (AI's turn)
          ├─→ ai_select_move() ──→ run_neural_network()
          ├─→ record_ai_move() ──→ run_neural_network()
          └─→ make_move(AI_O)
```

### Self-Play Training Flow
```
loop()
  │
  └─→ selfplay_training_mode == true
        │
        v
  fast_train_self_play_session()
    │
    ├─→ (first run)
    │     ├─→ initialize_game(true)
    │     └─→ clear_selfplay_history()
    │
    ├─→ (game over)
    │     ├─→ update_selfplay_weights(winner)
    │     │     ├─→ update_selfplay_network(selfplay_history_x, x_reward)
    │     │     │     └─→ update_network_for_move() (for each move)
    │     │     │
    │     │     └─→ update_selfplay_network(selfplay_history_o, o_reward)
    │     │           └─→ update_network_for_move() (for each move)
    │     │
    │     ├─→ save_weights_to_flash() (every 500 games)
    │     └─→ initialize_game(true)
    │
    └─→ (each turn)
          ├─→ ai_select_move() ──→ run_neural_network()
          ├─→ record_selfplay_move(move, player_sym)
          │     ├─→ board_to_input_vector()
          │     ├─→ run_neural_network()
          │     └─→ (save to X or O history)
          │
          └─→ make_move(current_player)
```

### Web Game Flow
```
Client connects to 192.168.4.1
  │
  v
handle_web_root(request)
  │
  ├─→ find_player_slot(ip)
  │     ├─→ (existing) return slot
  │     └─→ (new) initialize player slot
  │
  ├─→ Generate board HTML
  ├─→ Replace template placeholders
  └─→ Send HTML page
        │
        v
  Client clicks symbol (X or O)
        │
        v
  handle_web_choose_symbol(request)
    │
    ├─→ Set player_sym
    ├─→ Clear board
    └─→ Send OK
        │
        v
  Client clicks "New Game"
        │
        v
  handle_web_newgame(request)
    │
    ├─→ Clear board
    ├─→ Set active = 2 (playing)
    ├─→ (if O) AI makes first move
    └─→ Send OK
        │
        v
  Client clicks cell
        │
        v
  handle_web_move(request)
    │
    ├─→ Validate turn
    ├─→ Make player move
    ├─→ check_winner_player()
    ├─→ (if game continues)
    │     └─→ ai_select_move_player() ──→ run_neural_network()
    │     └─→ Make AI move
    │     └─→ check_winner_player()
    │
    └─→ Send OK
          │
          v
  Client reloads page ──→ handle_web_root()
```

---

## Quick Reference

### Neural Network Architecture
```
Input (27) → Hidden (54) → Output (9)
   │            │            │
   │            │            └─→ Move scores
   │            └─→ Sigmoid activation
   └─→ One-hot encoding per position
```

### Weight Arrays
| Array | Dimensions | Size (floats) | Size (bytes) |
|-------|------------|---------------|--------------|
| `input_weights` | 27 × 54 | 1,458 | 5,832 |
| `hidden_weights` | 54 × 9 | 486 | 1,944 |
| `input_bias` | 54 | 54 | 216 |
| `hidden_bias` | 9 | 9 | 36 |
| **Total** | | **2,007** | **8,028** |

### Game States
| State | Value |
|-------|-------|
| `EMPTY` | 0 |
| `PLAYER_X` | 1 |
| `AI_O` | 2 |

### Player Active States
| State | Value | Description |
|-------|-------|-------------|
| Inactive | 0 | Slot available |
| Ready | 1 | Symbol chosen, waiting for game |
| Playing | 2 | Game in progress |

### Key Constants
| Constant | Value | Description |
|----------|-------|-------------|
| `SAVE_WEIGHTS_INTERVAL` | 500 | Games between saves |
| `LEARNING_RATE` | 0.01 | TD learning rate |
| `MAX_PLAYERS` | 4 | Concurrent players |
| `PLAYER_TIMEOUT_MS` | 60000 | Idle timeout (60s) |
| `FAST_TRAIN_GAMES` | 45 | Default training games |
| `FAST_TRAIN_DELAY` | 10 | ms between training games |
| `MAX_GAME_MOVES` | 20 | History buffer size |
