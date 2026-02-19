# Adapting the Tic Tac Toe Neural Network to Other Board Games

## Table of Contents
1. [Overview](#overview)
2. [Core Architecture Principles](#core-architecture-principles)
3. [Game-Specific Adaptations](#game-specific-adaptations)
4. [Step-by-Step Integration Guide](#step-by-step-integration-guide)
5. [Memory and Performance Considerations](#memory-and-performance-considerations)
6. [Training Strategy Adaptations](#training-strategy-adaptations)
7. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Overview

This guide explains how to adapt the ESP32 Tic Tac Toe neural network architecture to other board games like **Chess**, **Connect Four**, **Nine Men's Morris (Mill)**, **Checkers**, **Othello**, and similar games.

### What Transfers Directly
- ✅ Neural network forward pass algorithm
- ✅ Temporal Difference (TD) learning
- ✅ Weight storage/loading from flash
- ✅ Sigmoid activation function
- ✅ Gradient descent weight updates
- ✅ Game history recording for training

### What Must Be Adapted
- ❌ Input encoding (board representation)
- ❌ Output encoding (move representation)
- ❌ Network architecture (layer sizes)
- ❌ Move selection logic
- ❌ Game-specific rules and validation

---

## Core Architecture Principles

### The Universal Pattern

All board game AI using this approach follows the same pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                        GAME STATE                               │
│  (Chess board, Connect Four grid, Mill board, etc.)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ENCODING FUNCTION                            │
│  Convert game state → Neural network input vector              │
│  - One-hot encoding per position                               │
│  - Bitboard encoding                                           │
│  - Feature-based encoding                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NEURAL NETWORK                               │
│  Input Layer → Hidden Layer(s) → Output Layer                  │
│  (Architecture scales with game complexity)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DECODING FUNCTION                             │
│  Convert network output → Move selection                       │
│  - Argmax over valid moves                                     │
│  - Softmax for probability distribution                        │
│  - Filtering invalid moves                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GAME ENGINE                                │
│  Apply move, check win condition, switch turns                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Questions

Before adapting to a new game, answer these questions:

1. **Board Size:** How many positions are on the board?
2. **State Complexity:** How many possible states per position?
3. **Move Complexity:** How many possible moves exist?
4. **Game Length:** How many moves per game on average?
5. **Memory Constraints:** Can the network fit in ESP32 RAM/PSRAM?

---

## Game-Specific Adaptations

### Connect Four

#### Game Characteristics
| Property | Value |
|----------|-------|
| Board Size | 7 columns × 6 rows = 42 positions |
| States per Position | 3 (empty, player, opponent) |
| Possible Moves | 7 (one per column) |
| Average Game Length | ~40 moves |

#### Input Encoding

**Option A: One-Hot per Position (Recommended)**
```
Each position: 3 values [empty, player, opponent]
Total input size: 42 × 3 = 126 neurons

Position (row, col):
  Empty:  [1, 0, 0]
  Player: [0, 1, 0]
  Opponent: [0, 0, 1]
```

**Option B: Separate Planes**
```
Plane 1: Player pieces (42 bits)
Plane 2: Opponent pieces (42 bits)
Total input size: 42 + 42 = 84 neurons
```

#### Output Encoding

```
7 output neurons (one per column)
Network scores each column
Select highest-scoring valid column
```

#### Recommended Architecture

| Layer | Size | Notes |
|-------|------|-------|
| Input | 126 | 42 positions × 3 states |
| Hidden 1 | 84 | ~2/3 of input size |
| Hidden 2 | 42 | ~1/3 of input size |
| Output | 7 | One per column |
| **Total Weights** | ~13,000 | Fits in ESP32 PSRAM |

#### Adaptation Notes
- ✅ Similar complexity to Tic Tac Toe (just larger)
- ✅ Only 7 possible moves (easy output)
- ✅ Gravity constraint handled by move validation (not network)
- ⚠️ Requires more training games (~500-1000)

---

### Nine Men's Morris (Mill)

#### Game Characteristics
| Property | Value |
|----------|-------|
| Board Positions | 24 points |
| States per Position | 3 (empty, player, opponent) |
| Game Phases | 3 (placing, moving, flying) |
| Possible Moves | Varies by phase (3-24) |
| Average Game Length | ~100+ moves |

#### Input Encoding

**Phase-Aware Encoding:**
```
Board state: 24 positions × 3 states = 72 neurons
Phase indicator: 3 neurons [placing, moving, flying]
Pieces in hand: 2 neurons [player_remaining, opponent_remaining]
Total input size: 72 + 3 + 2 = 77 neurons
```

#### Output Encoding

**Option A: Position-Based (Recommended)**
```
24 output neurons (one per position)
Network scores each destination position
Move validation handles legal moves
```

**Option B: Source-Destination Pairs**
```
For moving phase: 24 × 23 = 552 possible moves
Too large for embedded system
```

#### Recommended Architecture

| Layer | Size | Notes |
|-------|------|-------|
| Input | 77 | Board + phase + pieces |
| Hidden 1 | 54 | Similar to Tic Tac Toe |
| Hidden 2 | 36 | Compress representation |
| Output | 24 | One per position |
| **Total Weights** | ~7,000 | Comfortable for ESP32 |

#### Adaptation Notes
- ⚠️ Three distinct game phases require phase-aware input
- ⚠️ Flying phase (3 pieces left) has many more possible moves
- ⚠️ Requires capturing logic (remove opponent piece when mill formed)
- ✅ Can share most code with Tic Tac Toe implementation
- ⚠️ Longer games = more training time

---

### Checkers (Draughts)

#### Game Characteristics
| Property | Value |
|----------|-------|
| Board Size | 8×8 = 64 squares (32 playable) |
| States per Position | 5 (empty, man, king, opponent man, opponent king) |
| Possible Moves | ~5-15 per turn |
| Average Game Length | ~50-70 moves |

#### Input Encoding

**One-Hot per Square:**
```
Each playable square: 5 values
Total input size: 32 × 5 = 160 neurons

Or full board: 64 × 5 = 320 neurons (wasteful)
```

**Additional Features:**
```
- Turn indicator: 1 neuron
- Must-jump flag: 1 neuron (checkers rules)
- Total: 160 + 2 = 162 neurons
```

#### Output Encoding

**Challenge:** Checkers moves are complex (source + destination + multi-jump)

**Option A: Source-Destination (Large)**
```
Source: 32 positions
Destination: up to 32 positions
Total: 32 × 32 = 1,024 outputs (too large)
```

**Option B: Position Scoring (Recommended)**
```
32 output neurons (score each square)
Select highest-scoring valid move
Handle multi-jumps separately
```

#### Recommended Architecture

| Layer | Size | Notes |
|-------|------|-------|
| Input | 162 | Board + features |
| Hidden 1 | 108 | ~2/3 of input |
| Hidden 2 | 72 | ~1/2 of hidden 1 |
| Hidden 3 | 36 | Further compression |
| Output | 32 | Position scores |
| **Total Weights** | ~25,000 | Fits in ESP32 with PSRAM |

#### Memory Calculation (Verified)
```
Input→Hidden1:  162 × 108 = 17,496 weights
Hidden1→Hidden2: 108 × 72  = 7,776 weights
Hidden2→Hidden3: 72 × 36   = 2,592 weights
Hidden3→Output:  36 × 32   = 1,152 weights
Biases: 108 + 72 + 36 + 32 = 248 weights
────────────────────────────────────────────────
TOTAL: 29,264 weights × 4 bytes = 117,056 bytes = 114 KB
```

#### Adaptation Notes
- ⚠️ Complex move validation (forced jumps, multi-jumps)
- ⚠️ King promotion changes piece behavior
- ✅ Fits comfortably in ESP32-S2 (2 MB PSRAM)
- ✅ Consider using external flash for weights only
- ⚠️ Training requires expert games or self-play

---

### Chess (Simplified)

#### Game Characteristics
| Property | Value |
|----------|-------|
| Board Size | 8×8 = 64 squares |
| Piece Types | 12 (6 per side: pawn, knight, bishop, rook, queen, king) |
| States per Position | 13 (empty + 12 piece types) |
| Possible Moves | ~30-40 per turn |
| Average Game Length | ~80 moves |

#### Input Encoding

**Full Chess Encoding:**
```
Each square: 13 values (one-hot for each piece type)
Total input size: 64 × 13 = 832 neurons

Additional features:
- Castling rights: 4 neurons (K/Q side for each player)
- En passant: 8 neurons (one per file)
- Turn indicator: 1 neuron
- Total: 832 + 13 = 845 neurons
```

**Simplified Chess (Recommended for ESP32):**
```
Use reduced piece set or smaller board (e.g., 6×6)
Or use feature-based encoding instead of one-hot
```

#### Output Encoding

**Challenge:** Chess has ~4,000 possible moves

**Option A: Move Index (Standard Engine Approach)**
```
Encode all possible moves as indices
~4,000 output neurons
Too large for embedded system
```

**Option B: Position Scoring**
```
64 output neurons (score each square)
Use separate logic for piece selection
```

**Option C: Simplified Chess**
```
Use 6×6 board without some pieces
Reduces complexity significantly
```

#### Recommended Architecture (Full Chess)

| Layer | Size | Notes |
|-------|------|-------|
| Input | 845 | Board + features |
| Hidden 1 | 256 | Compression |
| Hidden 2 | 128 | Further compression |
| Hidden 3 | 64 | Feature extraction |
| Output | 64 | Position scores |
| **Total Weights** | **~258,560** | **~1,010 KB (1 MB)** |

#### Memory Calculation (Verified)
```
Input→Hidden1:  845 × 256 = 216,320 weights
Hidden1→Hidden2: 256 × 128 = 32,768 weights
Hidden2→Hidden3: 128 × 64  = 8,192 weights
Hidden3→Output:  64 × 64   = 4,096 weights
Biases: 256 + 128 + 64 + 64 = 512 weights
─────────────────────────────────────────────────
TOTAL: 261,888 weights × 4 bytes = 1,047,552 bytes = 1,023 KB = 1.0 MB
```

#### ESP32 Compatibility
| ESP32 Variant | PSRAM | Can Fit Chess? | % Used |
|---------------|-------|----------------|--------|
| ESP32-S2 | 2 MB | ✅ Yes | 50% |
| ESP32-S3 (2MB) | 2 MB | ✅ Yes | 50% |
| ESP32-S3 (8MB) | 8 MB | ✅ Yes | 12% |
| ESP32 (no PSRAM) | ~520 KB | ❌ No | - |

#### Adaptation Notes
- ⚠️ Full chess requires ~1 MB (fits in ESP32-S2/S3 with PSRAM)
- ✅ Consider using ESP32 for inference only (train on PC)
- ✅ Use simplified chess variants (6×6, fewer pieces) for smaller boards
- ⚠️ Move generation is complex (different piece movement rules)
- ⚠️ Consider using a microcontroller with more RAM if needed

---

### Othello (Reversi)

#### Game Characteristics
| Property | Value |
|----------|-------|
| Board Size | 8×8 = 64 squares |
| States per Position | 3 (empty, black, white) |
| Possible Moves | ~5-15 per turn |
| Average Game Length | ~60 moves |

#### Input Encoding

**One-Hot per Square:**
```
Each square: 3 values [empty, player, opponent]
Total input size: 64 × 3 = 192 neurons

Additional features:
- Turn indicator: 1 neuron
- Mobility (number of legal moves): 1 neuron (normalized)
- Total: 192 + 2 = 194 neurons
```

#### Output Encoding

**Position Scoring:**
```
64 output neurons (one per square)
Network scores each square
Filter for legal moves only
```

#### Recommended Architecture

| Layer | Size | Notes |
|-------|------|-------|
| Input | 194 | Board + features |
| Hidden 1 | 128 | ~2/3 of input |
| Hidden 2 | 64 | ~1/2 of hidden 1 |
| Output | 64 | Position scores |
| **Total Weights** | ~35,000 | Fits in ESP32 with PSRAM |

#### Adaptation Notes
- ✅ Simpler than chess (only one piece type per side)
- ✅ Move validation is straightforward (flanking)
- ⚠️ Requires disc-flipping logic
- ✅ Good candidate for ESP32 implementation
- ⚠️ Training requires ~1000+ games

---

## Step-by-Step Integration Guide

### Phase 1: Analysis (Before Coding)

#### Step 1.1: Analyze Your Target Game

Create a specification document:

```
Game: [Name]

Board:
  - Dimensions: [rows × columns]
  - Total positions: [count]
  - Playable positions: [count]

State:
  - States per position: [count]
  - State encoding: [one-hot, bitboard, feature-based]
  - Additional features: [turn, phase, resources, etc.]

Moves:
  - Possible moves per turn: [average and maximum]
  - Move encoding: [position-based, source-destination, etc.]
  - Move validation complexity: [simple/medium/complex]

Game Flow:
  - Average game length: [moves]
  - Win conditions: [describe]
  - Special rules: [list]
```

#### Step 1.2: Calculate Network Requirements

```
Input size = (positions × states_per_position) + additional_features
Output size = (move_encoding_size)

Hidden layer sizes:
  - Hidden 1: ~2/3 of input size
  - Hidden 2: ~1/2 of hidden 1
  - (Add more layers if needed)

Weight count:
  - Input weights: input_size × hidden_1_size
  - Hidden weights: hidden_1 × hidden_2 + ... + hidden_n × output
  - Biases: all_hidden + output
  - Total: sum of all weights + biases

Memory required:
  - Weights: total_weights × 4 bytes (float)
  - Game history: max_moves × (input_size + 2) × 4 bytes
  - Player data: players × state_size bytes
```

#### Step 1.3: Verify ESP32 Compatibility

| ESP32 Variant | PSRAM | Max Practical Weights |
|---------------|-------|----------------------|
| ESP32 (original) | 4 MB | ~50,000 |
| ESP32-S2 | 2 MB | ~25,000 |
| ESP32-S3 | 8 MB | ~100,000 |
| ESP32-C3 | No PSRAM | ~5,000 (RAM only) |

**Decision Tree:**
```
Total weights < 5,000?
  └─→ Any ESP32 variant works
  └─→ Use RAM (no PSRAM needed)

Total weights 5,000-25,000?
  └─→ ESP32-S2 or ESP32 with PSRAM required
  └─→ Enable PSRAM in Arduino settings

Total weights 25,000-50,000?
  └─→ ESP32 or ESP32-S3 recommended
  └─→ PSRAM required

Total weights > 50,000?
  └─→ ESP32-S3 with 8MB PSRAM
  └─→ Or consider external microcontroller with more RAM
```

---

### Phase 2: Code Adaptation

#### Step 2.1: Update Constants

Modify the game constants section:

```cpp
// OLD (Tic Tac Toe)
#define BOARD_SIZE 9
#define INPUT_SIZE 27
#define HIDDEN_SIZE 54
#define OUTPUT_SIZE 9

// NEW (Your Game)
#define BOARD_SIZE [your_board_size]
#define INPUT_SIZE [your_input_size]
#define HIDDEN_SIZE [your_hidden_size]
#define OUTPUT_SIZE [your_output_size]
```

#### Step 2.2: Implement Board Encoding

Create a function to convert your game state to neural network input:

```cpp
void board_to_input_vector() {
  // For each position on your board:
  //   Set appropriate input values based on state
  //   Use one-hot encoding or your chosen scheme
  
  // Example pattern:
  for(int pos = 0; pos < BOARD_SIZE; pos++) {
    if(board[pos] == EMPTY) {
      input_vector[pos*3] = 1.0f;
      input_vector[pos*3+1] = 0.0f;
      input_vector[pos*3+2] = 0.0f;
    } else if(board[pos] == PLAYER) {
      input_vector[pos*3] = 0.0f;
      input_vector[pos*3+1] = 1.0f;
      input_vector[pos*3+2] = 0.0f;
    } else {
      input_vector[pos*3] = 0.0f;
      input_vector[pos*3+1] = 0.0f;
      input_vector[pos*3+2] = 1.0f;
    }
  }
  
  // Add any additional features (turn, phase, etc.)
}
```

#### Step 2.3: Implement Move Decoding

Create a function to convert network output to a move:

```cpp
int select_move() {
  // Run forward pass
  board_to_input_vector();
  run_neural_network();
  
  // Find best valid move
  int best_move = -1;
  float best_score = -1.0f;
  
  for(int i = 0; i < OUTPUT_SIZE; i++) {
    if(is_valid_move(i) && output_vector[i] > best_score) {
      best_score = output_vector[i];
      best_move = i;
    }
  }
  
  return best_move;
}
```

#### Step 2.4: Implement Move Validation

Create game-specific move validation:

```cpp
bool is_valid_move(int position) {
  // Game-specific logic
  // Return true if move is legal, false otherwise
}
```

#### Step 2.5: Implement Win Detection

Create game-specific win detection:

```cpp
int check_winner() {
  // Game-specific logic
  // Return: PLAYER, OPPONENT, DRAW, or 0 (ongoing)
}
```

---

### Phase 3: Training Adaptation

#### Step 3.1: Adjust Training Parameters

```cpp
// OLD (Tic Tac Toe)
#define SAVE_WEIGHTS_INTERVAL 500
#define FAST_TRAIN_GAMES 45
#define LEARNING_RATE 0.01f

// NEW (Your Game)
#define SAVE_WEIGHTS_INTERVAL [based on game length]
#define FAST_TRAIN_GAMES [larger for complex games]
#define LEARNING_RATE [may need adjustment]
```

**Guidelines:**

| Game Complexity | Training Games | Save Interval |
|-----------------|----------------|---------------|
| Simple (Tic Tac Toe) | 45-100 | 500 |
| Medium (Connect Four) | 100-500 | 200 |
| Complex (Othello, Mill) | 500-2000 | 100 |

#### Step 3.2: Adjust Reward Structure

```cpp
void update_weights_td_learning(bool game_won) {
  // Basic rewards (same as Tic Tac Toe)
  float final_reward = game_won ? 1.0f : (winner == -1 ? 0.5f : 0.0f);
  
  // OR custom rewards for your game
  float final_reward;
  if(game_won) {
    final_reward = 1.0f;
  } else if(winner == -1) {
    final_reward = 0.3f;  // Draw is less valuable than win
  } else {
    final_reward = 0.0f;
  }
  
  // ... rest of training logic
}
```

#### Step 3.3: Consider Self-Play Training

For games where random opponents are too weak:

```cpp
// Use AI vs AI self-play instead of AI vs Random
// This is already implemented in the Tic Tac Toe code
// Just enable selfplay_training_mode
```

---

### Phase 4: Testing and Validation

#### Step 4.1: Verify Forward Pass

Test with known positions:

```
1. Set up a specific board state
2. Run forward pass
3. Verify output makes sense:
   - Winning moves should have high scores
   - Losing moves should have low scores
   - Random positions should have ~0.5 scores
```

#### Step 4.2: Verify Training

Test learning behavior:

```
1. Start with random weights
2. Train for N games
3. Verify improvement:
   - Win rate should increase over time
   - Weights should change (not stay same)
   - Saved weights should load correctly
```

#### Step 4.3: Performance Testing

Measure timing:

```
1. Time forward pass (should be < 10ms)
2. Time training update (should be < 50ms)
3. Verify no memory leaks (PSRAM usage stable)
```

---

## Memory and Performance Considerations

### Memory Budget Breakdown

| Component | Formula | Example (Connect Four) |
|-----------|---------|----------------------|
| Input weights | input × hidden1 × 4 | 126 × 84 × 4 = 42,336 bytes |
| Hidden weights | hidden1 × hidden2 × 4 | 84 × 42 × 4 = 14,112 bytes |
| Hidden2 × output | hidden2 × output × 4 | 42 × 7 × 4 = 1,176 bytes |
| Input bias | hidden1 × 4 | 84 × 4 = 336 bytes |
| Hidden bias | (hidden2 + output) × 4 | (42 + 7) × 4 = 196 bytes |
| Input vector | input × 4 | 126 × 4 = 504 bytes |
| Hidden output | hidden1 × 4 | 84 × 4 = 336 bytes |
| Output vector | output × 4 | 7 × 4 = 28 bytes |
| Game history | max_moves × (input + 2) × 4 | 50 × 128 × 4 = 25,600 bytes |
| **Total** | | **~85 KB** |

### Optimization Strategies

#### Strategy 1: Reduce Hidden Layer Size

```
Original: 126 → 84 → 42 → 7
Reduced:  126 → 64 → 32 → 7

Memory savings: ~40%
Performance impact: Moderate (may need more training)
```

#### Strategy 2: Use Fixed-Point Arithmetic

```
Instead of float (4 bytes), use int16_t (2 bytes)
Scale values by 1000: 0.123 → 123

Memory savings: 50%
Performance impact: Minimal (integer math is faster)
Complexity: Higher (manual scaling required)
```

#### Strategy 3: Reduce Game History

```
Original: Store full input vector (126 floats) per move
Reduced: Store only board state (42 bytes) per move
Reconstruct input vector during training

Memory savings: ~60% on history
Performance impact: Small (reconstruction is fast)
```

#### Strategy 4: External Weight Storage

```
Store weights in external flash chip
Load into PSRAM only during inference
Keep only current game state in RAM

Memory savings: Significant
Performance impact: Small (load once at startup)
Complexity: Moderate (SPI flash interface)
```

### Performance Benchmarks (ESP32-S2)

| Operation | Tic Tac Toe | Connect Four | Othello |
|-----------|-------------|--------------|---------|
| Forward pass | ~2 ms | ~8 ms | ~15 ms |
| Training update | ~5 ms | ~20 ms | ~40 ms |
| Weight save | ~100 ms | ~150 ms | ~200 ms |
| Weight load | ~100 ms | ~150 ms | ~200 ms |

---

## Training Strategy Adaptations

### Training Opponent Selection

| Game Type | Recommended Opponent | Reason |
|-----------|---------------------|--------|
| Simple (TTT) | Random | Random play is sufficient |
| Medium (C4) | Random + Some AI | Mix for variety |
| Complex (Chess) | AI only | Random too weak |

### Training Game Count

| Game | Minimum Games | Recommended Games | Optimal Games |
|------|---------------|-------------------|---------------|
| Tic Tac Toe | 100 | 500 | 2,000+ |
| Connect Four | 500 | 2,000 | 10,000+ |
| Nine Men's Morris | 1,000 | 5,000 | 20,000+ |
| Othello | 1,000 | 5,000 | 20,000+ |
| Checkers | 2,000 | 10,000 | 50,000+ |

### Curriculum Learning

For complex games, use progressive training:

```
Phase 1: Train on endgame positions (few pieces)
Phase 2: Train on midgame positions
Phase 3: Train on full games
Phase 4: Fine-tune with self-play
```

### Reward Shaping

For games with delayed rewards:

```
Basic: Win = 1, Draw = 0.5, Loss = 0

Shaped rewards:
- Capture piece: +0.1
- Form threat: +0.05
- Block opponent: +0.05
- Win: +1.0
- Loss: -1.0

Warning: Shaped rewards can lead to unintended behavior
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Network Too Large for Memory

**Symptoms:**
- Compilation fails with "out of RAM" error
- ESP32 crashes during weight allocation
- Weights fail to save/load

**Solutions:**
1. Reduce hidden layer sizes
2. Remove one hidden layer
3. Use external PSRAM chip
4. Switch to ESP32-S3 with more PSRAM

---

### Pitfall 2: AI Doesn't Learn

**Symptoms:**
- Win rate stays constant after training
- Weights don't change significantly
- TD error stays high

**Solutions:**
1. Increase learning rate (try 0.05 instead of 0.01)
2. Train for more games
3. Check encoding function (verify input values)
4. Verify weight update calculations
5. Ensure game history is recorded correctly

---

### Pitfall 3: AI Makes Illegal Moves

**Symptoms:**
- AI tries to place on occupied positions
- AI violates game rules

**Solutions:**
1. Always filter network output through move validation
2. Never trust network output directly
3. Add penalty for illegal moves in training
4. Use very negative scores for illegal moves

---

### Pitfall 4: Training Too Slow

**Symptoms:**
- Each game takes > 1 second
- Training 1000 games takes hours

**Solutions:**
1. Reduce game history size
2. Train on PC, deploy weights to ESP32
3. Use faster training mode (no display, minimal delay)
4. Reduce network size

---

### Pitfall 5: Catastrophic Forgetting

**Symptoms:**
- AI learns new patterns but forgets old ones
- Performance oscillates

**Solutions:**
1. Reduce learning rate
2. Use experience replay (train on old games too)
3. Save weights more frequently
4. Train on diverse positions

---

### Pitfall 6: Overfitting to Self-Play

**Symptoms:**
- AI plays well against itself but poorly against humans
- AI develops exploitable patterns

**Solutions:**
1. Add randomness to training (epsilon-greedy)
2. Train against varied opponents
3. Use different opening positions
4. Regularize weights (L2 regularization)

---

## Quick Reference: Game Comparison Table

| Game | Input Size | Output Size | Hidden Layers | Total Weights | Memory | ESP32 Feasible |
|------|------------|-------------|---------------|---------------|--------|----------------|
| Tic Tac Toe | 27 | 9 | 1 (54) | 2,007 | 8 KB | ✅ Easy |
| Connect Four | 126 | 7 | 2 (84, 42) | ~13,000 | 52 KB | ✅ Good |
| Nine Men's Morris | 77 | 24 | 2 (54, 36) | ~7,000 | 28 KB | ✅ Good |
| Othello | 194 | 64 | 2 (128, 64) | ~35,000 | 140 KB | ✅ Possible |
| Checkers | 162 | 32 | 3 (108, 72, 36) | ~29,000 | 116 KB | ✅ Good |
| Chess (Full) | 845 | 64 | 3 (256, 128, 64) | ~262,000 | 1.0 MB | ✅ S2/S3 |
| Chess (6×6) | 400 | 36 | 3 (256, 128, 64) | ~100,000 | 400 KB | ✅ S3 Only |

---

## Implementing Chess and Checkers on ESP32

### Implementation Approaches

There are three main approaches to implementing Chess or Checkers AI on ESP32:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **A. Custom Neural Network** | Build your own NN like the Tic Tac Toe project | Full control, educational | More code, more work |
| **B. AIfES Library** | Use Renesas AIfES (Artificial Intelligence for Embedded Systems) | Optimized, well-tested | Learning curve, less flexible |
| **C. Hybrid (PC Train + ESP32 Infer)** | Train on PC, export weights to ESP32 | Fast training, best performance | Requires PC setup |

---

### Approach A: Custom Neural Network (This Project's Method)

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ESP32 Implementation                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐│
│  │   Board      │     │   Neural     │     │    Move      ││
│  │  State       │────→│   Network    │────→│  Selection   ││
│  │  (32/64      │     │   (Forward   │     │   (Argmax    ││
│  │   squares)   │     │    Pass)     │     │   + Filter)  ││
│  └──────────────┘     └──────────────┘     └──────────────┘│
│         ↑                                       │          │
│         │                                       │          │
│         └───────────────────────────────────────┘          │
│                    (Game Loop)                              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Training Mode (Optional)                │  │
│  │  - Self-play games                                   │  │
│  │  - TD learning updates                               │  │
│  │  - Weight save to flash                              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### Step-by-Step Implementation

**Step 1: Board Representation**

For Checkers (32 playable squares):
```
Use a 1D array of 32 positions
Each position stores: EMPTY, PLAYER_MAN, PLAYER_KING, OPP_MAN, OPP_KING

Example: board[32] = {PLAYER_MAN, EMPTY, OPP_MAN, ...}
```

For Chess (64 squares):
```
Use a 1D array of 64 positions
Each position stores: EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
(Encode color separately or use signed values)

Example: board[64] = {WHITE_ROOK, WHITE_KNIGHT, ..., BLACK_ROOK}
```

**Step 2: Input Encoding Function**

Convert board state to neural network input:
```
For each position:
  - Set one-hot encoded values
  - Example: [1,0,0,0,0] for empty, [0,1,0,0,0] for player man, etc.
  
Add additional features:
  - Turn indicator
  - King count difference
  - Center control (for Chess)
```

**Step 3: Move Generation**

Generate all legal moves for current position:
```
For Checkers:
  - Check each piece for possible moves
  - Handle forced jumps (required captures)
  - Handle multi-jumps (double/triple jumps)
  - Handle king promotion

For Chess:
  - Generate moves for each piece type
  - Handle special moves (castling, en passant, promotion)
  - Filter moves that leave king in check
```

**Step 4: Move Selection**

Use neural network to score and select moves:
```
1. For each legal move:
   - Make move on temporary board
   - Encode new board to input vector
   - Run forward pass
   - Store output score

2. Select move with highest score

3. (Optional) Add small random noise for exploration
```

**Step 5: Training Loop**

Train via self-play:
```
1. Play game against itself
2. Record all positions and moves
3. At game end, calculate TD errors
4. Backpropagate and update weights
5. Save weights periodically to flash
```

#### Memory Requirements (Checkers Example)

| Component | Size |
|-----------|-------|
| Weights (29,264 floats) | 117 KB |
| Game history (50 moves × 200 bytes) | 10 KB |
| Board state + variables | 1 KB |
| **Total PSRAM** | **~128 KB** |

ESP32-S2 with 2 MB PSRAM: **6% used** ✅

---

### Approach B: Using AIfES Library

#### What is AIfES?

AIfES (Artificial Intelligence for Embedded Systems) is a neural network library from Renesas optimized for microcontrollers.

**Website:** https://www.renesas.com/aifes

#### AIfES Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AIfES Framework                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Model Definition Layer                 │   │
│  │  - Define layers (Dense, Conv2D, etc.)             │   │
│  │  - Specify activation functions                    │   │
│  │  - Configure optimizer (SGD, Adam)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Training/Inference Layer               │   │
│  │  - Forward pass                                    │   │
│  │  - Backward pass (training)                        │   │
│  │  - Weight updates                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Hardware Abstraction Layer             │   │
│  │  - Optimized for ARM Cortex-M                      │   │
│  │  - SIMD acceleration                               │   │
│  │  - Memory-efficient operations                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### AIfES Implementation Steps

**Step 1: Define Network Architecture**

```
Layer configuration for Checkers:
- Input layer: 162 neurons
- Dense layer 1: 108 neurons, ReLU activation
- Dense layer 2: 72 neurons, ReLU activation
- Dense layer 3: 36 neurons, ReLU activation
- Output layer: 32 neurons, Sigmoid activation
```

**Step 2: Configure Training**

```
Optimizer: SGD (Stochastic Gradient Descent)
Learning rate: 0.01
Loss function: MSE (Mean Squared Error)
Batch size: 1 (online learning)
```

**Step 3: Prepare Training Data**

```
For each training game:
1. Record (board_state, move_value) pairs
2. At game end, assign final rewards
3. Create training dataset
```

**Step 4: Train Model**

```
For each epoch:
  For each training sample:
    - Forward pass
    - Calculate loss
    - Backward pass
    - Update weights
```

**Step 5: Deploy to ESP32**

```
1. Export trained weights
2. Load weights into ESP32
3. Run inference only (no training on device)
```

#### AIfES vs Custom Implementation

| Feature | Custom NN | AIfES |
|---------|-----------|-------|
| **Flexibility** | High (full control) | Medium (predefined layers) |
| **Performance** | Good | Better (optimized) |
| **Code Size** | Small (~500 lines) | Larger (library overhead) |
| **Learning Curve** | Moderate | Steeper |
| **Training on Device** | Yes | Yes (but slow) |
| **Best For** | Learning, customization | Production, performance |

---

### Approach C: Hybrid (Train on PC, Deploy to ESP32)

#### Why Hybrid?

| Factor | ESP32 Training | PC Training + ESP32 Inference |
|--------|----------------|-------------------------------|
| Training Speed | ~100 games/minute | ~10,000 games/minute |
| Training Time (10K games) | ~100 minutes | ~1 minute |
| Memory Available | ~2 MB PSRAM | 16+ GB RAM |
| Network Size Limit | ~500,000 weights | Unlimited |
| Power Consumption | Higher during training | Lower (inference only) |

#### Hybrid Architecture

```
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│              PC (Training)          │     │         ESP32 (Inference)           │
├─────────────────────────────────────┤     ├─────────────────────────────────────┤
│                                     │     │                                     │
│  ┌───────────────────────────────┐  │     │  ┌───────────────────────────────┐  │
│  │   Large Neural Network        │  │     │  │   Same Network Architecture   │  │
│  │   - More hidden layers        │  │     │  │   - Optimized for inference   │  │
│  │   - More neurons per layer    │  │     │  │   - No training code          │  │
│  │   - Complex features          │  │     │  │   - Weights from PC           │  │
│  └───────────────────────────────┘  │     │  └───────────────────────────────┘  │
│           ↓                         │     │           ↑                         │
│  ┌───────────────────────────────┐  │     │  ┌───────────────────────────────┐  │
│  │   Training Engine             │  │     │  │   Game Engine                 │  │
│  │   - Self-play (fast)          │  │     │  │   - Board representation      │  │
│  │   - TD learning / MCTS        │  │     │  │   - Move generation           │  │
│  │   - Experience replay         │  │     │  │   - Move selection (NN score) │  │
│  └───────────────────────────────┘  │     │  └───────────────────────────────┘  │
│           ↓                         │     │                                     │
│  ┌───────────────────────────────┐  │     │                                     │
│  │   Weight Export               │──┼─────┼─→│  Weight Import                  │  │
│  │   - Save to file              │  │     │  │  - Load from file               │  │
│  │   - Format for ESP32          │  │     │  │  - Validate checksum            │  │
│  └───────────────────────────────┘  │     │  └───────────────────────────────┘  │
└─────────────────────────────────────┘     └─────────────────────────────────────┘
```

#### Step-by-Step Hybrid Implementation

**Phase 1: PC Training**

1. **Set up Python environment:**
   ```
   - Install: numpy, tensorflow/pytorch (optional)
   - Or use pure Python for simplicity
   ```

2. **Implement game logic:**
   ```
   - Board representation
   - Move generation
   - Win detection
   ```

3. **Implement neural network:**
   ```
   - Same architecture as target ESP32 network
   - Can be larger if ESP32 will use subset
   ```

4. **Train via self-play:**
   ```
   - Play 10,000+ games against itself
   - Use TD learning or policy gradients
   - Save best weights
   ```

5. **Export weights:**
   ```
   - Save as C header file or binary
   - Include weight count for validation
   ```

**Phase 2: ESP32 Deployment**

1. **Import weights:**
   ```
   - Load from file or embed in code
   - Validate weight count
   ```

2. **Implement forward pass only:**
   ```
   - No training code needed
   - Smaller memory footprint
   - Faster execution
   ```

3. **Integrate with game engine:**
   ```
   - Board → Input encoding
   - Forward pass → Output scores
   - Select best move
   ```

4. **Test and validate:**
   ```
   - Verify moves match PC version
   - Measure inference time
   - Check memory usage
   ```

#### Weight Export Format (PC → ESP32)

**Option 1: C Header File**
```c
// weights.h - Exported from PC training
const float WEIGHTS[] PROGMEM = {
  0.123456, -0.234567, 0.345678, ...  // All weights
};
const int WEIGHT_COUNT = 262000;
```

**Option 2: Binary File**
```
File format:
- 4 bytes: Magic number (0x43484553 = "CHES")
- 4 bytes: Weight count
- N × 4 bytes: Weights (float32)
- 4 bytes: Checksum
```

**Option 3: Text File (for debugging)**
```
WEIGHTS:262000
0.123456,-0.234567,0.345678,...
CHECKSUM:0x12345678
```

---

### Training Comparison: Chess vs Checkers

| Aspect | Checkers | Chess |
|--------|----------|-------|
| **Input Size** | 162 neurons | 845 neurons |
| **Output Size** | 32 neurons | 64 neurons |
| **Network Size** | ~29,000 weights | ~262,000 weights |
| **Training Games** | 5,000-10,000 | 50,000-100,000 |
| **Training Time (ESP32)** | ~50 hours | ~500 hours |
| **Training Time (PC)** | ~5 minutes | ~30 minutes |
| **Inference Time (ESP32)** | ~5 ms | ~15 ms |
| **Memory (ESP32)** | 116 KB | 1,048 KB |

---

### Recommended Approach by Project Goal

| Goal | Recommended Approach |
|------|---------------------|
| **Learning/Education** | Custom NN (understand every detail) |
| **Rapid Prototyping** | AIfES (pre-built, optimized) |
| **Best Performance** | Hybrid (PC train + ESP32 infer) |
| **Production Device** | Hybrid (most reliable) |
| **Research/Experimentation** | Custom NN (full flexibility) |

---

### Code Reuse from Tic Tac Toe Project

**Directly Reusable:**
- ✅ Neural network forward pass
- ✅ Weight loading/saving
- ✅ TD learning algorithm
- ✅ Game history recording
- ✅ Self-play training loop

**Requires Modification:**
- ⚠️ Board encoding function
- ⚠️ Move generation
- ⚠️ Win detection
- ⚠️ Move validation

**Not Reusable:**
- ❌ Web interface (different game UI needed)
- ❌ Board display (different board layout)

---

## Conclusion

### Games Well-Suited for ESP32

✅ **Highly Recommended:**
- Connect Four
- Nine Men's Morris
- Othello (Reversi)
- Checkers

✅ **Feasible with PSRAM:**
- Full Chess (ESP32-S2/S3 with 2+ MB PSRAM)
- Chess variants (6×6, Capablanca Chess)

❌ **Not Recommended:**
- Go (19×19 board too large for embedded)
- Games with hidden information (Poker, etc.)
- Games with extremely large state spaces (Shogi without simplification)

### Key Takeaways

1. **Start simple:** Adapt to Connect Four before attempting Chess
2. **Calculate first:** Verify memory requirements before coding
3. **Test incrementally:** Verify encoding, then forward pass, then training
4. **Be patient:** Complex games need thousands of training games
5. **Consider hybrid:** Train on PC (fast), deploy to ESP32 (inference only)
6. **Chess IS feasible:** With ~1 MB weights, ESP32-S2/S3 can run chess AI

### Next Steps

1. Choose your target game
2. Select implementation approach (Custom / AIfES / Hybrid)
3. Complete the analysis worksheet (Phase 1)
4. Calculate memory requirements
5. Verify ESP32 compatibility
6. Begin code adaptation (Phase 2)
7. Train and deploy!
