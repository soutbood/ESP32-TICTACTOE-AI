# Neural Network Mathematics - Tic Tac Toe AI

## Table of Contents
1. [Neural Network Architecture Overview](#neural-network-architecture-overview)
2. [Forward Pass Mathematics](#forward-pass-mathematics)
3. [Temporal Difference Learning](#temporal-difference-learning)
4. [Weight Update Mathematics](#weight-update-mathematics)
5. [Backpropagation Derivation](#backpropagation-derivation)
6. [Complete Training Example](#complete-training-example)
7. [Weight Visualization](#weight-visualization)
8. [Verified Parameter Table](#verified-parameter-table)

---

## Neural Network Architecture Overview

### Network Structure

```
Input Layer (27 neurons)     Hidden Layer (54 neurons)    Output Layer (9 neurons)
                                                                            
  ┌─────────┐                  ┌─────────┐                 ┌─────────┐
  │  x[0]   │─────┐            │  h[0]   │─────┐           │  y[0]   │
  ├─────────┤     │            ├─────────┤     │           ├─────────┤
  │  x[1]   │─────┼───┐        │  h[1]   │─────┼───┐       │  y[1]   │
  ├─────────┤     │   │        ├─────────┤     │   │       ├─────────┤
  │  x[2]   │─────┼───┼───┐    │  h[2]   │─────┼───┼───┐   │  y[2]   │
  ├─────────┤     │   │   │    ├─────────┤     │   │   │   ├─────────┤
  │   ...   │─────┼───┼───┼───→│   ...   │─────┼───┼───┼──→│  y[3]   │
  ├─────────┤     │   │   │    ├─────────┤     │   │   │   ├─────────┤
  │  x[26]  │─────┘   │   │    │  h[53]  │─────┘   │   │   │  y[4]   │
  └─────────┘         │   │    └─────────┘         │   │   ├─────────┤
                      │   │                        │   │   │  y[5]   │
                      │   │                        │   │   ├─────────┤
                      │   │                        │   │   │  y[6]   │
                      │   │                        │   │   ├─────────┤
                      │   │                        │   │   │  y[7]   │
                      │   │                        │   │   ├─────────┤
                      │   │                        │   │   │  y[8]   │
                      │   │                        │   │   └─────────┘
                      │   │                        │   │
                  W1: 27×54 weights            W2: 54×9 weights
                  b1: 54 biases                b2: 9 biases
```

### Verified Architecture Parameters

| Parameter | Symbol | Value | Source Code |
|-----------|--------|-------|-------------|
| Board Size | - | 9 | `#define BOARD_SIZE 9` |
| Input Size | n_input | 27 | `#define INPUT_SIZE 27` |
| Hidden Size | n_hidden | 54 | `#define HIDDEN_SIZE 54` |
| Output Size | n_output | 9 | `#define OUTPUT_SIZE 9` |
| Hidden per Position | - | 6 | 54 ÷ 9 = 6 |
| Learning Rate | α | 0.01 | `#define LEARNING_RATE 0.01f` |

### Board Encoding (One-Hot)

Each of the 9 board positions is encoded as 3 values:

| Position State | Encoding | Input Indices |
|----------------|----------|---------------|
| Empty | `[1, 0, 0]` | `[i×3], [i×3+1], [i×3+2]` |
| Player (X) | `[0, 1, 0]` | `[i×3], [i×3+1], [i×3+2]` |
| AI (O) | `[0, 0, 1]` | `[i×3], [i×3+1], [i×3+2]` |

**Example Board:**
```
  X |   | O
  ---------
    | X |  
  ---------
  O |   |  
```

**Input Vector (27 values):**
```
Position 0 (X): [0, 1, 0]  → indices 0, 1, 2
Position 1 ( ): [1, 0, 0]  → indices 3, 4, 5
Position 2 (O): [0, 0, 1]  → indices 6, 7, 8
Position 3 ( ): [1, 0, 0]  → indices 9, 10, 11
Position 4 (X): [0, 1, 0]  → indices 12, 13, 14
Position 5 ( ): [1, 0, 0]  → indices 15, 16, 17
Position 6 (O): [0, 0, 1]  → indices 18, 19, 20
Position 7 ( ): [1, 0, 0]  → indices 21, 22, 23
Position 8 ( ): [1, 0, 0]  → indices 24, 25, 26
```

**Complete Input Vector:**
```
x = [0,1,0,  1,0,0,  0,0,1,  1,0,0,  0,1,0,  1,0,0,  0,0,1,  1,0,0,  1,0,0]
     ──Pos 0──  ──Pos 1──  ──Pos 2──  ──Pos 3──  ──Pos 4──  ──Pos 5──  ──Pos 6──  ──Pos 7──  ──Pos 8──
     (indices 0-26)
```

---

## Forward Pass Mathematics

### Layer 1: Input → Hidden

**Equation:**
```
z₁[i] = b1[i] + Σ(j=0 to 26) x[j] × W1[j, i]    for i = 0 to 53
h[i] = σ(z₁[i])                                   for i = 0 to 53
```

**Where:**
- `x[j]` = input vector (27 values), j ∈ {0, 1, ..., 26}
- `W1[j, i]` = weight from input j to hidden neuron i (27 × 54 = 1,458 values)
- `b1[i]` = bias for hidden neuron i (54 values)
- `z₁[i]` = weighted sum/pre-activation for hidden neuron i
- `h[i]` = hidden layer output after sigmoid activation
- `σ` = sigmoid activation function

**Matrix Form:**
```
z₁ = b1 + xᵀ × W1      (shape: 54 × 1)
h = σ(z₁)              (shape: 54 × 1)
```

**Code Implementation (lines ~700-710):**
```cpp
void run_neural_network() {
  // Layer 1: Input → Hidden
  for(int i = 0; i < HIDDEN_SIZE; i++) {          // i = 0 to 53
    float sum = input_bias[i];                    // b1[i]
    for(int j = 0; j < INPUT_SIZE; j++) {         // j = 0 to 27
      sum += input_vector[j] * input_weights[j * HIDDEN_SIZE + i];  // x[j] × W1[j,i]
    }
    hidden_output[i] = sigmoid(sum);              // h[i] = σ(z₁[i])
  }

  // Layer 2: Hidden → Output
  for(int i = 0; i < OUTPUT_SIZE; i++) {          // i = 0 to 9
    float sum = hidden_bias[i];                   // b2[i]
    for(int j = 0; j < HIDDEN_SIZE; j++) {        // j = 0 to 54
      sum += hidden_output[j] * hidden_weights[j * OUTPUT_SIZE + i]; // h[j] × W2[j,i]
    }
    output_vector[i] = sigmoid(sum);              // y[i] = σ(z₂[i])
  }
}
```

### Sigmoid Activation Function

**Formula:**
```
σ(x) = 1 / (1 + e^(-x))
```

**Implementation (line ~720):**
```cpp
float sigmoid(float x) {
  if(x > 10.0f) return 1.0f;      // Prevent overflow
  if(x < -10.0f) return 0.0f;     // Prevent underflow
  return 1.0f / (1.0f + exp(-x));
}
```

**Properties:**
| Input (x) | Output σ(x) | Derivative σ'(x) |
|-----------|-------------|------------------|
| -∞ | 0.0000 | 0.0000 |
| -10 | 0.000045 | 0.000045 |
| -5 | 0.0067 | 0.0067 |
| -2 | 0.1192 | 0.1050 |
| -1 | 0.2689 | 0.1966 |
| 0 | 0.5000 | 0.2500 |
| +1 | 0.7311 | 0.1966 |
| +2 | 0.8808 | 0.1050 |
| +5 | 0.9933 | 0.0067 |
| +10 | 0.99995 | 0.000045 |
| +∞ | 1.0000 | 0.0000 |

**Derivative (for backpropagation):**
```
σ'(x) = σ(x) × (1 - σ(x))
```

**Proof:**
```
Let σ = σ(x) = 1 / (1 + e^(-x))

dσ/dx = d/dx[(1 + e^(-x))^(-1)]
      = -(1 + e^(-x))^(-2) × d/dx[1 + e^(-x)]
      = -(1 + e^(-x))^(-2) × (-e^(-x))
      = e^(-x) / (1 + e^(-x))²

Now: σ × (1 - σ)
   = [1 / (1 + e^(-x))] × [1 - 1/(1 + e^(-x))]
   = [1 / (1 + e^(-x))] × [(1 + e^(-x) - 1) / (1 + e^(-x))]
   = [1 / (1 + e^(-x))] × [e^(-x) / (1 + e^(-x))]
   = e^(-x) / (1 + e^(-x))²

Therefore: dσ/dx = σ × (1 - σ) ✓
```

### Layer 2: Hidden → Output

**Equation:**
```
z₂[k] = b2[k] + Σ(i=0 to 53) h[i] × W2[i, k]    for k = 0 to 8
y[k] = σ(z₂[k])                                   for k = 0 to 8
```

**Where:**
- `h[i]` = hidden layer output (54 values), i ∈ {0, 1, ..., 53}
- `W2[i, k]` = weight from hidden i to output k (54 × 9 = 486 values)
- `b2[k]` = bias for output neuron k (9 values)
- `z₂[k]` = weighted sum/pre-activation for output neuron k
- `y[k]` = output score for board position k

**Matrix Form:**
```
z₂ = b2 + hᵀ × W2      (shape: 9 × 1)
y = σ(z₂)              (shape: 9 × 1)
```

### Output Interpretation

Each output `y[k]` represents the **desirability** of placing a mark at position k:

| Output Value | Interpretation |
|--------------|----------------|
| 0.0 - 0.3 | Poor move (avoid) |
| 0.3 - 0.5 | Weak move |
| 0.5 - 0.7 | Good move |
| 0.7 - 1.0 | Excellent move |

**Move Selection (lines ~880-895):**
```cpp
int ai_select_move() {
  board_to_input_vector();      // Convert board to 27-element input
  run_neural_network();         // Forward pass → output_vector[9]

  // Collect empty positions
  int moves[BOARD_SIZE];
  int count = 0;
  for(int i = 0; i < BOARD_SIZE; i++) {
    if(board[i] == EMPTY) moves[count++] = i;
  }

  if(count == 0) return -1;     // No moves available

  // Find empty position with highest output
  int best = moves[0];
  float best_score = output_vector[moves[0]];
  for(int i = 1; i < count; i++) {
    if(output_vector[moves[i]] > best_score) {
      best_score = output_vector[moves[i]];
      best = moves[i];
    }
  }
  return best;
}
```

---

## Temporal Difference Learning

### Core Concept

Temporal Difference (TD) Learning updates predictions based on the **difference** between:
- **Predicted value**: What the network expected
- **Target value**: What actually happened (or what the next state predicts)

**TD Error Formula:**
```
δ = Target - Predicted
```

### TD Learning in Tic Tac Toe

**Problem:** We only know the game outcome at the end, but we need to credit/blame individual moves.

**Solution:** Use the value of the **next state** as the target for the **current state**.

### Backup Diagram

```
Move 0 → State S₀ → Network predicts V(S₀) = 0.50
                    │
                    ↓
Move 1 → State S₁ → Network predicts V(S₁) = 0.60
                    │
                    ↓
Move 2 → State S₂ → Network predicts V(S₂) = 0.70
                    │
                    ↓
Move 3 → State S₃ → Network predicts V(S₃) = 0.80
                    │
                    ↓
                  ...
                    │
                    ↓
Game End → Result R ∈ {1.0 (win), 0.5 (draw), 0.0 (loss)}
```

### TD Target Calculation

**For the last move (move N):**
```
Target[N] = Final Reward
δ[N] = Final Reward - Predicted[N]
```

**For earlier moves (move t, where t < N):**
```
Target[t] = Predicted Value of next state = Predicted[t+1]
δ[t] = Predicted[t+1] - Predicted[t]
```

**Code Implementation (lines ~740-760):**
```cpp
void update_weights_td_learning(bool game_won) {
  // Set final reward based on game outcome
  float final_reward = game_won ? 1.0f : (winner == -1 ? 0.5f : 0.0f);
  //                                   win     draw         loss

  // Count recorded moves in history
  int num_moves = 0;
  for(int i = 0; i < MAX_GAME_MOVES; i++) {
    if(game_history[i].move_made == -1) break;
    num_moves++;
  }

  // Backpropagate TD error from last move to first
  for(int i = num_moves - 1; i >= 0; i--) {
    // Target = final reward (last move) or next move's predicted value
    float target = (i == num_moves - 1) ? final_reward : game_history[i + 1].move_value_before;
    
    // TD Error = prediction error
    float td_error = target - game_history[i].move_value_before;
    
    // Update weights for this specific move
    update_network_for_move(game_history[i].board_state, 
                            game_history[i].move_made, 
                            td_error);
  }

  Serial.println("Network updated");
}
```

### Why TD Learning Works

**Intuition:**
1. **Last move**: Target = game outcome (direct feedback: win=1, draw=0.5, loss=0)
2. **Second-to-last move**: Target = value of position after that move
3. **Earlier moves**: Each move's target is the value of the state it leads to
4. This "backs up" the final result through the entire sequence of moves

**Example Game with TD Updates:**
```
Move 0: AI predicts V(S₀) = 0.50 (neutral starting position)
Move 1: AI predicts V(S₁) = 0.55 (slightly better)
Move 2: AI predicts V(S₂) = 0.65 (developing advantage)
Move 3: AI predicts V(S₃) = 0.75 (strong position)
Move 4: AI WINS! (final reward = 1.0)

TD Updates (processed backwards):
┌─────────┬──────────────┬──────────────┬───────────┬─────────────┐
│ Move    │ Target       │ Predicted    │ TD Error  │ Update      │
├─────────┼──────────────┼──────────────┼───────────┼─────────────┤
│ 4 (last)│ 1.0 (win)    │ 0.75         │ +0.25     │ Increase    │
│ 3       │ 0.75         │ 0.65         │ +0.10     │ Increase    │
│ 2       │ 0.65         │ 0.55         │ +0.10     │ Increase    │
│ 1       │ 0.55         │ 0.50         │ +0.05     │ Increase    │
│ 0       │ 0.50         │ 0.50         │  0.00     │ No change   │
└─────────┴──────────────┴──────────────┴───────────┴─────────────┘

Interpretation:
- Move 4 directly led to win → large positive update
- Moves 1-3 led to increasingly better positions → moderate positive updates
- Move 0 was already accurate → no update needed
```

---

## Weight Update Mathematics

### Gradient Descent Principle

**Goal:** Minimize prediction error by adjusting weights in the direction that reduces error.

**Update Rule:**
```
W_new = W_old + ΔW
ΔW = α × (-∂E/∂W)
```

**Where:**
- `α` = learning rate (0.01 in this implementation)
- `E` = error/loss function
- `-∂E/∂W` = negative gradient (direction of steepest descent)

### Loss Function

**Mean Squared Error (MSE):**
```
E = ½ × (Target - Output)² = ½ × δ²
```

**Why ½?** Makes the derivative cleaner (the 2 cancels with the exponent).

### Output Layer Weight Updates (W2)

**Gradient Derivation:**

Using the chain rule:
```
∂E/∂W2[i,k] = ∂E/∂y[k] × ∂y[k]/∂z₂[k] × ∂z₂[k]/∂W2[i,k]
```

**Step 1: Error w.r.t. output**
```
∂E/∂y[k] = ∂/∂y[k] [½ × (Target[k] - y[k])²]
         = -(Target[k] - y[k])
         = -δ[k]
```

**Step 2: Output w.r.t. pre-activation**
```
∂y[k]/∂z₂[k] = ∂/∂z₂[k] [σ(z₂[k])]
             = σ'(z₂[k])
             = y[k] × (1 - y[k])
```

**Step 3: Pre-activation w.r.t. weight**
```
∂z₂[k]/∂W2[i,k] = ∂/∂W2[i,k] [b2[k] + Σ(j) h[j] × W2[j,k]]
                = h[i]
```

**Combined:**
```
∂E/∂W2[i,k] = -δ[k] × y[k] × (1 - y[k]) × h[i]
```

**Simplified (absorbing sigmoid derivative into error term):**

In practice, we use the TD error δ directly:
```
∂E/∂W2[i,k] ≈ -δ × h[i]
```

**Weight Update:**
```
W2[i,k]_new = W2[i,k]_old + α × δ × h[i]
```

**Code Implementation (lines ~765-770):**
```cpp
void update_network_for_move(float* board_state, int move_idx, float td_error) {
  // ... set input vector and run forward pass ...

  // Update hidden→output weights
  for(int i = 0; i < HIDDEN_SIZE; i++) {          // i = 0 to 53
    float grad = LEARNING_RATE * td_error * hidden_output[i];
    // grad = α × δ × h[i]
    hidden_weights[i * OUTPUT_SIZE + move_idx] += grad;
    // W2[i, move_idx] += α × δ × h[i]
  }

  // ... update input→hidden weights ...
}
```

**Numerical Example:**
```
Given:
- TD error δ = +0.2 (underestimated by 0.2)
- Hidden activation h[5] = 0.75
- Learning rate α = 0.01
- Current weight W2[5, 4] = 0.30

Update:
ΔW2[5, 4] = 0.01 × 0.2 × 0.75 = 0.0015
W2[5, 4]_new = 0.30 + 0.0015 = 0.3015

Interpretation: Since δ > 0 (underestimated), increase the weight
to make this output more likely for similar inputs in the future.
```

### Hidden Layer Weight Updates (W1) - Backpropagation

**Gradient Derivation:**

Using the chain rule:
```
∂E/∂W1[j,i] = ∂E/∂h[i] × ∂h[i]/∂z₁[i] × ∂z₁[i]/∂W1[j,i]
```

**Step 1: Error w.r.t. hidden activation**

The error flows back from the output layer:
```
∂E/∂h[i] = Σ(k) ∂E/∂z₂[k] × ∂z₂[k]/∂h[i]
```

For a single output neuron (the chosen move k = move_idx):
```
∂E/∂z₂[k] = -δ (from output layer)
∂z₂[k]/∂h[i] = W2[i, k]

Therefore:
∂E/∂h[i] = -δ × W2[i, move_idx]
```

**Step 2: Hidden activation w.r.t. pre-activation**
```
∂h[i]/∂z₁[i] = σ'(z₁[i]) = h[i] × (1 - h[i])
```

**Step 3: Pre-activation w.r.t. weight**
```
∂z₁[i]/∂W1[j,i] = ∂/∂W1[j,i] [b1[i] + Σ(m) x[m] × W1[m, i]]
                = x[j]
```

**Combined:**
```
∂E/∂W1[j,i] = (-δ × W2[i, move_idx]) × (h[i] × (1 - h[i])) × x[j]
            = -δ × W2[i, move_idx] × h[i] × (1 - h[i]) × x[j]
```

**Weight Update:**
```
W1[j,i]_new = W1[j,i]_old + α × δ × W2[i, move_idx] × h[i] × (1 - h[i]) × x[j]
```

**Code Implementation (lines ~772-780):**
```cpp
void update_network_for_move(float* board_state, int move_idx, float td_error) {
  // ... set input vector and run forward pass ...

  // Update input→hidden weights
  for(int i = 0; i < INPUT_SIZE; i++) {           // i = 0 to 26
    for(int j = 0; j < HIDDEN_SIZE; j++) {        // j = 0 to 53
      // Error propagated from output layer through W2
      float out_err = hidden_weights[j * OUTPUT_SIZE + move_idx] * td_error;
      // out_err = W2[j, move_idx] × δ

      // Full gradient with sigmoid derivative
      float grad = LEARNING_RATE * out_err * input_vector[i] * 
                   hidden_output[j] * (1.0f - hidden_output[j]);
      // grad = α × (W2[j, move_idx] × δ) × x[i] × h[j] × (1 - h[j])

      input_weights[i * HIDDEN_SIZE + j] += grad;
      // W1[i, j] += grad
    }
  }
}
```

**Numerical Example:**
```
Given:
- TD error δ = +0.2
- Weight from hidden to output: W2[5, 4] = 0.30
- Hidden activation: h[5] = 0.75
- Input activation: x[10] = 1.0 (position 3 is empty)
- Learning rate: α = 0.01
- Current weight: W1[10, 5] = 0.15

Step-by-step calculation:
1. Output error at hidden 5:
   out_err = W2[5, 4] × δ = 0.30 × 0.2 = 0.06

2. Sigmoid derivative:
   h[5] × (1 - h[5]) = 0.75 × 0.25 = 0.1875

3. Gradient:
   grad = 0.01 × 0.06 × 1.0 × 0.1875 = 0.0001125

4. Weight update:
   W1[10, 5]_new = 0.15 + 0.0001125 = 0.1501125

Interpretation: The connection from input 10 (empty position 3)
to hidden neuron 5 is slightly strengthened because this pattern
contributed to a positive outcome.
```

### Bias Updates

**Hidden Layer Bias (b1):**

Same as weight update, but input is always 1:
```
b1[i]_new = b1[i]_old + α × δ × W2[i, move_idx] × h[i] × (1 - h[i])
```

**Output Layer Bias (b2):**

Direct gradient:
```
b2[k]_new = b2[k]_old + α × δ
```

**Note:** The current implementation does not explicitly update biases separately.
Biases are stored in `input_bias[]` and `hidden_bias[]` arrays but the update
function only modifies `input_weights[]` and `hidden_weights[]`.

---

## Backpropagation Derivation

### Complete Mathematical Derivation

**Loss Function (Mean Squared Error):**
```
E = ½ × (Target - Output)² = ½ × δ²
```

### Output Layer Gradients

**For output neuron k:**

```
∂E/∂y[k] = -(Target[k] - y[k]) = -δ[k]
```

**Pre-activation gradient:**
```
∂E/∂z₂[k] = ∂E/∂y[k] × ∂y[k]/∂z₂[k]
          = -δ[k] × σ'(z₂[k])
          = -δ[k] × y[k] × (1 - y[k])
```

**Weight gradient:**
```
∂E/∂W2[i,k] = ∂E/∂z₂[k] × ∂z₂[k]/∂W2[i,k]
            = -δ[k] × y[k] × (1 - y[k]) × h[i]
```

**Simplified (using TD error directly):**
```
∂E/∂W2[i,k] ≈ -δ × h[i]

Update: W2[i,k] += α × δ × h[i]
```

### Hidden Layer Gradients

**Error backpropagated to hidden neuron i:**
```
∂E/∂h[i] = Σ(k) ∂E/∂z₂[k] × ∂z₂[k]/∂h[i]
         = Σ(k) (-δ[k] × y[k] × (1 - y[k])) × W2[i,k]
```

**For single output (chosen move k = move_idx):**
```
∂E/∂h[i] = -δ × W2[i, move_idx]    (simplified)
```

**Pre-activation gradient:**
```
∂E/∂z₁[i] = ∂E/∂h[i] × ∂h[i]/∂z₁[i]
          = -δ × W2[i, move_idx] × h[i] × (1 - h[i])
```

**Weight gradient:**
```
∂E/∂W1[j,i] = ∂E/∂z₁[i] × ∂z₁[i]/∂W1[j,i]
            = -δ × W2[i, move_idx] × h[i] × (1 - h[i]) × x[j]
```

**Update rule:**
```
W1[j,i] += α × δ × W2[i, move_idx] × h[i] × (1 - h[i]) × x[j]
```

### Gradient Descent Update Summary

**General form:**
```
W_new = W_old - α × ∂E/∂W
      = W_old + α × (-∂E/∂W)
```

**Output layer (W2):**
```
W2[i,k]_new = W2[i,k]_old + α × δ × h[i]
```

**Hidden layer (W1):**
```
W1[j,i]_new = W1[j,i]_old + α × δ × W2[i, move_idx] × h[i] × (1 - h[i]) × x[j]
```

---

## Complete Training Example

### Example Game Walkthrough

**Initial State (Empty Board):**
```
Board: [ , , ,  , , ,  , , ]
       ─────────────────────
Input: [1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0]
        ──0── ──1── ──2── ──3── ──4── ──5── ──6── ──7── ──8──
```

### Move 1: AI plays center (position 4)

**Forward Pass:**
```
Input vector x (27 values) → Neural Network → Output y (9 values)

Assume network outputs:
y = [0.30, 0.25, 0.30, 0.25, 0.82, 0.25, 0.30, 0.25, 0.30]
     ──────────────────────────────────────────────────────
     Position 4 has highest output (0.82) → AI chooses position 4
```

**Record in History:**
```cpp
game_history[0].board_state = current_input_vector;  // 27 values
game_history[0].move_made = 4;
game_history[0].move_value_before = 0.82;
```

**Board after Move 1:**
```
[ , , ,  ,O, ,  , , ]   (AI = O)
```

### Move 2: Player plays corner (position 0)

**Board:**
```
[X, , ,  ,O, ,  , , ]   (Player = X, AI = O)
```

### Move 3: AI plays opposite corner (position 8)

**Forward Pass:**
```
Input: [0,1,0, 1,0,0, 1,0,0, 1,0,0, 0,0,1, 1,0,0, 1,0,0, 1,0,0, 0,0,1]
        ──0── ──1── ──2── ──3── ──4── ──5── ──6── ──7── ──8──

Assume outputs:
y = [0.10, 0.35, 0.25, 0.30, 0.45, 0.20, 0.25, 0.30, 0.72]
                                         ──────────────────
                                         Position 8 has highest (0.72)
```

**Record in History:**
```cpp
game_history[1].board_state = current_input_vector;
game_history[1].move_made = 8;
game_history[1].move_value_before = 0.72;
```

**Board after Move 3:**
```
[X, , ,  ,O, ,  , ,O]
```

### Continue Game...

**Assume Final Result: AI Wins on Move 5!**

**Complete Game History:**
| Move Index | Board State Description | Move Made | Predicted Value |
|------------|------------------------|-----------|-----------------|
| 0 | Empty board | 4 | 0.82 |
| 1 | After player move 0 | 8 | 0.72 |
| 2 | After player move 2 | 6 | 0.68 |
| 3 | After player move 5 | 3 | 0.85 |

### Backward Pass (Training)

**Final Reward:** 1.0 (AI won)

**TD Errors (computed backwards):**

**Move 3 (last move, index 3):**
```
Target = 1.0 (final reward - AI won!)
Predicted = 0.85
δ = 1.0 - 0.85 = +0.15

Interpretation: AI underestimated this winning position by 0.15
Action: Increase weights to raise predicted value for similar positions
```

**Move 2 (index 2):**
```
Target = 0.85 (value of next state after move 3)
Predicted = 0.68
δ = 0.85 - 0.68 = +0.17

Interpretation: Move 2 led to a much better position than expected
Action: Significantly increase weights (largest TD error)
```

**Move 1 (index 1):**
```
Target = 0.68 (value of next state after move 2)
Predicted = 0.72
δ = 0.68 - 0.72 = -0.04

Interpretation: Slightly overestimated this position
Action: Slightly decrease weights
```

**Move 0 (index 0):**
```
Target = 0.72 (value of next state after move 1)
Predicted = 0.82
δ = 0.72 - 0.82 = -0.10

Interpretation: Overestimated the opening position
Action: Decrease weights for this opening pattern
```

### Weight Change Calculation (Move 3, δ = +0.15)

**Hidden→Output Weight Update:**

```
Given:
- TD error: δ = +0.15
- Hidden neuron 10 activation: h[10] = 0.68
- Learning rate: α = 0.01
- Current weight: W2[10, 3] = 0.25

Calculation:
ΔW2[10, 3] = α × δ × h[10]
           = 0.01 × 0.15 × 0.68
           = 0.00102

W2[10, 3]_new = 0.25 + 0.00102 = 0.25102

Interpretation: Connection from hidden neuron 10 to output 3
is strengthened because this pattern contributed to a win.
```

**Input→Hidden Weight Update:**

```
Given:
- TD error: δ = +0.15
- Weight W2[10, 3] = 0.25 (hidden 10 → output 3)
- Hidden activation: h[10] = 0.68
- Sigmoid derivative: h[10] × (1 - h[10]) = 0.68 × 0.32 = 0.2176
- Input 5 active: x[5] = 1.0 (position 1 is empty)
- Current weight: W1[5, 10] = 0.12

Calculation:
out_err = W2[10, 3] × δ = 0.25 × 0.15 = 0.0375

grad = α × out_err × x[5] × h[10] × (1 - h[10])
     = 0.01 × 0.0375 × 1.0 × 0.2176
     = 0.0000816

W1[5, 10]_new = 0.12 + 0.0000816 = 0.1200816

Interpretation: The pattern "position 1 empty" contributing to
hidden neuron 10 is slightly strengthened.
```

---

## Weight Visualization

### Weight Matrix Structure

**Input Weights (27 × 54 = 1,458 values):**

```
Memory layout: input_weights[j * HIDDEN_SIZE + i]
where j = input index (0-26), i = hidden index (0-53)

        h[0]    h[1]    h[2]   ...   h[53]
       ┌───────────────────────────────────┐
x[0]   │ w₀,₀    w₀,₁    w₀,₂   ...  w₀,₅₃ │  ← Position 0, empty
x[1]   │ w₁,₀    w₁,₁    w₁,₂   ...  w₁,₅₃ │  ← Position 0, player
x[2]   │ w₂,₀    w₂,₁    w₂,₂   ...  w₂,₅₃ │  ← Position 0, AI
x[3]   │ w₃,₀    w₃,₁    w₃,₂   ...  w₃,₅₃ │  ← Position 1, empty
 ...   │  ...     ...     ...    ...   ... │
x[26]  │ w₂₆,₀   w₂₆,₁   w₂₆,₂  ...  w₂₆,₅₃│  ← Position 8, AI
       └───────────────────────────────────┘

Total: 27 × 54 = 1,458 floats = 5,832 bytes
```

**Hidden Weights (54 × 9 = 486 values):**

```
Memory layout: hidden_weights[j * OUTPUT_SIZE + i]
where j = hidden index (0-53), i = output index (0-8)

         y[0]    y[1]    y[2]   ...   y[8]
        ┌──────────────────────────────────┐
h[0]    │ w'₀,₀   w'₀,₁   w'₀,₂  ...  w'₀,₈│
h[1]    │ w'₁,₀   w'₁,₁   w'₁,₂  ...  w'₁,₈│
 ...    │  ...     ...     ...   ...   ... │
h[53]   │ w'₅₃,₀  w'₅₃,₁  w'₅₃,₂  ...  w'₅₃,₈│
        └──────────────────────────────────┘

Total: 54 × 9 = 486 floats = 1,944 bytes
```

### Verified Weight Count

| Component | Formula | Count | Bytes (float = 4) |
|-----------|---------|-------|-------------------|
| Input weights (W1) | 27 × 54 | 1,458 | 5,832 |
| Hidden weights (W2) | 54 × 9 | 486 | 1,944 |
| Input bias (b1) | 54 | 54 | 216 |
| Hidden bias (b2) | 9 | 9 | 36 |
| **Total** | | **2,007** | **8,028** |

**Note:** The comment in `ai_weights.h` incorrectly states 2,025 values.
The correct total is **2,007 values**.

### Weight Evolution During Training

**Initial Weights (Random):**
```cpp
// From initialize_neural_network():
input_weights[i] = (random(200) - 100) / 1000.0f;  // Range: [-0.1, 0.1]
hidden_weights[i] = (random(200) - 100) / 1000.0f; // Range: [-0.1, 0.1]
biases[i] = 0.0f;

Sample: [-0.052, 0.083, -0.019, 0.047, -0.071, ...]
Mean ≈ 0, StdDev ≈ 0.058
```

**After 100 Games:**
```
Weights begin showing patterns:
- Positive for winning configurations
- Negative for losing configurations

Sample: [-0.12, 0.08, 0.15, -0.09, 0.03, ...]
Some weights growing, others shrinking
```

**After 1,000 Games:**
```
Weights more differentiated:
- Strong positive for immediate wins
- Strong negative for opponent threats
- Subtle patterns for forks, traps

Sample: [-0.35, 0.22, 0.48, -0.31, 0.15, ...]
Clear structure emerging
```

**After 10,000+ Games:**
```
Weights converge to near-optimal:
- Consistent evaluation of board patterns
- Rare mistakes
- Near-perfect play

Sample: [-0.67, 0.41, 0.89, -0.52, 0.33, ...]
Stable, well-trained weights
```

### Learning Rate Effects

**Large Learning Rate (α = 0.1):**
```
Advantages:
✓ Fast initial learning
✓ Quick adaptation to new patterns

Disadvantages:
✗ Oscillations around optimal values
✗ May overshoot good weights
✗ Less stable training
✗ Risk of divergence
```

**Small Learning Rate (α = 0.001):**
```
Advantages:
✓ Stable, smooth convergence
✓ Fine-tuned final weights
✓ No oscillation

Disadvantages:
✗ Very slow learning
✗ May get stuck in local minima
✗ Requires 10× more games
```

**Current Setting (α = 0.01):**
```
Balanced approach:
✓ Reasonable learning speed
✓ Stable convergence
✓ Good for embedded training
✓ ~45 games per training session is sufficient
```

---

## Verified Parameter Table

### Architecture Parameters

| Parameter | Symbol | Value | Code Definition |
|-----------|--------|-------|-----------------|
| Board Size | - | 9 | `#define BOARD_SIZE 9` |
| Input Size | nᵢ | 27 | `#define INPUT_SIZE 27` |
| Hidden Size | nₕ | 54 | `#define HIDDEN_SIZE 54` |
| Output Size | nₒ | 9 | `#define OUTPUT_SIZE 9` |
| Hidden per Position | nₕ/pos | 6 | 54 ÷ 9 = 6 |

### Weight Dimensions

| Weight Matrix | Dimensions | Count | Memory (bytes) |
|---------------|------------|-------|----------------|
| W1 (Input→Hidden) | 27 × 54 | 1,458 | 5,832 |
| W2 (Hidden→Output) | 54 × 9 | 486 | 1,944 |
| b1 (Hidden Bias) | 54 × 1 | 54 | 216 |
| b2 (Output Bias) | 9 × 1 | 9 | 36 |
| **Total** | | **2,007** | **8,028** |

### Training Parameters

| Parameter | Symbol | Value | Code Definition |
|-----------|--------|-------|-----------------|
| Learning Rate | α | 0.01 | `#define LEARNING_RATE 0.01f` |
| Save Interval | - | 500 | `#define SAVE_WEIGHTS_INTERVAL 500` |
| Fast Train Games | - | 45 | `#define FAST_TRAIN_GAMES 45` |
| Fast Train Delay | - | 10 ms | `#define FAST_TRAIN_DELAY 10` |
| Max Game Moves | - | 20 | `#define MAX_GAME_MOVES 20` |

### Multi-User Parameters

| Parameter | Value | Code Definition |
|-----------|-------|-----------------|
| Max Players | 4 | `#define MAX_PLAYERS 4` |
| Player Timeout | 60,000 ms | `#define PLAYER_TIMEOUT_MS 60000UL` |
| Bytes per Player | 20 | `struct PlayerGame` |
| Total Player Memory | 80 bytes | 4 × 20 |

### Computational Complexity

| Operation | Formula | Count |
|-----------|---------|-------|
| **Forward Pass** | | |
| Input→Hidden mults | 27 × 54 | 1,458 |
| Input→Hidden adds | 27 × 54 | 1,458 |
| Hidden activations | 54 | 54 |
| Hidden→Output mults | 54 × 9 | 486 |
| Hidden→Output adds | 54 × 9 | 486 |
| Output activations | 9 | 9 |
| **Forward Total** | | **~3,960 ops** |
| | | |
| **Backward Pass (per move)** | | |
| Output weight updates | 54 | 54 |
| Hidden weight updates | 27 × 54 | 1,458 |
| **Backward Total** | | **~1,512 ops** |
| | | |
| **Per Game (avg 5 AI moves)** | | |
| Forward ops | 5 × 3,960 | 19,800 |
| Backward ops | 5 × 1,512 | 7,560 |
| **Game Total** | | **~27,360 ops** |

### Memory Layout Summary

| Component | Location | Size |
|-----------|----------|------|
| Player data | PSRAM | 80 bytes |
| Neural network weights | PSRAM | 8,028 bytes |
| Game history | PSRAM | ~2,240 bytes |
| **Total PSRAM** | | **~10.3 KB** |
| | | |
| Weights file (flash) | LittleFS | 8,028 bytes |
| Embedded weights | PROGMEM | 8,028 bytes |

---

## Appendix: Sigmoid Function Reference Table

| x | σ(x) | σ'(x) = σ(x)(1-σ(x)) | Interpretation |
|---|------|---------------------|----------------|
| -10.0 | 0.000045 | 0.000045 | Saturated low |
| -5.0 | 0.006693 | 0.006648 | Very low |
| -4.0 | 0.017986 | 0.017662 | Very low |
| -3.0 | 0.047426 | 0.045185 | Low |
| -2.0 | 0.119203 | 0.104994 | Below average |
| -1.5 | 0.182426 | 0.149069 | Below average |
| -1.0 | 0.268941 | 0.196612 | Below average |
| -0.5 | 0.377541 | 0.235004 | Slightly below |
| 0.0 | 0.500000 | 0.250000 | **Neutral (max gradient)** |
| +0.5 | 0.622459 | 0.235004 | Slightly above |
| +1.0 | 0.731059 | 0.196612 | Above average |
| +1.5 | 0.817574 | 0.149069 | Above average |
| +2.0 | 0.880797 | 0.104994 | High |
| +3.0 | 0.952574 | 0.045185 | Very high |
| +4.0 | 0.982014 | 0.017662 | Very high |
| +5.0 | 0.993307 | 0.006648 | Saturated high |
| +10.0 | 0.999955 | 0.000045 | Saturated high |

**Key Properties:**
- Maximum derivative at x = 0: **σ'(0) = 0.25**
- Range: **σ(x) ∈ (0, 1)** for all finite x
- Symmetry: **σ(-x) = 1 - σ(x)**
- Derivative symmetry: **σ'(-x) = σ'(x)**
