/*
 * Tic Tac Toe AI for ESP32-S2-MINI - MULTI-USER OPTIMIZED VERSION
 * 
 * Copyright (C) 2024 Tic Tac Toe AI Project
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Features:
 * - ESPAsyncWebServer for non-blocking concurrent connections
 * - Shared neural network (single instance for all players)
 * - Per-player minimal state (19 bytes each, 4 players = 76 bytes)
 * - HTML stored in PROGMEM (flash, not RAM)
 * - Idle connection timeout (60 seconds)
 * - Embedded weights (optional) - weights stored in code
 *
 * Compatible with: ESP32, ESP32-S2, ESP32-S3
 * Requires: ESPAsyncWebServer + AsyncTCP libraries
 *
 * Hardware: ESP32-S2-MINI (4MB Flash, 2MB PSRAM recommended)
 *
 * Installation:
 *   Arduino IDE > Sketch > Include Library > Manage Libraries
 *   Search and install: "ESPAsyncWebServer" and "AsyncTCP"
 *
 * Usage: Connect to WiFi "TicTacToe_AI" (password: 12345678)
 *        Open browser: 192.168.4.1
 *        Each device gets their own game session
 */

#include <Arduino.h>
#include <LittleFS.h>
#include <math.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <AsyncTCP.h>

#define FILESYSTEM LittleFS

// WiFi Access Point settings
const char* AP_SSID = "TicTacToe_AI";
const char* AP_PASSWORD = "12345678";

// Async web server on port 80
AsyncWebServer server(80);

// ============ GAME CONSTANTS ============
#define BOARD_SIZE 9
#define INPUT_SIZE 27
#define HIDDEN_SIZE 54       // 6 hidden neurons per board position
#define OUTPUT_SIZE 9
#define EMPTY 0
#define PLAYER_X 1
#define AI_O 2

// ============ MULTI-USER CONFIGURATION ============
#define MAX_PLAYERS 4                // Max concurrent players (RAM: ~80 bytes)
#define PLAYER_TIMEOUT_MS 60000UL    // Idle timeout (60 seconds)

// ============ TRAINING CONFIGURATION ============
#define SAVE_WEIGHTS_INTERVAL 500
#define LEARNING_RATE 0.01f
#define FAST_TRAIN_GAMES 45
#define FAST_TRAIN_DELAY 10
#define MAX_GAME_MOVES 20

// ============ EMBEDDED WEIGHTS CONFIGURATION ============
// Set to 1 to use embedded weights from ai_weights.h
// Set to 0 to load from filesystem only
#define USE_EMBEDDED_WEIGHTS 1

#if USE_EMBEDDED_WEIGHTS
#include <pgmspace.h>
#include "ai_weights.h"
#endif

// ============ GAME STATE VARIABLES ============
int board[BOARD_SIZE];
bool game_over = false;
int winner = EMPTY;
bool player_turn = true;
int games_played = 0;
bool fast_training_mode = false;
int fast_train_target = FAST_TRAIN_GAMES;

// Statistics
int ai_wins = 0;
int player_wins = 0;
int random_wins = 0;
int selfplay_wins_x = 0;  // AI (X) wins in self-play
int selfplay_wins_o = 0;  // AI (O) wins in self-play
int selfplay_draws = 0;   // Draws in self-play

// Training control
bool stop_training_requested = false;
bool selfplay_training_mode = false;  // true = AI vs AI, false = AI vs Random

// Self-play training state
int selfplay_train_count = 0;
bool selfplay_x_turn = true;
bool selfplay_first_run = true;

// ============ PER-PLAYER GAME STATE (PSRAM) ============
struct PlayerGame {
  uint32_t ip;           // Player IP (4 bytes) - identifies session
  uint8_t board[9];      // Board state (9 bytes) - 0=empty, 1=X, 2=O
  uint8_t player_sym;    // Player symbol (1 byte) - X or O
  uint8_t active;        // Active flag (1 byte) - 0=free, 1=ready, 2=playing
  uint8_t player_turn;   // Player's turn flag (1 byte) - 0=AI turn, 1=player turn
  uint32_t last_seen;    // Last activity timestamp (4 bytes) - timeout tracking
                         // TOTAL: 20 bytes per player
};

PlayerGame* players;  // Pointer to PSRAM array (4 players × 19 bytes = 76 bytes)

// ============ NEURAL NETWORK (SHARED - PSRAM) ============
float* input_weights;
float* hidden_weights;
float* input_bias;
float* hidden_bias;
float* input_vector;
float* hidden_output;
float* output_vector;

// Game history for training (PSRAM)
struct GameMove {
  float board_state[INPUT_SIZE];
  int move_made;
  float move_value_before;
};
GameMove* game_history;

// ============ FORWARD DECLARATIONS ============
void allocate_psram_memory();
void initialize_neural_network();
bool load_weights_from_flash();
bool save_weights_to_flash();
#if USE_EMBEDDED_WEIGHTS
bool load_embedded_weights();
#endif
void update_weights_td_learning(bool game_won);
void update_network_for_move(float* board_state, int move_idx, float td_error);
void initialize_game(bool silent_mode);
void print_board();
int check_winner();
bool is_game_over();
void end_game();
int random_move();
void fast_train_session();
void fast_train_self_play_session();  // NEW: AI vs AI self-play training
void process_serial_command();
void print_help();
void export_weights();
void export_weights_as_header();
bool import_weights(const char* data);
void save_weights_to_file();
void load_weights_from_file();
void setup_wifi_ap();
void setup_web_server();
void stop_training();

// Multi-user web handlers
int find_player_slot(uint32_t ip);
void handle_web_root(AsyncWebServerRequest *request);
void handle_web_move(AsyncWebServerRequest *request);
void handle_web_newgame(AsyncWebServerRequest *request);
void handle_web_status(AsyncWebServerRequest *request);
void handle_web_choose_symbol(AsyncWebServerRequest *request);
int check_winner_player(uint8_t* pboard);
int ai_select_move_player(uint8_t* pboard, int ai_sym, int player_sym);
void update_player_timeout(uint32_t ip);

// ============ COMPRESSED HTML (PROGMEM) ============
const char HTML_PAGE[] PROGMEM = R"rawliteral(
<!DOCTYPE html><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TicTacToe AI</title><style>
body{font-family:sans-serif;text-align:center;background:#1a1a2e;color:#eee;margin:0;padding:10px}
h1{color:#00d9ff;font-size:1.5em;margin:10px 0}
.board{display:grid;grid-template-columns:repeat(3,50px);gap:3px;margin:10px auto;max-width:170px}
.cell{width:50px;height:50px;background:#16213e;border:2px solid #00d9ff;border-radius:5px;font-size:28px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center}
.cell:active{background:#0f3460}.X{color:#ff6b6b}.O{color:#4ecdc4}
.btn{background:#00d9ff;color:#1a1a2e;border:none;padding:8px 16px;margin:3px;border-radius:5px;cursor:pointer;font-size:14px}
.btn:active{background:#00b8d4}
.stat{background:#16213e;padding:8px;margin:5px auto;max-width:200px;border-radius:5px;font-size:12px}
.sym{padding:12px 20px;font-size:16px;margin:5px}.sym-s{background:#4ecdc4}
#res{font-size:18px;font-weight:700;margin:10px;min-height:25px}
</style>
<h1>TicTacToe AI</h1>
<div class="stat">AI:<b id="aiW">%AW</b> Ply:<b id="pW">%PW</b> Dr:<b id="dr">%DR</b> Rate:<b id="rt">%RT</b></div>
<h3>Symbol:</h3><button class="btn sym %XS%" onclick="sel('X')">X</button><button class="btn sym %OS%" onclick="sel('O')">O</button>
<div id="res">%RS%</div><div class="board">%BD%</div>
<button class="btn" onclick="ng()">New Game</button><button class="btn" onclick="st()">Status</button>
<script>
function m(p){fetch("/m?p="+p).then(()=>location.reload());}
function ng(){fetch("/ng").then(()=>location.reload());}
function st(){fetch("/st").then(r=>r.text()).then(d=>alert(d));}
function sel(s){fetch("/sy?s="+s).then(()=>location.reload());}
</script>
)rawliteral";

// ============ SETUP ============
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=================================");
  Serial.println("Tic Tac Toe AI - Multi-User");
  Serial.println("Max players: " + String(MAX_PLAYERS));
  #if USE_EMBEDDED_WEIGHTS
    Serial.println("Embedded weights: ENABLED");
  #else
    Serial.println("Embedded weights: DISABLED");
  #endif
  Serial.println("=================================");
  print_help();

  // Setup WiFi AP
  setup_wifi_ap();

  // Initialize PSRAM
  #if defined(CONFIG_ESP32_SPIRAM_SUPPORT) || defined(CONFIG_ESP32S2_SPIRAM_SUPPORT)
    if(psramInit()) {
      Serial.println("PSRAM initialized");
    } else {
      Serial.println("PSRAM failed! Using RAM");
    }
  #endif

  // Allocate PSRAM memory
  allocate_psram_memory();

  // Mount filesystem
  if (!FILESYSTEM.begin(true)) {
    Serial.println("FILESYSTEM mount failed");
  } else {
    Serial.println("FILESYSTEM mounted");
  }

  // Load weights
  Serial.println("\n>>> Loading AI weights...");
  // Try filesystem first (trained weights persist across resets)
  if(load_weights_from_flash()) {
    Serial.println("✓ Weights loaded from flash (trained)");
  }
  #if USE_EMBEDDED_WEIGHTS
    else if(load_embedded_weights()) {
      Serial.println("✓ Using embedded weights from ai_weights.h");
    }
    else {
      Serial.println("⚠ No weights found, initializing random");
      initialize_neural_network();
    }
  #else
    else {
      Serial.println("⚠ No weights found, initializing random");
      initialize_neural_network();
    }
  #endif

  // Seed random
  randomSeed(analogRead(0));

  // Setup web server
  setup_web_server();

  // Start serial game
  initialize_game(false);
}

void loop() {
  // Check player timeouts
  uint32_t now = millis();
  for(int i = 0; i < MAX_PLAYERS; i++) {
    if(players[i].active && (now - players[i].last_seen > PLAYER_TIMEOUT_MS)) {
      Serial.println("Player " + String(i) + " timed out");
      players[i].active = 0;
    }
  }

  // Serial commands
  if(Serial.available() > 0) {
    if(game_over || player_turn) {
      process_serial_command();
      return;
    } else {
      while(Serial.available()) Serial.read();
    }
  }

  // Fast training mode
  if(fast_training_mode) {
    fast_train_session();
    return;
  }
  
  // Self-play training mode (AI vs AI)
  if(selfplay_training_mode) {
    fast_train_self_play_session();
    return;
  }

  // Normal game
  if(!game_over) {
    if(player_turn) {
      // Wait for serial input
    } else {
      ai_turn_logic();
    }
  }

  delay(50);
}

// ============ MEMORY ALLOCATION ============
void allocate_psram_memory() {
  // Allocate players array in PSRAM
  #if defined(CONFIG_ESP32_SPIRAM_SUPPORT) || defined(CONFIG_ESP32S2_SPIRAM_SUPPORT)
    if(psramFound()) {
      players = (PlayerGame*)ps_malloc(sizeof(PlayerGame) * MAX_PLAYERS);
      Serial.println("Player data in PSRAM");
    } else {
      players = (PlayerGame*)malloc(sizeof(PlayerGame) * MAX_PLAYERS);
      Serial.println("Player data in RAM (PSRAM not available)");
    }
  #else
    players = (PlayerGame*)malloc(sizeof(PlayerGame) * MAX_PLAYERS);
    Serial.println("Player data in RAM");
  #endif
  
  // Initialize players array
  for(int i = 0; i < MAX_PLAYERS; i++) {
    players[i].active = 0;
    players[i].ip = 0;
    players[i].player_sym = PLAYER_X;
    for(int j = 0; j < BOARD_SIZE; j++) players[i].board[j] = 0;
    players[i].last_seen = 0;
  }
  
  // Allocate neural network in PSRAM
  #if defined(CONFIG_ESP32_SPIRAM_SUPPORT) || defined(CONFIG_ESP32S2_SPIRAM_SUPPORT)
    if(psramFound()) {
      input_weights = (float*)ps_malloc(sizeof(float) * INPUT_SIZE * HIDDEN_SIZE);
      hidden_weights = (float*)ps_malloc(sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE);
      input_bias = (float*)ps_malloc(sizeof(float) * HIDDEN_SIZE);
      hidden_bias = (float*)ps_malloc(sizeof(float) * OUTPUT_SIZE);
      input_vector = (float*)ps_malloc(sizeof(float) * INPUT_SIZE);
      hidden_output = (float*)ps_malloc(sizeof(float) * HIDDEN_SIZE);
      output_vector = (float*)ps_malloc(sizeof(float) * OUTPUT_SIZE);
      game_history = (GameMove*)ps_malloc(sizeof(GameMove) * MAX_GAME_MOVES);
      Serial.println("Neural network in PSRAM");
      return;
    }
  #endif
  input_weights = (float*)malloc(sizeof(float) * INPUT_SIZE * HIDDEN_SIZE);
  hidden_weights = (float*)malloc(sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE);
  input_bias = (float*)malloc(sizeof(float) * HIDDEN_SIZE);
  hidden_bias = (float*)malloc(sizeof(float) * OUTPUT_SIZE);
  input_vector = (float*)malloc(sizeof(float) * INPUT_SIZE);
  hidden_output = (float*)malloc(sizeof(float) * HIDDEN_SIZE);
  output_vector = (float*)malloc(sizeof(float) * OUTPUT_SIZE);
  game_history = (GameMove*)malloc(sizeof(GameMove) * MAX_GAME_MOVES);
  Serial.println("Neural network in RAM");
}

// ============ EMBEDDED WEIGHTS LOADER ============
#if USE_EMBEDDED_WEIGHTS
bool load_embedded_weights() {
  Serial.println("Loading embedded weights from flash...");
  
  int idx = 0;
  int total = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE;
  
  // Verify array size
  if(sizeof(EMBEDDED_WEIGHTS) / sizeof(float) != total) {
    Serial.println("⚠ Weight array size mismatch!");
    Serial.println("  Expected: " + String(total));
    Serial.println("  Got: " + String(sizeof(EMBEDDED_WEIGHTS) / sizeof(float)));
    return false;
  }
  
  // Load input weights (486 values)
  for(int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
    input_weights[i] = pgm_read_float(&EMBEDDED_WEIGHTS[idx++]);
  }
  
  // Load hidden weights (162 values)
  for(int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
    hidden_weights[i] = pgm_read_float(&EMBEDDED_WEIGHTS[idx++]);
  }
  
  // Load input bias (18 values)
  for(int i = 0; i < HIDDEN_SIZE; i++) {
    input_bias[i] = pgm_read_float(&EMBEDDED_WEIGHTS[idx++]);
  }
  
  // Load hidden bias (9 values)
  for(int i = 0; i < OUTPUT_SIZE; i++) {
    hidden_bias[i] = pgm_read_float(&EMBEDDED_WEIGHTS[idx++]);
  }
  
  Serial.println("✓ Embedded weights loaded: " + String(total) + " values");
  return true;
}
#endif

// ============ PLAYER SLOT MANAGEMENT ============
int find_player_slot(uint32_t ip) {
  for(int i = 0; i < MAX_PLAYERS; i++) {
    if(players[i].active && players[i].ip == ip) {
      return i;
    }
  }
  for(int i = 0; i < MAX_PLAYERS; i++) {
    if(!players[i].active) {
      players[i].ip = ip;
      players[i].active = 1;
      players[i].player_sym = PLAYER_X;
      for(int j = 0; j < BOARD_SIZE; j++) players[i].board[j] = 0;
      players[i].last_seen = millis();
      Serial.println("New player " + String(i) + " from " + String(ip));
      return i;
    }
  }
  return -1;
}

void update_player_timeout(uint32_t ip) {
  for(int i = 0; i < MAX_PLAYERS; i++) {
    if(players[i].active && players[i].ip == ip) {
      players[i].last_seen = millis();
      break;
    }
  }
}

// ============ WEB SERVER SETUP ============
void setup_web_server() {
  server.on("/", HTTP_GET, handle_web_root);
  server.on("/m", HTTP_GET, handle_web_move);
  server.on("/ng", HTTP_GET, handle_web_newgame);
  server.on("/st", HTTP_GET, handle_web_status);
  server.on("/sy", HTTP_GET, handle_web_choose_symbol);

  server.begin();
  Serial.println("Web server started (async, multi-user)");
}

// ============ WEB HANDLERS ============
void handle_web_root(AsyncWebServerRequest *request) {
  uint32_t ip = request->client()->remoteIP();
  int slot = find_player_slot(ip);

  if(slot == -1) {
    request->send(503, "text/plain", "Server full - max " + String(MAX_PLAYERS) + " players");
    return;
  }

  update_player_timeout(ip);
  PlayerGame* p = &players[slot];

  char boardHtml[600];
  boardHtml[0] = '\0';

  char cellBuf[80];
  for(int i = 0; i < BOARD_SIZE; i++) {
    char cell = '-';
    const char* cellClass = "cell";

    if(p->board[i] == PLAYER_X) { cell = 'X'; cellClass = "cell X"; }
    else if(p->board[i] == AI_O) { cell = 'O'; cellClass = "cell O"; }

    bool clickable = (p->active == 2) && (p->player_turn == 1);

    if(clickable) {
      snprintf(cellBuf, sizeof(cellBuf), "<div class=\"%s\" onclick=\"m(%d)\">%c</div>", cellClass, i, cell);
    } else {
      snprintf(cellBuf, sizeof(cellBuf), "<div class=\"%s\">%c</div>", cellClass, cell);
    }
    strcat(boardHtml, cellBuf);
  }

  String html = String(HTML_PAGE);
  html.replace("%BD%", String(boardHtml));
  html.replace("%AW%", String(ai_wins));
  html.replace("%PW%", String(player_wins));
  html.replace("%DR%", String(games_played - ai_wins - player_wins));
  html.replace("%RT%", games_played > 0 ? String((ai_wins * 100) / games_played) + "%" : "0%");

  const char* result = "";
  if(p->active != 2) result = "Press New Game!";
  else if(p->player_turn == 1) result = "Your turn!";
  else result = "AI thinking...";
  html.replace("%RS%", result);

  html.replace("%XS%", p->player_sym == PLAYER_X ? "sym-s" : "");
  html.replace("%OS%", p->player_sym == AI_O ? "sym-s" : "");

  request->send(200, "text/html", html);
}

void handle_web_move(AsyncWebServerRequest *request) {
  uint32_t ip = request->client()->remoteIP();
  update_player_timeout(ip);

  int slot = find_player_slot(ip);
  if(slot == -1 || players[slot].active != 2) {
    request->send(400, "text/plain", "Game not active");
    return;
  }

  if(!players[slot].player_turn) {
    request->send(400, "text/plain", "Not your turn");
    return;
  }

  if(request->hasParam("p")) {
    int pos = request->getParam("p")->value().toInt();
    int ai_sym = (players[slot].player_sym == PLAYER_X) ? AI_O : PLAYER_X;

    if(pos >= 0 && pos <= 8 && players[slot].board[pos] == 0) {
      players[slot].board[pos] = players[slot].player_sym;
      players[slot].player_turn = 0;  // AI's turn now

      int w = check_winner_player(players[slot].board);

      if(w != 0) {
        players[slot].active = 1;
        if(w == players[slot].player_sym) {
          player_wins++;
          Serial.println("Player " + String(slot) + " won!");
        } else if(w == -1) {
          Serial.println("Player " + String(slot) + " draw");
        } else {
          ai_wins++;
          Serial.println("AI won vs player " + String(slot));
        }
        games_played++;
      } else {
        // AI makes its move
        int ai_move = ai_select_move_player(players[slot].board, ai_sym, players[slot].player_sym);
        if(ai_move != -1) {
          players[slot].board[ai_move] = ai_sym;

          w = check_winner_player(players[slot].board);
          if(w != 0) {
            players[slot].active = 1;
            if(w == ai_sym) ai_wins++;
            else if(w == -1) { }
            else player_wins++;
            games_played++;
          } else {
            players[slot].player_turn = 1;  // Player's turn again
          }
        }
      }
    }
  }
  request->send(200, "text/plain", "OK");
}

void handle_web_newgame(AsyncWebServerRequest *request) {
  uint32_t ip = request->client()->remoteIP();
  update_player_timeout(ip);

  int slot = find_player_slot(ip);
  if(slot == -1) {
    request->send(503, "text/plain", "Server full");
    return;
  }

  for(int i = 0; i < BOARD_SIZE; i++) players[slot].board[i] = 0;
  players[slot].active = 2;
  players[slot].player_turn = (players[slot].player_sym == PLAYER_X);

  // If player chose O, AI (X) goes first
  if(players[slot].player_sym == AI_O) {
    int ai_move = ai_select_move_player(players[slot].board, PLAYER_X, AI_O);
    if(ai_move != -1) {
      players[slot].board[ai_move] = PLAYER_X;
      players[slot].player_turn = 1;  // Now player's turn
    }
  }

  request->send(200, "text/plain", "OK");
}

void handle_web_status(AsyncWebServerRequest *request) {
  update_player_timeout(request->client()->remoteIP());
  
  char status[100];
  snprintf(status, sizeof(status), "Games:%d AI:%d Player:%d Draws:%d",
           games_played, ai_wins, player_wins, games_played - ai_wins - player_wins);
  request->send(200, "text/plain", status);
}

void handle_web_choose_symbol(AsyncWebServerRequest *request) {
  uint32_t ip = request->client()->remoteIP();
  update_player_timeout(ip);

  int slot = find_player_slot(ip);
  if(slot == -1) {
    request->send(503, "text/plain", "Server full");
    return;
  }

  if(request->hasParam("s")) {
    String sym = request->getParam("s")->value();
    if(sym == "X") {
      players[slot].player_sym = PLAYER_X;
      players[slot].player_turn = 1;  // Player (X) goes first
    } else if(sym == "O") {
      players[slot].player_sym = AI_O;
      players[slot].player_turn = 0;  // AI (X) goes first, not player
    }
    for(int i = 0; i < BOARD_SIZE; i++) players[slot].board[i] = 0;
    players[slot].active = 1;
  }
  request->send(200, "text/plain", "OK");
}

// ============ GAME LOGIC ============
int check_winner_player(uint8_t* pboard) {
  int wins[8][3] = {{0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};
  
  for(int i = 0; i < 8; i++) {
    if(pboard[wins[i][0]] && pboard[wins[i][0]] == pboard[wins[i][1]] && 
       pboard[wins[i][0]] == pboard[wins[i][2]]) {
      return pboard[wins[i][0]];
    }
  }
  
  for(int i = 0; i < BOARD_SIZE; i++) {
    if(!pboard[i]) return 0;
  }
  return -1;
}

int ai_select_move_player(uint8_t* pboard, int ai_sym, int player_sym) {
  for(int i = 0; i < BOARD_SIZE; i++) {
    if(pboard[i] == 0) {
      input_vector[i*3] = 1.0f;
      input_vector[i*3+1] = 0.0f;
      input_vector[i*3+2] = 0.0f;
    } else if(pboard[i] == player_sym) {
      input_vector[i*3] = 0.0f;
      input_vector[i*3+1] = 1.0f;
      input_vector[i*3+2] = 0.0f;
    } else {
      input_vector[i*3] = 0.0f;
      input_vector[i*3+1] = 0.0f;
      input_vector[i*3+2] = 1.0f;
    }
  }
  
  run_neural_network();
  
  int best_move = -1;
  float best_score = -1.0f;
  
  for(int i = 0; i < BOARD_SIZE; i++) {
    if(pboard[i] == 0 && output_vector[i] > best_score) {
      best_score = output_vector[i];
      best_move = i;
    }
  }
  
  return best_move;
}

// ============ NEURAL NETWORK FUNCTIONS ============
void initialize_neural_network() {
  for(int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
    input_weights[i] = (float)(random(200) - 100) / 1000.0f;
  }
  for(int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
    hidden_weights[i] = (float)(random(200) - 100) / 1000.0f;
  }
  for(int i = 0; i < HIDDEN_SIZE; i++) input_bias[i] = 0.0f;
  for(int i = 0; i < OUTPUT_SIZE; i++) hidden_bias[i] = 0.0f;
  Serial.println("Neural network initialized");
}

bool load_weights_from_flash() {
  File file = FILESYSTEM.open("/tictactoe_weights.dat", "r");
  if(!file) return false;
  
  size_t r1 = file.read((uint8_t*)input_weights, sizeof(float) * INPUT_SIZE * HIDDEN_SIZE);
  size_t r2 = file.read((uint8_t*)hidden_weights, sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE);
  size_t r3 = file.read((uint8_t*)input_bias, sizeof(float) * HIDDEN_SIZE);
  size_t r4 = file.read((uint8_t*)hidden_bias, sizeof(float) * OUTPUT_SIZE);
  file.close();
  
  size_t total = r1 + r2 + r3 + r4;
  size_t expected = sizeof(float) * (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE);
  
  Serial.println("Loaded " + String(total) + "/" + String(expected) + " bytes");
  return total == expected;
}

bool save_weights_to_flash() {
  File file = FILESYSTEM.open("/tictactoe_weights.dat", "w");
  if(!file) return false;
  
  file.write((uint8_t*)input_weights, sizeof(float) * INPUT_SIZE * HIDDEN_SIZE);
  file.write((uint8_t*)hidden_weights, sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE);
  file.write((uint8_t*)input_bias, sizeof(float) * HIDDEN_SIZE);
  file.write((uint8_t*)hidden_bias, sizeof(float) * OUTPUT_SIZE);
  file.close();
  
  Serial.println("Weights saved to flash");
  return true;
}

float sigmoid(float x) {
  if(x > 10.0f) return 1.0f;
  if(x < -10.0f) return 0.0f;
  return 1.0f / (1.0f + exp(-x));
}

void run_neural_network() {
  for(int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = input_bias[i];
    for(int j = 0; j < INPUT_SIZE; j++) {
      sum += input_vector[j] * input_weights[j * HIDDEN_SIZE + i];
    }
    hidden_output[i] = sigmoid(sum);
  }
  
  for(int i = 0; i < OUTPUT_SIZE; i++) {
    float sum = hidden_bias[i];
    for(int j = 0; j < HIDDEN_SIZE; j++) {
      sum += hidden_output[j] * hidden_weights[j * OUTPUT_SIZE + i];
    }
    output_vector[i] = sigmoid(sum);
  }
}

void update_weights_td_learning(bool game_won) {
  float final_reward = game_won ? 1.0f : (winner == -1 ? 0.5f : 0.0f);
  
  int num_moves = 0;
  for(int i = 0; i < MAX_GAME_MOVES; i++) {
    if(game_history[i].move_made == -1) break;
    num_moves++;
  }
  
  for(int i = num_moves - 1; i >= 0; i--) {
    float target = (i == num_moves - 1) ? final_reward : game_history[i + 1].move_value_before;
    float td_error = target - game_history[i].move_value_before;
    update_network_for_move(game_history[i].board_state, game_history[i].move_made, td_error);
  }
  
  Serial.println("Network updated");
}

void update_network_for_move(float* board_state, int move_idx, float td_error) {
  for(int i = 0; i < INPUT_SIZE; i++) input_vector[i] = board_state[i];
  run_neural_network();
  
  for(int i = 0; i < HIDDEN_SIZE; i++) {
    float grad = LEARNING_RATE * td_error * hidden_output[i];
    hidden_weights[i * OUTPUT_SIZE + move_idx] += grad;
  }
  
  for(int i = 0; i < INPUT_SIZE; i++) {
    for(int j = 0; j < HIDDEN_SIZE; j++) {
      float out_err = hidden_weights[j * OUTPUT_SIZE + move_idx] * td_error;
      float grad = LEARNING_RATE * out_err * input_vector[i] * hidden_output[j] * (1.0f - hidden_output[j]);
      input_weights[i * HIDDEN_SIZE + j] += grad;
    }
  }
}

void record_ai_move(int move) {
  int slot = 0;
  for(; slot < MAX_GAME_MOVES; slot++) {
    if(game_history[slot].move_made == -1) break;
  }
  
  if(slot < MAX_GAME_MOVES) {
    board_to_input_vector();
    for(int i = 0; i < INPUT_SIZE; i++) {
      game_history[slot].board_state[i] = input_vector[i];
    }
    game_history[slot].move_made = move;
    run_neural_network();
    game_history[slot].move_value_before = output_vector[move];
  }
}

void board_to_input_vector() {
  for(int i = 0; i < BOARD_SIZE; i++) {
    if(board[i] == EMPTY) {
      input_vector[i*3] = 1.0f; input_vector[i*3+1] = 0.0f; input_vector[i*3+2] = 0.0f;
    } else if(board[i] == PLAYER_X) {
      input_vector[i*3] = 0.0f; input_vector[i*3+1] = 1.0f; input_vector[i*3+2] = 0.0f;
    } else {
      input_vector[i*3] = 0.0f; input_vector[i*3+1] = 0.0f; input_vector[i*3+2] = 1.0f;
    }
  }
}

void initialize_game(bool silent_mode) {
  for(int i = 0; i < BOARD_SIZE; i++) board[i] = EMPTY;
  for(int i = 0; i < MAX_GAME_MOVES; i++) game_history[i].move_made = -1;
  game_over = false;
  winner = EMPTY;
  player_turn = true;
  
  if(!silent_mode) {
    Serial.println("\nNew game started!");
    print_board();
  }
}

bool is_valid_move(int position) {
  return (position >= 0 && position < BOARD_SIZE && board[position] == EMPTY);
}

bool make_move(int position, int player) {
  if(is_valid_move(position)) {
    board[position] = player;
    return true;
  }
  return false;
}

void print_board() {
  Serial.println("\nCurrent Board:");
  for(int i = 0; i < BOARD_SIZE; i += 3) {
    Serial.print(" ");
    for(int j = 0; j < 3; j++) {
      switch(board[i+j]) {
        case EMPTY: Serial.print("-"); break;
        case PLAYER_X: Serial.print("X"); break;
        case AI_O: Serial.print("O"); break;
      }
      if(j < 2) Serial.print(" | ");
    }
    Serial.println();
    if(i < 6) Serial.println("-----------");
  }
}

int check_winner() {
  int wins[8][3] = {{0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};
  
  for(int i = 0; i < 8; i++) {
    if(board[wins[i][0]] != EMPTY && board[wins[i][0]] == board[wins[i][1]] && 
       board[wins[i][0]] == board[wins[i][2]]) {
      return board[wins[i][0]];
    }
  }
  
  for(int i = 0; i < BOARD_SIZE; i++) {
    if(board[i] == EMPTY) return 0;
  }
  return -1;
}

bool is_game_over() {
  int result = check_winner();
  if(result != EMPTY) {
    game_over = true;
    winner = result;
    return true;
  }
  return false;
}

void ai_turn_logic() {
  Serial.println("AI is thinking...");
  int ai_move = ai_select_move();
  
  if(ai_move != -1) {
    record_ai_move(ai_move);
    make_move(ai_move, AI_O);
    Serial.println("AI moved: " + String(ai_move));
    print_board();
    
    if(is_game_over()) {
      end_game();
    } else {
      player_turn = true;
    }
  } else {
    Serial.println("AI has no moves!");
    end_game();
  }
}

int ai_select_move() {
  board_to_input_vector();
  run_neural_network();
  
  int moves[BOARD_SIZE];
  int count = 0;
  for(int i = 0; i < BOARD_SIZE; i++) {
    if(board[i] == EMPTY) moves[count++] = i;
  }
  
  if(count == 0) return -1;
  
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

void end_game() {
  Serial.println("\nGame Over!");
  if(winner == PLAYER_X) {
    Serial.println("Player wins!");
    player_wins++;
  } else if(winner == AI_O) {
    Serial.println("AI wins!");
    ai_wins++;
  } else {
    Serial.println("Draw!");
  }
  
  if(!fast_training_mode) {
    update_weights_td_learning(winner == AI_O);
  }
  
  games_played++;
  
  if(games_played % SAVE_WEIGHTS_INTERVAL == 0) {
    Serial.println("Saving weights...");
    save_weights_to_flash();
    delay(1000);
  }
  
  Serial.println("Games: " + String(games_played));
  Serial.println("New game in 3 seconds...");
  delay(3000);
  initialize_game(false);
}

int random_move() {
  for(int attempt = 0; attempt < 5; attempt++) {
    int pos = random(BOARD_SIZE);
    if(is_valid_move(pos)) return pos;
  }
  
  int moves[BOARD_SIZE];
  int count = 0;
  for(int i = 0; i < BOARD_SIZE; i++) {
    if(is_valid_move(i)) moves[count++] = i;
  }
  
  return count > 0 ? moves[random(count)] : -1;
}

void fast_train_session() {
  static int train_count = 0;
  static bool ai_turn = false;
  static bool first = true;
  
  if(first) {
    initialize_game(true);
    train_count = 0;
    ai_turn = false;
    first = false;
    Serial.println("Training: Random(X) vs AI(O)");
    return;
  }
  
  if(game_over) {
    if(winner == PLAYER_X) {
      random_wins++;
      Serial.println("Game " + String(train_count + 1) + ": Random won");
    } else if(winner == AI_O) {
      ai_wins++;
      Serial.println("Game " + String(train_count + 1) + ": AI won");
    } else {
      Serial.println("Game " + String(train_count + 1) + ": Draw");
    }
    
    update_weights_td_learning(winner == AI_O);
    games_played++;
    train_count++;
    
    if(games_played % SAVE_WEIGHTS_INTERVAL == 0) {
      Serial.println("Saving weights...");
      save_weights_to_flash();
      delay(1000);
    }
    
    delay(FAST_TRAIN_DELAY);
    
    if(stop_training_requested) {
      Serial.println("\nTraining stopped at " + String(train_count) + " games");
      Serial.println("AI: " + String(ai_wins) + " Random: " + String(random_wins));
      save_weights_to_flash();
      fast_training_mode = false;
      first = true;
      stop_training_requested = false;
      initialize_game(false);
      return;
    }
    
    if(train_count >= fast_train_target) {
      Serial.println("\nTraining complete: " + String(fast_train_target) + " games");
      Serial.println("AI: " + String(ai_wins) + " Random: " + String(random_wins));
      if(games_played % SAVE_WEIGHTS_INTERVAL != 0) save_weights_to_flash();
      fast_training_mode = false;
      first = true;
      initialize_game(false);
      return;
    }
    
    initialize_game(true);
    ai_turn = false;
    return;
  }
  
  if(!ai_turn) {
    int move = random_move();
    if(move != -1) {
      make_move(move, PLAYER_X);
      if(!is_game_over()) ai_turn = true;
    }
  } else {
    int move = ai_select_move();
    if(move != -1) {
      record_ai_move(move);
      make_move(move, AI_O);
    }
    ai_turn = false;
  }
}

// ============ SELF-PLAY TRAINING (AI vs AI) ============
// Two separate history buffers for X and O perspectives
struct GameMove selfplay_history_x[MAX_GAME_MOVES];
struct GameMove selfplay_history_o[MAX_GAME_MOVES];

void record_selfplay_move(int move, int player_sym) {
  struct GameMove* history = (player_sym == PLAYER_X) ? selfplay_history_x : selfplay_history_o;
  
  int slot = 0;
  for(; slot < MAX_GAME_MOVES; slot++) {
    if(history[slot].move_made == -1) break;
  }

  if(slot < MAX_GAME_MOVES) {
    board_to_input_vector();
    for(int i = 0; i < INPUT_SIZE; i++) {
      history[slot].board_state[i] = input_vector[i];
    }
    history[slot].move_made = move;
    run_neural_network();
    history[slot].move_value_before = output_vector[move];
  }
}

void clear_selfplay_history() {
  for(int i = 0; i < MAX_GAME_MOVES; i++) {
    selfplay_history_x[i].move_made = -1;
    selfplay_history_o[i].move_made = -1;
  }
}

void update_selfplay_weights(int game_winner) {
  // Update weights from X's perspective
  float x_won = (game_winner == PLAYER_X) ? 1.0f : ((game_winner == -1) ? 0.25f : 0.0f);
  update_selfplay_network(selfplay_history_x, x_won);

  // Update weights from O's perspective
  float o_won = (game_winner == AI_O) ? 1.0f : ((game_winner == -1) ? 0.25f : 0.0f);
  update_selfplay_network(selfplay_history_o, o_won);
}

void update_selfplay_network(struct GameMove* history, float final_reward) {
  int num_moves = 0;
  for(int i = 0; i < MAX_GAME_MOVES; i++) {
    if(history[i].move_made == -1) break;
    num_moves++;
  }
  
  if(num_moves == 0) return;
  
  for(int i = num_moves - 1; i >= 0; i--) {
    float target = (i == num_moves - 1) ? final_reward : history[i + 1].move_value_before;
    float td_error = target - history[i].move_value_before;
    update_network_for_move(history[i].board_state, history[i].move_made, td_error);
  }
}

void fast_train_self_play_session() {
  if(selfplay_first_run) {
    initialize_game(true);
    clear_selfplay_history();
    selfplay_train_count = 0;
    selfplay_x_turn = true;
    selfplay_first_run = false;
    Serial.println("Self-Play Training: AI(X) vs AI(O)");
    return;
  }

  if(game_over) {
    if(winner == PLAYER_X) {
      selfplay_wins_x++;
      Serial.println("Game " + String(selfplay_train_count + 1) + ": X won");
    } else if(winner == AI_O) {
      selfplay_wins_o++;
      Serial.println("Game " + String(selfplay_train_count + 1) + ": O won");
    } else {
      selfplay_draws++;
      Serial.println("Game " + String(selfplay_train_count + 1) + ": Draw");
    }

    // Learn from both perspectives
    update_selfplay_weights(winner);
    selfplay_train_count++;

    if(selfplay_train_count % SAVE_WEIGHTS_INTERVAL == 0) {
      Serial.println("Saving weights...");
      save_weights_to_flash();
      delay(1000);
    }

    delay(FAST_TRAIN_DELAY);

    if(stop_training_requested) {
      Serial.println("\nTraining stopped at " + String(selfplay_train_count) + " games");
      Serial.println("X: " + String(selfplay_wins_x) + " O: " + String(selfplay_wins_o) + " Draw: " + String(selfplay_draws));
      save_weights_to_flash();
      selfplay_training_mode = false;
      selfplay_first_run = true;
      stop_training_requested = false;
      initialize_game(false);
      return;
    }

    if(selfplay_train_count >= fast_train_target) {
      Serial.println("\nSelf-Play Training complete: " + String(fast_train_target) + " games");
      Serial.println("X: " + String(selfplay_wins_x) + " O: " + String(selfplay_wins_o) + " Draw: " + String(selfplay_draws));
      if(selfplay_train_count % SAVE_WEIGHTS_INTERVAL != 0) save_weights_to_flash();
      selfplay_training_mode = false;
      selfplay_first_run = true;
      initialize_game(false);
      return;
    }

    initialize_game(true);
    clear_selfplay_history();
    selfplay_x_turn = true;
    return;
  }

  // Both players use AI
  int move = ai_select_move();
  if(move != -1) {
    int current_player = selfplay_x_turn ? PLAYER_X : AI_O;
    record_selfplay_move(move, current_player);
    make_move(move, current_player);

    // Check if game is over
    int result = check_winner();
    if(result != 0) {
      game_over = true;
      winner = result;
    } else {
      // Check for draw
      bool has_empty = false;
      for(int i = 0; i < BOARD_SIZE; i++) {
        if(board[i] == EMPTY) {
          has_empty = true;
          break;
        }
      }
      if(!has_empty) {
        game_over = true;
        winner = -1;
      }
    }
  } else {
    // No moves available - game over (draw)
    game_over = true;
    winner = -1;
  }

  selfplay_x_turn = !selfplay_x_turn;
}

void process_serial_command() {
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  if(cmd.startsWith("train") && cmd.indexOf("trainself") < 0) {
    int games = FAST_TRAIN_GAMES;
    String after = cmd.substring(5);
    after.trim();
    if(after.length() > 0) {
      int p = after.toInt();
      if(p > 0) games = p;
    }
    Serial.println("\n>>> Training " + String(games) + " games (AI vs Random)...\n");
    fast_train_target = games;
    fast_training_mode = true;
    selfplay_training_mode = false;
  }
  else if(cmd.startsWith("trainself")) {
    int games = FAST_TRAIN_GAMES;
    String after = cmd.substring(9);
    after.trim();
    if(after.length() > 0) {
      int p = after.toInt();
      if(p > 0) games = p;
    }
    Serial.println("\n>>> Self-Play Training " + String(games) + " games...\n");
    fast_train_target = games;
    selfplay_training_mode = true;
    fast_training_mode = false;
    // Reset self-play state for new training session (but keep cumulative stats)
    selfplay_first_run = true;
    selfplay_train_count = 0;
  }
  else if(cmd == "help") print_help();
  else if(cmd == "embed") {
    export_weights_as_header();
  }
  else if(cmd == "newgame") {
    Serial.println("\n>>> New game...");
    initialize_game(false);
  }
  else if(cmd == "status") {
    Serial.println("\n=== Status ===");
    Serial.println("Games: " + String(games_played));
    Serial.println("AI: " + String(ai_wins) + " Player: " + String(player_wins));
    Serial.println("Random: " + String(random_wins));
    Serial.println("SelfPlay X: " + String(selfplay_wins_x) + " O: " + String(selfplay_wins_o) + " Draw: " + String(selfplay_draws));
    if(games_played > 0) Serial.println("AI Rate: " + String((ai_wins * 100) / games_played) + "%");
  }
  else if(cmd == "stop") {
    if(fast_training_mode || selfplay_training_mode) stop_training();
    else Serial.println("Not training\n");
  }
  else if(cmd.length() > 0 && player_turn && !game_over) {
    int pos = cmd.toInt();
    if(pos >= 0 && pos <= 8) {
      if(make_move(pos, PLAYER_X)) {
        Serial.println("Player: " + String(pos));
        print_board();
        if(is_game_over()) {
          end_game();
        } else {
          player_turn = false;
        }
      } else {
        Serial.println("Taken!\n");
      }
    } else {
      Serial.println("Invalid (0-8)!\n");
    }
  }
}

void print_help() {
  Serial.println("\n=== Commands ===");
  Serial.println("train [N]     - Train N games AI vs Random (default 45)");
  Serial.println("trainself [N] - Train N games AI vs AI self-play (default 45)");
  Serial.println("embed         - Export weights as C header (ai_weights.h)");
  Serial.println("newgame       - Start new game");
  Serial.println("status        - Show stats");
  Serial.println("stop          - Stop training");
  Serial.println("help          - This message");
  Serial.println("0-8           - Make move\n");
}

void export_weights() {
  Serial.println("\n=== EXPORT ===");
  Serial.print("AIWEIGHTS:");
  for(int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
    Serial.print(input_weights[i], 6);
    Serial.print(",");
    if(i % 50 == 49) { delay(5); }
  }
  for(int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
    Serial.print(hidden_weights[i], 6);
    Serial.print(",");
    if(i % 50 == 49) { delay(5); }
  }
  for(int i = 0; i < HIDDEN_SIZE; i++) {
    Serial.print(input_bias[i], 6);
    Serial.print(",");
  }
  for(int i = 0; i < OUTPUT_SIZE; i++) {
    Serial.print(hidden_bias[i], 6);
    if(i < OUTPUT_SIZE - 1) Serial.print(",");
  }
  Serial.println("\n=== END ===\n");
}

void export_weights_as_header() {
  Serial.println("\n// ============================================");
  Serial.println("// ai_weights.h - Copy this to ai_weights.h file");
  Serial.println("// ============================================");
  Serial.println("// Generated: " + String(millis()) + "ms after boot");
  Serial.println("// Games played: " + String(games_played));
  Serial.println("// AI Wins: " + String(ai_wins));
  Serial.println();
  
  int total = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE;
  
  for(int i = 0; i < total; i++) {
    float val;
    int iw = INPUT_SIZE * HIDDEN_SIZE;
    int hw = HIDDEN_SIZE * OUTPUT_SIZE;
    
    if(i < iw) val = input_weights[i];
    else if(i < iw + hw) val = hidden_weights[i - iw];
    else if(i < iw + hw + HIDDEN_SIZE) val = input_bias[i - iw - hw];
    else val = hidden_bias[i - iw - hw - HIDDEN_SIZE];
    
    Serial.print(val, 6);
    if(i < total - 1) Serial.print(",");
    
    if((i + 1) % 10 == 0) Serial.println();
  }
  
  Serial.println();
  Serial.println("// ============================================");
  Serial.println("// End of weights (" + String(total) + " values)");
  Serial.println("// ============================================\n");
}

bool import_weights(const char* data) {
  String str = String(data);
  if(!str.startsWith("AIWEIGHTS:")) {
    Serial.println("Invalid format");
    return false;
  }
  str = str.substring(10);
  
  int total = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE;
  int count = 0;
  char buf[32];
  int bi = 0;
  
  for(int i = 0; i <= str.length(); i++) {
    char c = str.charAt(i);
    if(c == ',' || c == '\0' || c == '\n') {
      if(bi > 0) {
        buf[bi] = '\0';
        float v = atof(buf);
        
        int iw = INPUT_SIZE * HIDDEN_SIZE;
        int hw = HIDDEN_SIZE * OUTPUT_SIZE;
        
        if(count < iw) input_weights[count] = v;
        else if(count < iw + hw) hidden_weights[count - iw] = v;
        else if(count < iw + hw + HIDDEN_SIZE) input_bias[count - iw - hw] = v;
        else hidden_bias[count - iw - hw - HIDDEN_SIZE] = v;
        
        count++;
        bi = 0;
      }
    } else {
      if(bi < 31) buf[bi++] = c;
    }
  }
  
  if(count == total) {
    Serial.println("Imported " + String(count) + " values\n");
    return true;
  } else {
    Serial.println("Expected " + String(total) + " got " + String(count) + "\n");
    return false;
  }
}

void save_weights_to_file() {
  File f = FILESYSTEM.open("/ai_weights.txt", "w");
  if(!f) {
    Serial.println("Create failed\n");
    return;
  }
  f.println("AIWEIGHTS:");
  for(int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
    f.print(input_weights[i], 6);
    f.print(",");
  }
  for(int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
    f.print(hidden_weights[i], 6);
    f.print(",");
  }
  for(int i = 0; i < HIDDEN_SIZE; i++) {
    f.print(input_bias[i], 6);
    f.print(",");
  }
  for(int i = 0; i < OUTPUT_SIZE; i++) {
    f.print(hidden_bias[i], 6);
    if(i < OUTPUT_SIZE - 1) f.print(",");
  }
  f.println();
  f.close();
  Serial.println("Saved to /ai_weights.txt\n");
}

void load_weights_from_file() {
  File f = FILESYSTEM.open("/ai_weights.txt", "r");
  if(!f) {
    Serial.println("Not found\n");
    return;
  }
  Serial.println("Loading from /ai_weights.txt...");
  
  char buf[32];
  int bi = 0;
  int count = 0;
  int total = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE;
  
  while(f.available() && count < total) {
    char c = f.read();
    if(c == ',' || c == '\n' || c == '\r') {
      if(bi > 0) {
        buf[bi] = '\0';
        float v = atof(buf);
        
        int iw = INPUT_SIZE * HIDDEN_SIZE;
        int hw = HIDDEN_SIZE * OUTPUT_SIZE;
        
        if(count < iw) input_weights[count] = v;
        else if(count < iw + hw) hidden_weights[count - iw] = v;
        else if(count < iw + hw + HIDDEN_SIZE) input_bias[count - iw - hw] = v;
        else hidden_bias[count - iw - hw - HIDDEN_SIZE] = v;
        
        count++;
        bi = 0;
      }
    } else if(c != ' ' && c != '\n' && c != '\r') {
      if(bi < 31) buf[bi++] = c;
    }
  }
  f.close();
  
  if(count == total) {
    Serial.println("Loaded " + String(count) + " values\n");
  } else {
    Serial.println("Loaded " + String(count) + "/" + String(total) + " values\n");
  }
}

void setup_wifi_ap() {
  Serial.println("\n=================================");
  Serial.println("Setting up AP...");
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  Serial.println("IP: " + WiFi.softAPIP().toString());
  Serial.println("SSID: " + String(AP_SSID));
  Serial.println("Pass: " + String(AP_PASSWORD));
  Serial.println("=================================\n");
}

void stop_training() {
  if(fast_training_mode || selfplay_training_mode) {
    stop_training_requested = true;
    Serial.println("Stopping training...");
  }
}
