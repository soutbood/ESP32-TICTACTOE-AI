THIS PROJECT WILL BE RE-WRITTEN WITHOUT LLM HELP
# 🎮 Tic Tac Toe AI for ESP32

[![ESP32](https://img.shields.io/badge/Platform-ESP32-blue)](https://www.espressif.com/en/products/socs/esp32)
[![Arduino](https://img.shields.io/badge/Framework-Arduino-00979D)](https://www.arduino.cc/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-orange)](LICENSE)

A neural network-powered Tic Tac Toe AI that runs entirely on ESP32 microcontrollers. Features a web-based interface, on-device training, and support for multiple concurrent players.

![Features](https://img.shields.io/badge/Players-1--4-orange) 
![Memory](https://img.shields.io/badge/PSRAM-~10KB-lightgrey)
![Weights](https://img.shields.io/badge/Weights-2,007-blue)

---

## ✨ Features

- 🧠 **Neural Network AI** - 27→54→9 feedforward network that learns from gameplay
- 🎯 **On-Device Training** - Temporal Difference (TD) learning runs directly on ESP32
- 🌐 **Web Interface** - Play from any device via WiFi (no app needed)
- 👥 **Multi-User** - Up to 4 concurrent players with separate game sessions
- 💾 **Persistent Learning** - Weights saved to flash every 500 games
- 🔄 **Self-Play Mode** - AI can train against itself
- 📱 **Responsive Design** - Works on phones, tablets, and desktops
- ⚡ **Fast Training** - ~45 games in seconds with minimal delay

---

## 🎯 Quick Start

### What You Need
- ESP32 board (ESP32-S2 or ESP32-S3 recommended with PSRAM)
- USB cable
- Arduino IDE installed

### 5-Minute Setup

1. **Install Libraries** (Arduino IDE → Sketch → Include Library → Manage Libraries)
   - Search and install: `ESPAsyncWebServer`
   - Search and install: `AsyncTCP`

2. **Open the Sketch**
   - Open `TicTacToe_AI_ESP32_MultiUser.ino` in Arduino IDE

3. **Upload**
   - Select your ESP32 board and COM port
   - Click Upload

4. **Play!**
   - Connect to WiFi: `TicTacToe_AI` (password: `12345678`)
   - Open browser: `192.168.4.1`
   - Start playing!

---

## 🎮 How to Play

### Web Interface (Recommended)

1. **Connect** to the ESP32 WiFi network
2. **Open** `192.168.4.1` in your browser
3. **Choose** your symbol (X goes first, O goes second)
4. **Click** "New Game" to start
5. **Tap** any cell to make your move

![Web Interface](https://via.placeholder.com/400x300/1a1a2e/00d9ff?text=Tic+Tac+Toe+Web+UI)

### Serial Monitor (Alternative)

1. Open Serial Monitor at **115200 baud**
2. Enter positions 0-8 to make moves:
   ```
     0 | 1 | 2
     ---------
     3 | 4 | 5
     ---------
     6 | 7 | 8
   ```

---

## 🧠 Neural Network Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Input     │     │   Hidden    │     │   Output    │
│   (27)      │────→│   (54)      │────→│   (9)       │
│  neurons    │     │  neurons    │     │  neurons    │
└─────────────┘     └─────────────┘     └─────────────┘
     │                   │                   │
  Board              Sigmoid            Move Scores
  State              Activation         (0-8 positions)
```

### Board Encoding
Each position uses one-hot encoding:
- `[1,0,0]` = Empty
- `[0,1,0]` = Player (X)
- `[0,0,1]` = AI (O)

### Weight Distribution
| Component | Count | Memory |
|-----------|-------|--------|
| Input weights | 1,458 | 5.8 KB |
| Hidden weights | 486 | 1.9 KB |
| Biases | 63 | 252 bytes |
| **Total** | **2,007** | **~8 KB** |

---

## ⚙️ Configuration

Key settings in `TicTacToe_AI_ESP32_MultiUser.ino`:

```cpp
// WiFi Settings
const char* AP_SSID = "TicTacToe_AI";
const char* AP_PASSWORD = "12345678";

// Training Settings
#define LEARNING_RATE 0.01f        // How fast AI learns
#define SAVE_WEIGHTS_INTERVAL 500  // Save every 500 games
#define FAST_TRAIN_GAMES 45        // Default training games

// Multi-User Settings
#define MAX_PLAYERS 4              // Max concurrent players
#define PLAYER_TIMEOUT_MS 60000    // 60 second idle timeout
```

---

## 📡 Serial Commands

| Command | Description |
|---------|-------------|
| `train [N]` | Train N games vs random opponent (default: 45) |
| `trainself [N]` | Train N games vs itself (default: 45) |
| `stop` | Stop training and save weights |
| `embed` | Export weights as C header file |
| `status` | Show game statistics |
| `help` | Show all commands |
| `0-8` | Make a move |

### Training Examples

```
# Train 100 games against random opponent
train 100

# Train 500 games of self-play
trainself 500

# Check statistics
status
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Forward pass time | ~2 ms |
| Training update | ~5 ms |
| Memory usage | ~10 KB PSRAM |
| Weight save time | ~100 ms |
| Training speed | ~100 games/minute |

---

## 🔧 Hardware Compatibility

| Board | PSRAM | Status |
|-------|-------|--------|
| ESP32 (original) | 4 MB | ✅ Works |
| ESP32-S2 | 2 MB | ✅ Works |
| ESP32-S3 | 8 MB | ✅ Best |
| ESP32-C3 | None | ⚠️ Limited |

---

## 📚 Documentation

| File | Description |
|------|-------------|
| [`FUNCTION_REFERENCE.md`](FUNCTION_REFERENCE.md) | Complete function documentation |
| [`NEURAL_NETWORK_MATH.md`](NEURAL_NETWORK_MATH.md) | Math behind the neural network |
| [`ADAPTING_TO_OTHER_GAMES.md`](ADAPTING_TO_OTHER_GAMES.md) | Guide for Chess, Checkers, etc. |
| [`TicTacToe_AI_ESP32_Documentation.md`](TicTacToe_AI_ESP32_Documentation.md) | Original documentation |

---

## 🚀 Advanced Usage

### Transfer Trained AI to Another ESP32

**Method 1: Export Weights**
```
1. Run: embed (in Serial Monitor)
2. Copy the output
3. Paste into ai_weights.h
4. Upload to new ESP32
```

**Method 2: File Transfer**
```
1. Run: savefile (creates /ai_weights.txt)
2. Download file from ESP32
3. Upload to new ESP32
4. Run: loadfile
```

### Training Tips

- **Start small**: 45-100 games for initial testing
- **Self-play**: Better for learning optimal strategy
- **Save often**: Weights auto-save every 500 games
- **Be patient**: 1000+ games for noticeable improvement

---

## 🛠️ Troubleshooting

### Web interface won't load
- Make sure you're connected to `TicTacToe_AI` WiFi
- Try clearing browser cache
- Check Serial Monitor for errors

### AI makes random moves
- Weights may not have loaded - check Serial output
- Run `train 10` to initialize weights

### Compilation fails
- Install ESP32 board package in Arduino IDE
- Install `ESPAsyncWebServer` and `AsyncTCP` libraries
- Enable PSRAM in board settings

---

## Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Share your trained weights
- Adapt for other games (see `ADAPTING_TO_OTHER_GAMES.md`)

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See the [LICENSE](LICENSE) file for the full license text.

### What This Means

- ✅ **You can** use, modify, and distribute this software
- ✅ **You can** use it for commercial purposes
- ⚠️ **You must** disclose your source code if you modify it and run it on a server
- ⚠️ **You must** license your modifications under AGPL-3.0
- ⚠️ **You must** provide source code to users who interact with the software over a network

The AGPL-3.0 is a strong copyleft license designed to ensure that modifications remain open source, even when the software is used on network servers.

---

## Acknowledgments

- ESPAsyncWebServer library
- AsyncTCP library
- Arduino ESP32 core
- Temporal Difference Learning research

---

<div align="center">

**Made with ❤️ for embedded AI enthusiasts**

[Report Bug](../../issues) · [Request Feature](../../issues)

</div>
