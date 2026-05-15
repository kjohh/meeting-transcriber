# Claude Guidelines — Meeting Transcriber

個人用會議逐字稿工具。使用者介紹 / 安裝在 `README.md`。本檔聚焦 AI 進入此 repo 該知道的架構決策、設計理由、踩過的坑。

## 檔案佈局

```
app.py                                     ← Flask + pywebview entry，所有後端邏輯
static/index.html                          ← 前端單檔（含 CSS / JS）
launcher.swift                             ← .app 雙擊入口（會檢查 deps + 跑 app.py）
native/Sources/coreaudio_tap/main.swift    ← ScreenCaptureKit 系統音擷取
native/.build/release/coreaudio_tap        ← 編譯後的 binary（不進 git）
Meeting Transcriber.app/                   ← .app bundle，含 ad-hoc codesign
.config.json                               ← API key + backend 偏好（git-ignored）
.vocab.local                               ← 自訂詞彙表（git-ignored）
.vocab.local.example                       ← 詞彙範例
requirements.txt                           ← Python deps
```

## 架構速覽

```
sounddevice (mic) ─┐
                   ├──→ _mic_buf / _sys_buf ──→ silence-aware chunker ──→ _transcribe()
coreaudio_tap ─────┘                                                          │
(system audio)                                                                ├─→ _transcribe_cloud (Groq)
                                                                              └─→ _transcribe_local (pywhispercpp)
                                                                                            │
                                                            transcript line ←── trim repetition + dedup
                                                                              │
                                                                       _append_line → SSE → frontend
```

## 關鍵設計決策

### UI 為什麼是 Flask + pywebview，而非 tkinter

macOS 系統 Python 的 Tk (8.5) 在 dark mode 下渲染全黑，故改走 Flask + pywebview。Flask 在 daemon thread 啟動，pywebview 用 WKWebView 打開 `http://localhost:8765`，視窗關閉 = process 結束。

### 為什麼用 ScreenCaptureKit 而非 BlackHole

ScreenCaptureKit 是 macOS 13+ 原生 API，不需安裝虛擬音訊裝置。Swift binary 把 float32 PCM 16kHz mono 從 stdout 噴出，Python 用 subprocess 讀。

### 為什麼是 silence-aware chunking 而非固定秒數

原本是固定 10s 切片，但「我覺得這個 component 的 hover state 太 subtle 了」這種句子被切到中間，後半段會變成新句子轉錄，看起來不連貫。

借用 [lazy-take-notes](https://github.com/CJHwong/lazy-take-notes) 的 VAD heuristic：

- **`CHUNK_DURATION = 25.0`** — 滿 25s 強制切（避免長句無限累積）
- **`PAUSE_DURATION = 1.5`** — 尾部連續 1.5s 靜音才認定句末
- **`SILENCE_THRESHOLD = 0.01`** — RMS 閾值
- **`MIN_SPEECH = 2.0`** — buffer 至少 2s 語音才考慮 silence-trigger
- **`OVERLAP = 1.0`** — 僅「hard-cap」trigger 保留 overlap tail；silence-trigger 句子已結束，**不保留** overlap（避免 stale audio 拖到下一輪觸發鬼影 chunk）

### 為什麼有 `_BILINGUAL_PROMPT` 跟 prompt chain

Whisper 的 `prompt` 參數**不是 instruction，是 conditioning context**。中英夾雜場景：

1. 強制 `language="zh"`（中文 decoder 天然支援 latin token interleaving）
2. `_BILINGUAL_PROMPT` 提供典型中英夾雜「範例句」prime decoder 模仿這種風格
3. Prompt chain 把上一段 transcript 最後 80 字元加進去，給跨 chunk 連續性

Prompt chain 太長會誘發 Whisper 進入 **repetition loop**（複製 prompt 內容當輸出）。所以：

- chain 只留 1 段（不是多段）
- 上限 80 字元
- 偵測到輸出含 repetition loop → 該段**不**更新 chain，避免汙染下一輪

### `_trim_repetition` / `_is_repetition_loop`

Whisper 的經典 hallucination：對沒信心的 audio（靜音 / off-script / repetitive priming）會吐出同一句話 N 次。後處理：連續相同 sentence > 2 次 → 截斷；連續相同 sentence ≥ 3 次 → 視為 loop 不更新 prompt chain。

### Voice activity ratio gate

`_transcribe` 開頭有兩層 silence gate：

1. 整體 RMS < 0.01 → skip
2. 把 audio 切 100ms frames，計算「active 比例」（過 silence threshold 的 frames / 總 frames），< 25% → skip

第二層特別重要：silence-aware chunker 偶爾會被「1s 真語音 + 4s 靜音」騙過第一層 RMS check，但 voice activity ratio gate 抓得到。

### Backend 模型自動選擇

`pick_local_model(language)`：

- `zh` / `zh-en` → `breeze-q8`（聯發科 Breeze ASR 25，繁中強化）
- `en` / `auto` → `large-v3-turbo-q8_0`

UI 上 user 只選 backend (cloud/local) + language，model 是 derived。

### 模型 cache 路徑

`pywhispercpp.constants.MODELS_DIR` = `~/Library/Application Support/pywhispercpp/models/`。**與 lazy-take-notes 共用**：

```
hf/ggerganov__whisper.cpp/ggml-large-v3-turbo-q8_0.bin   (834 MB)
hf/alan314159__Breeze-ASR-25-whispercpp/ggml-model-q8_0.bin
breeze/ggml-model-q8_0.bin                               (1.5 GB)
```

`model_local_path()` 會檢查多個可能路徑（lazy-take-notes 用 `whisper-cpp/`，pywhispercpp 直接 download 用 `hf/owner__repo/`）。

## UI 狀態機

| 狀態 | Record | New | lang/backend | Upload/Vocab | Save |
|---|---|---|---|---|---|
| Idle | ● Start | – | ✓ | ✓ | – |
| Active（recording, !paused）| ⏸ Pause | – | – | – | – |
| Paused | ▶ Resume | ✓ | – | – | ✓ |

`recording` global 在 Active + Paused 都是 `true`。`sessionInProgress = recording`，`activeRecording = recording && !paused`。整個 session 期間 lang/backend selector 鎖定 — 切換中途不會生效，所以乾脆禁用。

## 開發注意事項

### Python 環境

- **必須**使用 Homebrew Python 3.13：`/opt/homebrew/bin/python3.13`
- 系統 Python 3.9 無法裝 pyobjc-core（pywebview 依賴）
- `pip install` 必須加 `--user --break-system-packages`（brew Python 走 PEP 668 鎖外部安裝）
- `launcher.swift` 第一次跑會自動處理；手動測 `python3 app.py` 也要符合上面條件

### Flask 設定

- Port 8765，`use_reloader=False`（避免 reloader 干擾背景執行緒）
- Threaded mode，daemon thread

### 全域 audio state

`_recording`, `_paused`, `_backend`, `_language`, `_sys_buf`, `_mic_buf`, `_prompt_chain` 等都是 module global。多執行緒共用，buffer 寫入要持 `_buf_lock`。

### Swift binary 重編

```bash
cd native && swift build -c release
```

改完 `launcher.swift`：

```bash
swiftc launcher.swift -o "Meeting Transcriber.app/Contents/MacOS/Meeting Transcriber"
xattr -cr "Meeting Transcriber.app"   # 清掉 xattr 不然 codesign 會抱怨
codesign --force --deep --sign - "Meeting Transcriber.app"
```

`codesign --force --deep --sign -` 是 ad-hoc sign，給 bundle 一個 stable hash 讓 macOS TCC 認得是「同一個 app」。否則每次重編都會被當新 app，要求重新授權螢幕錄製。

## 已知坑與限制

### Whisper hallucination

主要對應已寫進 code 的 mitigations：

- Silence trigger 不保留 overlap → 避免「真語音 tail + 後續靜音」被誤識為新 chunk
- Voice activity ratio gate → 整體 RMS 騙得過、frame-level 活動比例騙不過
- Repetition trim + loop 偵測 → 截掉 N 次重複 + 隔離 prompt chain

殘留 case：環境噪音穩定大（持續打字 / 風扇）可能讓 silence 偵測失靈 → buffer 一直長到 25s hard cap。對 personal 用沒影響。

### macOS 螢幕錄製授權

- 第一次：橘色鎖 dialog「想要錄製此電腦螢幕」→ 系統設定授權
- 之後每月：藍色舉手 dialog「正在要求略過系統私密視窗選擇器」→ macOS 15+ 強制 monthly re-confirm，**無法完全免除**（除非有 Apple Developer entitlement）
- 如果第一次授權後仍重複跳 → 檢查 codesign signature 是否還是 `linker-signed`（要 ad-hoc deep sign 才 stable）

### Groq 隱私

從 Groq Services Agreement：

> "Groq is not permitted to use Inputs or Outputs for training or fine-tuning any AI Model Services or other models, unless explicitly granted permission or instructed by Customer."

不分 free / paid tier，合約上禁止用客戶 input 訓練。Console 設定可開「zero data retention」(eligible customers)。但音訊仍會經 Groq server processing。敏感會議建議切 Local backend。

### Future annotations / 型別

`app.py` 開頭有 `from __future__ import annotations` + `Optional[X]`，跟 Python 3.13 相容，保留即可。

## 未來計畫（已知，別主動做）

- **py2app 打包**：分發給沒有 Homebrew Python 的人。目前 `.app` 只是 wrapper，仍需要本機 Python 環境
- **Chunk 參數 UI**：CHUNK_DURATION / PAUSE_DURATION 等做成 UI 設定
- **更多 model 選項**：medium / small 量化版給低階機器用
