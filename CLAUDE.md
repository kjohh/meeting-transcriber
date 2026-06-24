# Claude Guidelines — Meeting Transcriber

個人用會議逐字稿工具。使用者介紹 / 安裝在 `README.md`。本檔聚焦 AI 進入此 repo 該知道的架構決策、設計理由、踩過的坑。

## ⚠️ 發佈規則（Kyle 2026-06-23 lock，絕不違反）

**沒有 Kyle 當下明確說「發佈 / 發版 / release / 發 vX.Y.Z」,絕對不要跑 `scripts/release.sh` 或 `gh release create`。** 發到 GitHub Releases = 使用者會下載的對外動作,一律等他明說。

- 修 bug / 做功能 / `build-app.sh` build 進 `dist/` 給他本機測 → 可以做。
- commit / push 到 branch → 修正流程的一部分可做,但別過度。
- **release(發新版本)→ 一定等 Kyle 當下明確指令,不要「修好順手發版」。**

（2026-06-23 因連續自作主張發 v0.1.9 / v0.1.10 被糾正。Kyle 只在 v0.1.8 說過「推上去發布」,後面兩版我擅自發了。）

## 檔案佈局

```
app.py                                     ← Flask + pywebview entry，所有後端邏輯
static/index.html                          ← 前端單檔（含 CSS / JS）
setup.py                                   ← py2app 打包設定
requirements.txt                           ← Python deps
native/Sources/coreaudio_tap/main.swift    ← ScreenCaptureKit 系統音擷取
native/.build/release/coreaudio_tap        ← 編譯後的 binary（不進 git，但 py2app 會 bundle 進 .app）
scripts/build-app.sh                       ← 完整 build：icon → py2app → deep codesign → TCC reset → onboarding flag reset
scripts/build-icon.sh                      ← 1024×1024 PNG → icon.icns (用 sips + iconutil)
scripts/release.sh                         ← ditto-zip + gh release create 上傳到 GitHub Releases
assets/icon.png                            ← 1024×1024 source icon
.config.json                               ← API key + backend + onboarding flag（git-ignored，source mode 在 project root；bundle mode 在 ~/Library/Application Support/Meeting Transcriber/）
.vocab.local                               ← 自訂詞彙（git-ignored，同上規則）
.vocab.local.example                       ← 詞彙範本
dist/Meeting Transcriber.app               ← py2app 產出（git-ignored）
icon.icns                                  ← build-icon.sh 產出（git-ignored）
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

macOS 系統 Python 的 Tk (8.5) 在 dark mode 下渲染全黑，故走 Flask + pywebview。Flask 在 daemon thread 啟動，pywebview 用 WKWebView 打開 `http://localhost:8765`，視窗關閉 = process 結束。pywebview 也提供 JS API bridge 讓前端能呼叫 native macOS API（screen capture trigger / open settings / save file dialog）。

### Source mode vs bundle mode 的路徑分流

`_is_frozen_bundle()` 判斷是否在 py2app bundle 內跑（`sys.frozen` 或 `RESOURCEPATH` env）。
- `_resource_dir()` → bundle: `Contents/Resources/`，source: project root。用來找 Swift binary 跟 static/。
- `_user_data_dir()` → bundle: `~/Library/Application Support/Meeting Transcriber/`，source: project root。用來寫 config / vocab（bundle 內部 read-only）。

### 為什麼用 ScreenCaptureKit 而非 BlackHole

ScreenCaptureKit 是 macOS 13+ 原生 API，不需安裝虛擬音訊裝置。Swift binary 把 float32 PCM 16kHz mono 從 stdout 噴出，Python 用 subprocess 讀。

### 為什麼是 silence-aware chunking 而非固定秒數

原本是固定 10s 切片，但「我覺得這個 component 的 hover state 太 subtle 了」這種句子被切到中間，後半段會變成新句子轉錄，看起來不連貫。

借用 [lazy-take-notes](https://github.com/CJHwong/lazy-take-notes) 的 VAD heuristic：

- **`CHUNK_DURATION = 25.0`** — 滿 25s 強制切（避免長句無限累積）
- **`PAUSE_DURATION = 1.5`** — 尾部連續 1.5s 靜音才認定句末
- **`SILENCE_THRESHOLD = 0.005`** — 逐 frame voice-activity 閾值（低，讓輕聲講話的 frame 仍算 active）
- **`MIN_SPEECH = 2.0`** — buffer 至少 2s 語音才考慮 silence-trigger
- **`OVERLAP = 1.0`** — 僅「hard-cap」trigger 保留 overlap tail；silence-trigger 句子已結束，不保留（避免 stale audio 拖到下一輪觸發鬼影 chunk）。保留的 1s overlap 會被轉錄兩次，由 `_dedup_boundary` 在 append 前砍掉接縫重複（見下）

**`_chunk_worker` 把 snapshot + decision + consume 包在同一個 `_buf_lock` block 內** — 否則 `/pause` 觸發的 flush thread 跟 chunker 會 race 出 double transcribe。

### 為什麼有 `_BILINGUAL_PROMPT` 跟 prompt chain

Whisper 的 `prompt` 參數**不是 instruction，是 conditioning context**。中英夾雜場景：

1. 強制 `language="zh"`（中文 decoder 天然支援 latin token interleaving）
2. `_BILINGUAL_PROMPT` 提供典型中英夾雜「範例句」prime decoder 模仿這種風格
3. Prompt chain 把上一段 transcript 最後 80 字元加進去，給跨 chunk 連續性

Prompt chain 太長會誘發 Whisper 進入 **repetition loop**（複製 prompt 內容當輸出）。所以 chain 只留 1 段、上限 80 字元；偵測到輸出有 repetition loop 時不更新 chain。

### `_trim_repetition` / `_is_repetition_loop` / `_dedup_boundary`

Whisper 經典 hallucination：對沒信心的 audio（靜音 / off-script / repetitive priming）會吐出同一句話 N 次。後處理：連續相同 sentence > 2 次截斷；≥ 3 次視為 loop 不更新 prompt chain。

`_dedup_boundary` 是 **cross-chunk** 去重（與 `_trim_repetition` 的 chunk 內去重不同）：找「上一行結尾」與「這段開頭」的最長字級重疊（≥5 字、限 60 字窗），砍掉重複前綴。專治 hard-cap 保留的 1s overlap 被轉兩次的接縫重複。同樣套用在 `/upload` 的整檔結果上。

### `_drop_hallucinations` 的語言鎖判斷（雙向對稱）

- **`language="zh"`**：整段無中文字時，**只丟**落在 `_EN_HALLUCINATION` 清單的 stock filler（"thank you"、"thanks for watching"…），不無條件丟英文 —— 真正的英文句子（"let me share my screen"）保留。
- **`language="en"`**：出現 CJK / 假名（`_CJK_KANA_PUNCT_RE`）= Whisper drift / 幻覺（強制英文不該有中文）。strip 掉那些字,留下真英文,整段都是 CJK 就丟。**這同時讓 prompt chain 保持乾淨,是斷「一個靜音幻覺 chunk 把整場後半雪球成中文亂碼」的關鍵**（長 session 經典失敗模式）。
- **`auto`**：兩邊都不動（使用者沒宣告語言,中英夾雜可能是真的）。

### Voice activity ratio gate

`_transcribe` 開頭有兩層 silence gate：(1) 整體 RMS < `TRANSCRIBE_MIN_RMS`（0.012）跳過；(2) 100ms frames 的 active 比例 < `VOICE_ACTIVITY_RATIO`（0.15，即 15%）跳過。第二層特別重要——silence-aware chunker 偶爾會被「1s 真語音 + 4s 靜音」騙過 RMS check，frame-level 抓得到。

### EN→ZH 即時翻譯（opt-in）

工具列「🌐 翻譯」toggle（預設關,存 config `translate`,可錄音中即時切 —— 純後處理不碰 ASR pipeline）。開啟後 `_transcribe` 在 append 前對轉出的文字呼叫 `_maybe_translate` → `_translate_text`（Groq chat,`TRANSLATE_MODEL`,system prompt = 會議口譯英→台灣繁中、保留專有名詞、已是中文則原樣）。逐字稿轉成左右兩欄（左原文/右譯文,`_lines` 升級成 `list[dict]` 存 `ts/text/tr/tag`）。

關鍵限制:翻譯**一律走 Groq**(需金鑰),與 ASR backend 無關 —— 本機 ASR + 雲端翻譯仍會把文字上傳 Groq。Whisper 內建 translate 只能 X→英文,做不到英→中,所以走 LLM。延遲 = ASR + 翻譯,逐句非逐字。`_format_line` 統一 save/download 的雙語輸出格式。

### Backend 模型自動選擇

`pick_local_model(language)`：`zh` / `zh-en` → `breeze-q8`（Breeze ASR 25 繁中強化）；`en` / `auto` → `large-v3-turbo-q8_0`。UI 上 user 只選 backend (cloud/local) + language。

### 模型 cache 路徑

`pywhispercpp.constants.MODELS_DIR` = `~/Library/Application Support/pywhispercpp/models/`。**與 lazy-take-notes 共用**。`model_local_path()` 會檢查多個可能路徑（lazy-take-notes 用 `whisper-cpp/`，pywhispercpp 直接 download 用 `hf/owner__repo/`）。

### 金鑰驗證

`POST /key` 收到金鑰時打 Groq `models.list()` 驗證；失敗回中文錯誤訊息，不存 invalid key。前端 `validateAndSaveKey()` 統一 onboarding 跟 Settings 兩處的 save+test 流程。

### Onboarding flow

四步 wizard：歡迎 → 螢幕＋麥克風授權（含 polling 偵測 + mic live waveform 測試）→ 選 backend（雲端有 inline key input）→ 完成。Flag 存 `.config.json` 的 `onboarding_completed`。Settings 內「重新看引導」可繞 flag 重跑。

權限偵測純查詢（不觸發系統 prompt）：
- 螢幕：`Quartz.CGPreflightScreenCaptureAccess()`
- 麥克風：`AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_('soun')`

Mic 偵測 fallback：若 `_micTestRunning`（sd.InputStream 已成功開），即使 AVFoundation 回 not-granted 也視為已授權（stream 開得起來表示授權真的有）。

## UI 狀態機

| 狀態 | Record | New | lang/backend | Upload/Vocab | Save |
|---|---|---|---|---|---|
| 待機（Idle） | ● 開始 | – | ✓ | ✓ | – |
| Active（recording, !paused）| ⏸ 暫停 | – | – | – | – |
| Paused | ▶ 繼續 | ✓ | – | – | ✓ |

`recording` global 在 Active + Paused 都是 `true`。`sessionInProgress = recording`，`activeRecording = recording && !paused`。整個 session 期間 lang/backend selector 鎖定 — 切換中途不會生效，所以乾脆禁用。

## 開發注意事項

### Python 環境

- **必須**使用 Homebrew Python 3.13：`/opt/homebrew/bin/python3.13`
- 系統 Python 3.9 無法裝 pyobjc-core（pywebview 依賴）
- `pip install` 必須加 `--user --break-system-packages`（brew Python 走 PEP 668 鎖外部安裝）

### Flask 設定

- Port 8765，`use_reloader=False`
- Threaded mode，daemon thread

### Build / 簽章流程

跑 `./scripts/build-app.sh` 一次搞定：
1. 從 `assets/icon.png` 生 `icon.icns`（若不存在）
2. `py2app` 打包到 `dist/Meeting Transcriber.app`
3. **`codesign --force --deep --sign -`** 整個 bundle ad-hoc sign（py2app 只簽 main wrapper，沒簽內嵌 coreaudio_tap → TCC 不 inherit 授權 → 每次 spawn child binary 都會重新要求授權，所以必須 deep sign）
4. **`tccutil reset ScreenCapture / Microphone com.kylehsia.meeting-transcriber`** 清掉舊 TCC 紀錄（每次 rebuild signature hash 都不同 → macOS 視為新 app → 舊授權對新 app 無效，但 System Settings UI 因為 bundle id 相同顯示舊 entry，誤導使用者）
5. 清 `~/Library/Application Support/Meeting Transcriber/.config.json` 的 `onboarding_completed` flag → 下次跑會跳 onboarding

### 全域 audio state

`_recording`, `_paused`, `_backend`, `_language`, `_sys_buf`, `_mic_buf`, `_prompt_chain`, `_chunk_worker_thread`, `_mic_test_stream` 都是 module global。多執行緒共用，buffer 寫入要持 `_buf_lock`。

### `_chunk_worker_thread` join 機制

`/stop` 必須 `join()` 等 chunk_worker 完整退出（含最後 flush transcribe），否則 `newSession`（前端） chain `stop → clear → UI reset` 會被 late `_append_line` race，殘留文字到下一個 session 的 transcript。

### Pause 會釋放系統音擷取

`/pause` 期間 `_sys_capture_supervisor` 會 terminate Swift proc 並 idle（不持有 ScreenCaptureKit tap，避免 replayd 持續燒 CPU），resume 時自動重 spawn。代價：resume 後系統音有 ~1s 重連空窗。`_read_sys_stdout` 在 `_paused` 時也會 return（迴圈條件含 `not _paused`），這正是觸發 supervisor 釋放的訊號。麥克風 stream 不釋放（很輕、resume 要立即可用）。

## 已知坑與限制

### macOS TCC 跟 py2app rebuild

每次 py2app rebuild 產生新 code signature hash → macOS TCC 視為「同 bundle id 但不同 app instance」→ 舊授權對新 build 無效。System Settings UI 因為 bundle id 顯示同一個 entry，導致使用者以為「已授權」但 CGPreflight 回 false。`build-app.sh` 用 `tccutil reset` 解決，但 user 每次 update 版本都要重新授權。Apple Developer ID 簽章 ($99/yr) 才能避免。

### Whisper hallucination

Mitigations 已寫進 code：silence trigger 不保留 overlap、voice activity gate（ratio + 絕對語音時長 `MIN_ACTIVE_SPEECH` 下限，短插話不被誤丟）、repetition trim + loop 偵測 + 隔離 prompt chain、cross-chunk `_dedup_boundary`、zh-lock 只丟 stock-filler 英文。殘留 case：環境噪音穩定大（持續打字 / 風扇）可能讓 silence 偵測失靈 → buffer 一直長到 25s hard cap。對 personal 用沒影響。

### macOS 螢幕錄製授權

- 第一次：橘色鎖 dialog → 系統設定授權
- 月度：藍色舉手 dialog → macOS 15+ 強制 monthly re-confirm，**無法完全免除**（除非有 Apple Developer entitlement）

### Groq 隱私

Services Agreement 明訂禁止用客戶 input 訓練。但音訊仍會經 Groq server 處理。敏感會議切 Local backend。

## 未來計畫（已知，別主動做）

- 整體 UI / branding rebrand — gradient / glow / motion，更 fancy 的視覺
- 換 fancy 名字
- Apple Developer 帳號 + Developer ID 簽章 + Notarization → 解決 TCC reset / 月度 prompt
- LLM summary / action items
- Speaker diarization
- 匯出 docx / markdown
- Chunk 參數 UI 化（CHUNK_DURATION / PAUSE_DURATION 等）
