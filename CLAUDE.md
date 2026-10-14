# Claude Guidelines — Meeting Transcriber

個人用會議逐字稿工具。使用者介紹 / 安裝在 `README.md`。本檔聚焦 AI 進入此 repo 該知道的架構決策、設計理由、踩過的坑。

## ⚠️ 發佈規則（Kyle 2026-06-23 lock，絕不違反）

**沒有 Kyle 當下明確說「發佈 / 發版 / release / 發 vX.Y.Z」,絕對不要跑 `scripts/release.sh` 或 `gh release create`。** 發到 GitHub Releases = 使用者會下載的對外動作,一律等他明說。

- 修 bug / 做功能 / `build-app.sh` build 進 `dist/` 給他本機測 → 可以做。
- commit / push 到 branch → 修正流程的一部分可做,但別過度。
- **不開 PR**：改完確定做好（build／測過）就直接 push 到 `main`（Kyle 2026-10-14）。
- **release(發新版本)→ 一定等 Kyle 當下明確指令,不要「修好順手發版」。**

（2026-06-23 因連續自作主張發 v0.1.9 / v0.1.10 被糾正。Kyle 只在 v0.1.8 說過「推上去發布」,後面兩版我擅自發了。）

## 檔案佈局

```
app.py                                     ← Flask + pywebview entry，所有後端邏輯
static/index.html                          ← 前端單檔（含 CSS / JS）
setup.py                                   ← py2app 打包設定
requirements.txt                           ← Python deps
native/Sources/coreaudio_tap/main.swift    ← Core Audio process tap 系統音擷取（+ --preflight / --request 權限查詢）
native/.build/release/coreaudio_tap        ← 編譯後的 binary（不進 git，但 py2app 會 bundle 進 .app）
scripts/build-app.sh                       ← 完整 build：icon → 內建模型（checksum）→ Swift → py2app → sign-and-package
scripts/sign-and-package.sh                ← 由內而外簽章 + hardened runtime；DEVELOPER_ID 時公證 + dmg
entitlements.plist / entitlements-dev.plist ← release 最小集（麥克風）/ ad-hoc 開發版多 disable-library-validation
THIRD_PARTY_NOTICES.md                     ← 第三方授權（App 內「設定 → 關於」顯示）
tests/test_pure.py                         ← 純函式 unittest：python3.13 -m unittest discover -s tests
build-cache/ggml-small-q5_1.bin            ← 內建基本模型（build 腳本下載 + SHA256 驗證，gitignored）
scripts/build-icon.sh                      ← 1024×1024 PNG → icon.icns (用 sips + iconutil)
scripts/release.sh                         ← 檢查 dmg 已公證 → 傳到 R2（下載用）→ gh release create（備份 + appcast）
assets/icon.png                            ← 1024×1024 source icon
~/Library/Logs/Meeting Transcriber/        ← app.log（rotating）+ whisper.log
.draft.json                                ← 逐字稿自動存檔（同 .config.json 的目錄規則）
.config.json                               ← backend + onboarding flag 等設定（Groq 金鑰在鑰匙圈，不在這裡）（git-ignored，source mode 在 project root；bundle mode 在 ~/Library/Application Support/Meeting Transcriber/）
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

macOS 系統 Python 的 Tk (8.5) 在 dark mode 下渲染全黑，故走 Flask + pywebview。Flask 在 daemon thread 啟動，pywebview 用 WKWebView 打開 `http://localhost:<隨機 port>/?t=<token>`，視窗關閉 = process 結束。pywebview 也提供 JS API bridge 讓前端能呼叫 native macOS API（權限查詢與請求 / 打開系統設定 / 存檔對話框 / 顯示記錄檔）。

### Source mode vs bundle mode 的路徑分流

`_is_frozen_bundle()` 判斷是否在 py2app bundle 內跑（`sys.frozen` 或 `RESOURCEPATH` env）。
- `_resource_dir()` → bundle: `Contents/Resources/`，source: project root。用來找 Swift binary 跟 static/。
- `_user_data_dir()` → bundle: `~/Library/Application Support/Meeting Transcriber/`，source: project root。用來寫 config / vocab（bundle 內部 read-only）。

### 系統音：Core Audio process tap（macOS 14.4+）

原本用 ScreenCaptureKit，要「螢幕錄製」權限、macOS 15 起每月重問。改成 `CATapDescription` + private aggregate device，權限是「僅系統音訊錄製」（`NSAudioCaptureUsageDescription`），不讀畫面。Swift binary 把 float32 PCM 16kHz mono 從 stdout 噴出，stderr 協定 `READY` / `ERROR:`。預設輸出裝置換掉（接耳機、藍牙斷線）時 helper 以 exit 2 結束，由 `_sys_capture_supervisor` 重新 spawn。

- 權限查詢走 TCC SPI（同 insidegui/AudioCap 做法），在 helper 內：`--preflight` 不跳視窗、`--request` 跳系統對話框。TCC 把 helper 歸屬到 responsible process（.app），所以從 terminal 直接跑 helper 會以 terminal 的身分查，測試要用 .app。
- 沒權限時 tap 不報錯，只是沒資料（aggregate device 不會 running），所以 `/start` 先 preflight，denied 就直接顯示警示。
- `stop()` 不能在 IO queue 上呼叫（`AudioDeviceStop` 會等 IOProc → deadlock），listener 用獨立 queue。

### 本機轉錄子程序

`LocalWhisperWorker` 用單一 process 的 `ProcessPoolExecutor`（spawn）。initializer 把 whisper.cpp 的 C 輸出導到 whisper.log、檢查 ggml magic 後載入模型；每段音訊是一個 task。逾時或子程序死掉就把 worker 標成失效，下次 `_ensure_local_worker` 重建。

### App 內更新（Sparkle）

Sparkle 2.10.0（`build-app.sh` 釘版本 + SHA256 下載到 build-cache）包進 `Contents/Frameworks`，拿掉 XPCServices（非 sandbox App 不需要）。簽章順序：所有 Mach-O → `Updater.app` → `Sparkle.framework/Versions/B` → Python.framework → App。

- Info.plist：`SUFeedURL` 指向 latest release 的 `appcast.xml`，`SUPublicEDKey` 讀 repo 根目錄的 `sparkle_public_key.txt`（公開金鑰，要 commit）。沒有金鑰時 App 不啟動 Sparkle，退回原本的 GitHub 新版橫幅；正式 build（有 `DEVELOPER_ID`）沒有金鑰會直接失敗。
- 私鑰只在發版那台的鑰匙圈：`scripts/sparkle-keys.sh` 建立（重跑會沿用既有的）。私鑰遺失 = 已安裝的版本再也無法 App 內更新，要另外備份（`generate_keys -x`）。
- `release.sh` 用 `sign_update` 簽 dmg、產生只含最新一版的 `appcast.xml`，跟 dmg 一起上傳。
- **dmg 從 Cloudflare R2 下載，不是 GitHub**（2026-10-14 實測台灣下載 GitHub release 0.06–0.12 MB/s，R2 約 8 MB/s）。bucket `kylehsia-downloads`、網域 `downloads.kylehsia.com`：`meeting-transcriber/<版本>/` 每版一份（immutable），`latest/` 給 README（快取 5 分鐘）。appcast 的 enclosure 指向 R2 的版本路徑，**appcast 本身留在 GitHub**——已安裝的 App 的 `SUFeedURL` 寫死在 GitHub，換掉舊版就找不到更新。R2 只留最近 5 版（免費額度 10GB），發版時刪掉第 6 新的。`sparkle:version` 對應 CFBundleVersion（= APP_VERSION）。
- `start_sparkle()` 在主執行緒啟動 `SPUStandardUpdaterController`；選單「檢查更新…」呼叫 `checkForUpdates:`。Sparkle 的介面用 App 的 zh_TW。
- 本機測試：`MT_SPARKLE_PUBLIC_KEY` / `MT_SPARKLE_FEED_URL` 覆寫金鑰與更新來源；用 `generate_keys --account <測試名>` 做拋棄式金鑰，測完從鑰匙圈刪掉。

### 本機 API 保護

Flask 綁隨機 port（`make_server(..., 0)`，不再 sleep 等啟動）。每次啟動產生 `SESSION_TOKEN`，視窗開 `/?t=<token>` 換成 HttpOnly + SameSite=Strict cookie，之後每個 request 都要帶。加上 Origin 檢查，網頁與本機其他程式都打不進來。log 會 redact token 與 `gsk_` 金鑰。debug 用瀏覽器開頁面時設 `MT_NO_TOKEN=1`。

### 金鑰：macOS 鑰匙圈

Groq 金鑰存 Keychain（generic password，service `Meeting Transcriber`），舊版 `.config.json` 裡的會自動搬過去並刪除。前端只拿 `has_key`。啟動時背景讀取（`preload_api_key`），因為 ad-hoc 重新 build 後簽章變了，讀鑰匙圈會跳「允許存取」視窗並卡住該 thread；Developer ID 簽章穩定就不會。

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

### 本機模型：內建小模型 + 背景下載

`ggml-small-q5_1.bin`（190 MB）打包在 `Contents/Resources/models/`，所以本機模式裝好就能錄。`resolve_local_model(language)`：想要的模型（`zh` → `load_zh_model()`，預設 breeze-q8；`en` / `auto` → large-v3-turbo-q8_0）已下載就用它，否則用內建小模型。每次 `/start` 重新決定，所以下載完成後從下一場會議生效，錄音中途不換。onboarding 完成、每次啟動、切到本機或切語言時，`auto_download_preferred_model` / `ensureWantedModel` 會在背景下載缺的模型。

下載（`download_model`）是自己寫的 HTTP Range 續傳（不用 huggingface_hub）：寫到 `<file>.part`，進度從磁碟上的位元組算，SHA256（HF 的 X-Linked-ETag）對了才改名就位，失敗重試 4 次。`model_local_path` 只做檔案大小檢查（完整 SHA256 只在下載完跑一次）。

### 模型 cache 路徑

`pywhispercpp.constants.MODELS_DIR` = `~/Library/Application Support/pywhispercpp/models/`。**與 lazy-take-notes 共用**。

### 上傳檔案

`decode_to_wav16k` 用 macOS 內建 `afconvert`（mp3/m4a/wav/aiff/flac/mp4/mov），失敗才找 ffmpeg（webm/ogg 等）。雲端且 ≤ 24 MB 直接傳原檔；其他情況切成約 10 分鐘一段（在最後 15 秒內最安靜的 100ms 切），逐段轉錄、帶上一段結尾當 prompt，時間戳是檔案內的位置（00:10:00）。

### 逐字稿自動存檔 / 關閉確認

每新增或編輯一行就原子寫入 `.draft.json`；使用者儲存後標成 saved。啟動時有未儲存草稿就詢問恢復或捨棄。錄音中關視窗：`_set_quit_guard` 切換 pywebview 的 `confirm_close`（它在關閉當下才讀），不要用 closing handler 開 dialog（會 deadlock）。

### 金鑰驗證

`POST /key` 收到金鑰時打 Groq `models.list()` 驗證；失敗回中文錯誤訊息，不存 invalid key。前端 `validateAndSaveKey()` 統一 onboarding 跟 Settings 兩處的 save+test 流程。

### Onboarding flow

四步 wizard：歡迎 → 系統音訊＋麥克風權限（含 polling 偵測 + mic live waveform 測試）→ 選 backend（雲端有 inline key input）→ 完成。Flag 存 `.config.json` 的 `onboarding_completed`。Settings 內「重新看引導」可繞 flag 重跑。

權限偵測純查詢（不觸發系統 prompt）：
- 系統音訊：`coreaudio_tap --preflight`（TCC SPI，在 helper 內查）
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

- 隨機 port（`MT_PORT` 可指定），threaded，daemon thread

### Build / 簽章流程

跑 `./scripts/build-app.sh`：icon → 內建模型（checksum）→ Swift helper → py2app → `sign-and-package.sh`。

- 簽章由內而外逐一簽每個 Mach-O（dylib、.so、`MacOS/python`、Swift helper）→ Python.framework → app，全部開 hardened runtime。不用 `--deep`（公證會擋沒 runtime / timestamp 的巢狀 binary）。
- `DEVELOPER_ID="Developer ID Application: …"` 時加 timestamp、用 `entitlements.plist`（只有麥克風）；再設 `NOTARY_PROFILE`（`xcrun notarytool store-credentials` 建的 profile）會公證 app + dmg 並 staple。
- 沒設時是 ad-hoc 開發版，用 `entitlements-dev.plist`：ad-hoc 沒有 Team ID，hardened runtime 會拒絕載入 Python.framework（"different Team IDs"），所以開發版多一個 disable-library-validation。開發版每次 build 簽章都變，build 腳本會 `tccutil reset AudioCapture / Microphone`。
- `RESET_ONBOARDING=1 ./scripts/build-app.sh` 才會清 onboarding flag；`RESET_TCC=1` 才會清權限（會連 /Applications 的正式版權限一起清，因為是同一個 Bundle ID）。
- Bundle ID（`setup.py` 的 `BUNDLE_ID`）第一個公證版發出後就不能改，改了使用者要全部重新授權。

### 全域 audio state

`_recording`, `_paused`, `_backend`, `_language`, `_sys_buf`, `_mic_buf`, `_prompt_chain`, `_chunk_worker_thread`, `_mic_test_stream` 都是 module global。多執行緒共用，buffer 寫入要持 `_buf_lock`。

### `_chunk_worker_thread` join 機制

`/stop` 必須 `join()` 等 chunk_worker 完整退出（含最後 flush transcribe），否則 `newSession`（前端） chain `stop → clear → UI reset` 會被 late `_append_line` race，殘留文字到下一個 session 的 transcript。

### Pause 會釋放系統音擷取

`/pause` 期間 `_sys_capture_supervisor` 會 terminate Swift proc 並 idle（不持有 audio tap），resume 時自動重 spawn。代價：resume 後系統音有 ~1s 重連空窗。`_read_sys_stdout` 在 `_paused` 時也會 return（迴圈條件含 `not _paused`），這正是觸發 supervisor 釋放的訊號。麥克風 stream 不釋放（很輕、resume 要立即可用）。

## 已知坑與限制

### macOS TCC 跟 ad-hoc 簽章

ad-hoc 簽章每次 build 都不同，TCC 與鑰匙圈都把它當新 app：權限要重給、讀金鑰會跳允許視窗。Developer ID 簽章後就穩定。App 內原本的「重新授權」modal 與 `tccutil reset` 補救已移除。

### Whisper hallucination

Mitigations 已寫進 code：silence trigger 不保留 overlap、voice activity gate（ratio + 絕對語音時長 `MIN_ACTIVE_SPEECH` 下限，短插話不被誤丟）、repetition trim + loop 偵測 + 隔離 prompt chain、cross-chunk `_dedup_boundary`、zh-lock 只丟 stock-filler 英文。殘留 case：環境噪音穩定大（持續打字 / 風扇）可能讓 silence 偵測失靈 → buffer 一直長到 25s hard cap。對 personal 用沒影響。

### Groq 隱私

Services Agreement 明訂禁止用客戶 input 訓練。但音訊仍會經 Groq server 處理。敏感會議切 Local backend。

## 未來計畫（已知，別主動做）

- 整體 UI / branding rebrand — gradient / glow / motion，更 fancy 的視覺
- 換 fancy 名字
- LLM summary / action items
- Speaker diarization
- 匯出 docx
- Chunk 參數 UI 化（CHUNK_DURATION / PAUSE_DURATION 等）
