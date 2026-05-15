# Meeting Transcriber

個人用的會議逐字稿工具。錄製系統音（遠端會議對方的聲音）＋麥克風，依「靜音偵測」或「上限秒數」自動切片送 Whisper 轉文字，即時顯示在原生視窗裡。

支援兩種轉錄後端，可隨時切換：

| Backend | 模型 | 速度（10s 音訊） | 隱私 | 需要 |
|---|---|---|---|---|
| ☁ Cloud (Groq) | `whisper-large-v3-turbo` | 0.5–1s | 音訊經 Groq server | Groq API key |
| 💻 Local (whisper.cpp) | `large-v3-turbo-q8_0` / `breeze-q8` | 1–3s（M-series Mac） | 完全本機 | 模型檔（首次自動下載） |

Local 模式下，中文 / 中英混合會自動使用 **Breeze ASR 25**（聯發科繁中強化版），辨識中英夾雜場景顯著優於通用 Whisper。

## 架構

```
python3 app.py
  ├── Flask server (port 8765)           ← web UI + API endpoints
  ├── pywebview (WKWebView)              ← 原生視窗包覆，非瀏覽器
  ├── native/coreaudio_tap (Swift)       ← ScreenCaptureKit 錄系統音
  ├── sounddevice                        ← 錄麥克風
  └── 轉錄後端（依 UI 切換）
       ├── ☁ Cloud:  groq Python SDK → Groq Whisper API
       └── 💻 Local: pywhispercpp → whisper.cpp + ggml model
```

系統音 + 麥克風混音後進 buffer，由 **silence-aware chunker** 決定何時切片：

- 滿 25 秒 → 強制切（避免無停頓的長句被無限累積）
- 或偵測到尾部 1.5 秒靜音 + 前面有語音 → 自然句末切片
- 切完保留 1 秒重疊給下一片（僅「滿 25 秒」case）作為 context

切完的 chunk 進「voice-activity ratio 閘門」過濾大部分是靜音的 chunk（避免 Whisper hallucination），然後送對應 backend。

## 安裝

```bash
# 1. Python 套件（執行需要 Homebrew Python 3.13）
/opt/homebrew/bin/python3.13 -m pip install --user --break-system-packages -r requirements.txt

# 2. 編譯 Swift 錄音 binary（需要 Xcode Command Line Tools）
cd native && swift build -c release && cd ..
```

第一次執行需在 **系統設定 → 隱私權與安全性 → 螢幕與系統聲音錄製** 授權給 `Meeting Transcriber`。

> macOS 15+ 會每月跳一次「直接取用螢幕內容」確認對話框，這是 Apple 強制的隱私機制，按「允許」即可，無法完全免除。

## 執行

```bash
python3 app.py
# 或雙擊 Meeting Transcriber.app
```

`.app` bundle 是 launcher.swift 編譯的 native wrapper，會自動找 Python、檢查套件、起 Flask。模型缺失時 UI 會跳下載對話框。

## 主要功能

- **中英混合辨識**：預設語言 `zh-en`，強制 `language="zh"` + bilingual prompt 範例 prime decoder，避免英文段被翻譯成中文或亂碼
- **Silence-aware chunking**：句子不會被死板的計時器切兩半
- **Repetition trim**：Whisper 經典的「同句重複 N 次」hallucination 會被自動截斷為最多 2 次
- **自訂詞彙表**：UI 內編輯 → `.vocab.local`（git-ignored）→ 每次 chunk 即時讀取，prime Whisper 認識專有名詞 / 品牌 / 縮寫
- **Backend 切換**：UI dropdown 一鍵切 Cloud / Local，敏感會議走本地
- **模型 cache 共用 lazy-take-notes**：本機若已裝過 [lazy-take-notes](https://github.com/CJHwong/lazy-take-notes)，模型不會重複下載
- **原生 Save 對話框**：透過 pywebview JS API bridge 呼叫 macOS save panel，而非瀏覽器下載

## UI 狀態機

| 狀態 | Record | New | lang/backend | Upload/Vocab | Save |
|---|---|---|---|---|---|
| Idle（剛開或 New 後）| ● Start | – | ✓ | ✓ | – |
| Active（錄音中）| ⏸ Pause | – | – | – | – |
| Paused（暫停）| ▶ Resume | ✓ | – | – | ✓ |

設計原則：錄音 session 期間語言 / backend 鎖定，避免「UI 改了但 in-flight session 不適用」的假切換。Pause 中可 Save 中途進度或按 New 結束 session。

## 技術細節

- **系統音擷取**：Swift binary 用 ScreenCaptureKit (macOS 13+)，不需要 BlackHole 等虛擬音訊裝置
- **前端即時更新**：Server-Sent Events (`/events`)
- **API key / backend / vocab 儲存**：`.config.json` + `.vocab.local`（皆 git-ignored，本機）
- **本地模型 cache**：`~/Library/Application Support/pywhispercpp/models/`（與 lazy-take-notes 共用）
- **Prompt chain**：上一段 transcript 最後 80 字元當下一段 conditioning（給 Whisper 上下文連續性）。偵測到 repetition loop 時不更新 chain，避免汙染下一段

## 未來可以做的事

- **語言選擇 UI**：目前 lang dropdown 已有 zh-en / zh / en / auto 四選，但「en + 自訂語言列表」未來可加更多
- **py2app 打包**：包成可分發 .app，不需要使用者有 Homebrew Python（檔案會 200–400MB+，因為要包含 whisper.cpp + model）
- **Chunk 參數可調**：CHUNK_DURATION / PAUSE_DURATION / SILENCE_THRESHOLD 目前是常數，可加 UI 設定
