# Meeting Transcriber

個人用的會議逐字稿工具。錄製你的麥克風 + 電腦播放的聲音（線上會議對方的聲音），用 Whisper AI 即時轉成逐字稿。線上、實體、混合會議都能用。

支援兩種轉錄後端，可隨時切換：

| Backend | 模型 | 速度（10s 音訊） | 隱私 | 需要 |
|---|---|---|---|---|
| 🌐 Cloud (Groq) | `whisper-large-v3-turbo` | 0.5–1s | 音訊經 Groq server | 免費 Groq API key |
| 🔒 Local (whisper.cpp) | `large-v3-turbo-q8_0` / `breeze-q8` | 1–3s（Apple Silicon Mac） | 完全本機 | 模型檔（首次自動下載） |

Local 模式下，中文 / 中英混合會自動使用 **Breeze ASR 25**（聯發科繁中強化版），辨識中英夾雜場景顯著優於通用 Whisper。

## 安裝（給使用者）

**直接下載**：[Meeting.Transcriber.zip](https://github.com/kjohh/meeting-transcriber/releases/latest/download/Meeting.Transcriber.zip)
(永遠指向最新版,不用換連結)

或先看 [Releases 頁面](https://github.com/kjohh/meeting-transcriber/releases/latest) 的版本說明。

下載後 → 解壓 → 拖到「應用程式」資料夾 → 雙擊。

第一次打開會帶你走 onboarding：
1. 解釋工具用途
2. 一鍵授權螢幕錄製 + 麥克風（含 live 波形測試）
3. 選擇雲端 / 本機模式（雲端模式會請你填 Groq 金鑰並即時驗證；本機模式之後按開始錄音時會引導下載模型）
4. 完成 → 按 ● 開始錄音

> macOS 15 之後系統每月會跳一次螢幕錄製確認對話框，按「允許」即可，不影響使用。

## 開發 / 從原始碼 build

```bash
# 1. Python 套件（Homebrew Python 3.13）
/opt/homebrew/bin/python3.13 -m pip install --user --break-system-packages -r requirements.txt

# 2. 編譯 Swift 錄音 binary（Xcode Command Line Tools）
cd native && swift build -c release && cd ..

# 3. 本地開發跑 source mode
/opt/homebrew/bin/python3.13 app.py
# 或打包成 .app 用
./scripts/build-app.sh
# 產出在 dist/Meeting Transcriber.app
```

## 發佈

```bash
./scripts/build-app.sh                 # 重 build
./scripts/release.sh v0.1.0            # zip + upload to GitHub Releases
```

`release.sh` 用 `ditto -c -k --keepParent` 打 zip（保留 code signature，避免 Gatekeeper 拒絕），然後 `gh release create` 上傳。發完使用者從 Releases 頁面下載即可。

## 主要功能

- **中英混合辨識**：預設語言 `zh-en`，強制 `language="zh"` + bilingual prompt 範例 prime decoder，避免英文段被翻譯成中文或亂碼
- **Silence-aware chunking**：句子不會被死板的計時器切兩半。25 秒上限或 1.5 秒靜音偵測才切片
- **Repetition trim**：Whisper 經典的「同句重複 N 次」hallucination 自動截斷為最多 2 次
- **自訂詞彙表**：UI 內編輯 → `.vocab.local`（gitignored）→ 每次 chunk 即時讀取，prime Whisper 認識專有名詞 / 品牌 / 縮寫 / 人名
- **Backend 切換**：UI dropdown 一鍵切雲端 / 本機，敏感會議走本地
- **金鑰即時驗證**：填寫 Groq 金鑰時直接打 API 測試，無效不存
- **模型 cache 共用 lazy-take-notes**：本機若已裝過 [lazy-take-notes](https://github.com/CJHwong/lazy-take-notes)，模型不會重複下載
- **原生 Save 對話框**：透過 pywebview JS API bridge 呼叫 macOS save panel
- **權限自動偵測**：onboarding 內 polling `CGPreflightScreenCaptureAccess` + `AVCaptureDevice.authorizationStatus`，授權完成瞬間顯示綠 ✓

## UI 狀態機

| 狀態 | Record | New | lang/backend | Upload/Vocab | Save |
|---|---|---|---|---|---|
| 待機（剛開或 New 後）| ● 開始 | – | ✓ | ✓ | – |
| 錄音中 | ⏸ 暫停 | – | – | – | – |
| 暫停 | ▶ 繼續 | ✓ | – | – | ✓ |

session 期間語言 / backend 鎖定（避免「UI 改了但 in-flight session 不適用」的假切換）。Pause 中可 Save 中途進度或按 New 結束 session。

## 技術細節

- **系統音擷取**：Swift binary 用 ScreenCaptureKit (macOS 13+)，不需要 BlackHole 等虛擬音訊裝置
- **前端即時更新**：Server-Sent Events (`/events`)
- **設定儲存**：
  - Source mode：`./.config.json` + `./.vocab.local`（gitignored）
  - Bundle mode：`~/Library/Application Support/Meeting Transcriber/`
- **本地模型 cache**：`~/Library/Application Support/pywhispercpp/models/`（與 lazy-take-notes 共用）
- **TCC 處理**：每次 py2app rebuild 產生新 signature hash → macOS 視為不同 app → `scripts/build-app.sh` 會自動 `tccutil reset` 清舊紀錄

## 未來計畫

詳細待辦見 [TODO.md](TODO.md)。整體 UI / branding rebrand + Apple Developer 帳號簽章是計劃中下一階段。

## 致謝

本工具的核心引擎 — **silence-aware chunker、HF Hub model resolver、Whisper + Breeze ASR 整合** — 借鑒自同事 [@CJHwong](https://github.com/CJHwong) 的 [lazy-take-notes](https://github.com/CJHwong/lazy-take-notes)。lazy-take-notes 是一個極優秀的 terminal-based 會議轉錄工具,本專案算是基於它的概念延伸出一個有 UI、有 onboarding、能打包成 .app 給非技術朋友安裝的版本。模型 cache 路徑刻意與 lazy-take-notes 共用,如果兩個工具都裝,模型不會重複下載。
