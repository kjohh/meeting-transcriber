# Claude Guidelines — Meeting Transcriber

個人用會議逐字稿工具。技術細節在 README.md。

## 架構速覽

```
app.py                        ← Flask server + pywebview 入口，所有邏輯都在這
static/index.html             ← 前端 UI（單一 HTML 檔，含 CSS/JS）
native/Sources/coreaudio_tap/main.swift  ← Swift binary，系統音擷取
native/.build/release/coreaudio_tap     ← 編譯後的 binary（不進 git）
Meeting Transcriber.app/      ← 雙擊開啟的 macOS app bundle（不進 git）
.config.json                  ← 存 Groq API key（不進 git）
```

## 關鍵設計決策

- **UI 用 web（Flask + pywebview）而非 tkinter**：macOS 系統 Python 的 Tk 版本（8.5）在 dark mode 下渲染全黑，改用 Flask + pywebview（WKWebView）解決。Flask 跑在 daemon thread，pywebview 開視窗，視窗關掉整個 process 結束
- **系統音用 ScreenCaptureKit（Swift）**：不需要 BlackHole，macOS 13+ 原生支援，Swift binary 輸出 float32 PCM 16kHz 到 stdout，Python 讀進來
- **前端即時更新用 SSE**：`/events` endpoint，Flask generator 推送 transcript/status/state 事件
- **逐字稿固定 20 秒一段**：系統音＋麥克風分別 buffer，兩邊都到 target 長度才混音送 Groq

## 開發注意事項

- Python 執行環境是 `/opt/homebrew/bin/python3.13`（Homebrew）。系統 Python 3.9.6 無法裝 pywebview（pyobjc-core 編譯失敗），已全面改用 Homebrew Python 3.13
- app.py 內還有 `from __future__ import annotations` 和 `Optional[X]` 寫法，保留即可，3.13 相容
- 修改 Swift binary 後需重新編譯：`cd native && swift build -c release`
- Flask 跑在 port 8765，`use_reloader=False`（避免 reloader 干擾背景執行緒）
- 音訊相關的全域狀態（`_recording`, `_paused`, `_sys_buf` 等）在 app.py 頂層，多執行緒共用，寫入時要注意 `_buf_lock`

## 未來計畫（已知需求，別主動動）

- **py2app**：打包成可分發的 .app，適合開源給沒有 Python 環境的人用（目前的 .app bundle 只是 shell script wrapper，還是需要 Homebrew Python 在機器上）
