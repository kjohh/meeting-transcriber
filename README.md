# Meeting Transcriber

個人用的會議逐字稿工具。錄製系統音（遠端會議對方的聲音）＋麥克風，每 20 秒送一次 Groq Whisper API 轉文字，即時顯示在瀏覽器視窗裡。

## 架構

```
python3 app.py
  ├── Flask server (port 8765)          ← 提供 web UI + API endpoints
  ├── native/coreaudio_tap (Swift)      ← ScreenCaptureKit 錄系統音，輸出 float32 PCM
  ├── sounddevice                       ← 錄麥克風
  └── Groq Whisper API                  ← 雲端逐字稿（whisper-large-v3-turbo）
```

每 20 秒把系統音＋麥克風混音成一個 WAV chunk，送 Groq API，結果即時推到前端（SSE）。

## 安裝

```bash
# 1. Python 套件
pip3 install -r requirements.txt

# 2. 編譯 Swift 錄音 binary（需要 Xcode Command Line Tools）
cd native && swift build -c release && cd ..
```

## 執行

```bash
python3 app.py
# 或雙擊 Meeting Transcriber.command
```

第一次執行需在 System Settings → Privacy & Security → Screen & System Audio Recording 授權給 Terminal.app。

## 技術細節

- **系統音擷取**：Swift binary 用 ScreenCaptureKit（macOS 13+），不需要安裝 BlackHole 等虛擬音訊設備
- **轉錄模型**：Groq `whisper-large-v3-turbo`，免費方案 7,200 秒/小時
- **前端即時更新**：Server-Sent Events（SSE）
- **API key 儲存**：`.config.json`（本機，不進 git）

## 未來可以做的事

- **獨立視窗**：用 `pywebview` 包起來，不需要開瀏覽器
- **打包成 .app**：用 `py2app` 打包，可以分享給沒有 Python 環境的人，但檔案會很大（200-400MB）
- **逐字稿語言設定**：Groq API 支援指定語言，可以加一個 UI 選項
- **chunk 時間調整**：目前固定 20 秒，可以做成可調
