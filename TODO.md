# TODO

工具：個人會議轉錄。輕量 task list，不用 Jira / Issues。動了什麼勾掉 / 改狀態即可。

狀態：`TODO`（要做）/ `WIP`（進行中）/ `Done` / `Maybe`（再考慮）

## 安裝 & Onboarding（核心 friction，第一步要解決）

| 類型 | 描述 | 狀態 |
| --- | --- | --- |
| Install | py2app 打包成 self-contained .app（不含模型，~200MB） | TODO |
| Onboarding | First-launch 螢幕錄製授權說明 modal（解釋為何需要、引導開啟系統設定，附截圖）。完成 trigger 寫 `.config.json` 的 `onboarding_completed` flag | TODO |
| Onboarding | First-launch 模型下載步驟（旁邊註明每個模型對應哪些語言情境） | TODO |
| Onboarding | Settings 加「重新看 tutorial」按鈕，繞過 flag 重跑 onboarding（dev 自己也要看得到） | TODO |
| UX | 預設 backend 改 Local，Cloud 藏進 Settings | TODO |

## Settings 介面

| 類型 | 描述 | 狀態 |
| --- | --- | --- |
| UI | 加 Settings 頁/Modal，收容：Groq API key、模型下載清單、（未來）chunk 參數 | TODO |
| UX | Settings 入口（齒輪 icon）放工具列右側 | TODO |

## UI/UX 小改

| 類型 | 描述 | 狀態 |
| --- | --- | --- |
| UX | 按 New 時跳 confirm dialog（transcript 非空時） | TODO |
| UX | Save 變主要動作（強調色），New 變次要 | TODO |
| UX | Disabled 控制項加 tooltip 解釋為什麼鎖 | TODO |
| UX | Download modal 依當前 language 只顯示需要的模型 | TODO |
| UX | macOS 月度授權 prompt：在 status bar 或一次性 modal 提醒「正常，按允許即可」 | TODO |
| UX | Jargon 改名：「Vocab」→「常用詞彙」、考慮重新命名「Cloud / Local」 | Maybe |

## 產品功能（跟「給誰用」無關，純功能擴充）

| 類型 | 描述 | 狀態 |
| --- | --- | --- |
| Feature | LLM summary / action items 整理（可走 Ollama 本地或 Groq） | Maybe |
| Feature | Speaker diarization（多人會議區分發言者） | Maybe |
| Feature | 匯出 docx / markdown（不只 .txt） | Maybe |
| Feature | Chunk 參數做成 UI 設定（CHUNK_DURATION / PAUSE_DURATION） | Maybe |

## 已知 bug / 觀察

| 類型 | 描述 | 狀態 |
| --- | --- | --- |

（暫無）

## Done

- 加入 local backend（pywhispercpp + Breeze ASR + large-v3-turbo-q8）
- Silence-aware chunker（取代死板 10s 切片）
- Repetition trim + loop detection（解 Whisper hallucination）
- Voice activity ratio gate
- UI 重整：合併 Pause/Stop、加 New session、state machine
- 中英混合預設 + bilingual prompt priming
- Vocabulary editor modal + `.vocab.local`
- Native macOS save dialog（pywebview JS API）
- launcher.swift 改進（--user pip install、失敗時彈 dialog）
- Ad-hoc deep codesign（解決重複跳螢幕錄製授權）
- Audio waveform 換成輕量 RAF 滾動小波紋
