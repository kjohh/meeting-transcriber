# TODO

工具：個人會議轉錄。輕量 task list，不用 Jira / Issues。動了什麼勾掉 / 改狀態即可。

狀態：`TODO`（要做）/ `WIP`（進行中）/ `Done` / `Maybe`（再考慮）

## 下一階段（自己主導）

| 類型 | 描述 | 狀態 |
| --- | --- | --- |
| Rebrand | 整體 UI 重新設計 — 漸層 / 光暈 / 動態效果,做成完整的視覺作品 | TODO |
| Rebrand | 換 fancy 一點的產品名稱(目前還叫 Meeting Transcriber) | TODO |
| Distribution | Rebrand 成熟後註冊 Apple Developer Program,正式簽章 + Notarization → 解決月度 TCC re-prompt | Maybe |

## 等使用者回饋

| 類型 | 描述 | 狀態 |
| --- | --- | --- |
| Bug / 優化 | 等實際給朋友試之後再回收回饋,排優先序 | — |

## UI/UX 小改（之前討論待辦,跟 rebrand 一起做也可)

| 類型 | 描述 | 狀態 |
| --- | --- | --- |
| UX | 錄音中關 app 時跳警示確認(避免誤關丟資料)。實作用 pywebview 的 closing callback + confirm dialog | TODO |
| UX | 按 New 時跳 confirm dialog(transcript 非空時) | TODO |
| UX | Save 視覺更突出,New 視覺更次要 | TODO |
| UX | Disabled 控制項加 tooltip 解釋為什麼鎖 | TODO |
| UX | Download modal 依當前 language 只顯示需要的模型 | TODO |
| UX | macOS 月度授權 prompt:在 status bar 或一次性 modal 提醒「正常,按允許即可」 | TODO |

## 產品功能（跟「給誰用」無關,純功能擴充)

| 類型 | 描述 | 狀態 |
| --- | --- | --- |
| Feature | LLM summary / action items 整理(可走 Ollama 本地或 Groq) | Maybe |
| Feature | Speaker diarization(多人會議區分發言者) | Maybe |
| Feature | 匯出 docx / markdown(不只 .txt) | Maybe |
| Feature | Chunk 參數做成 UI 設定(CHUNK_DURATION / PAUSE_DURATION) | Maybe |

## 已知 bug / 觀察

| 類型 | 描述 | 狀態 |
| --- | --- | --- |

(暫無)

## Done

### 安裝 / 打包
- py2app 打包成 self-contained .app(139MB,不含模型)
- `scripts/build-app.sh` 一鍵 build + deep codesign + TCC reset + onboarding flag reset
- `scripts/build-icon.sh` 從 PNG 生 .icns
- `scripts/release.sh` ditto-zip + gh release create 上傳 GitHub Releases
- App icon(placeholder,之後 rebrand 換)
- 修 sounddevice libportaudio.dylib 在 zip 內 dlopen 失敗的問題(`packages` 加 sounddevice)

### Onboarding
- 4-step first-launch onboarding modal(自動跳 + 用 Settings「重新看引導」可手動觸發)
- Step 2 螢幕錄製 + 麥克風授權都在同一頁
- 一鍵授權按鈕(spawn coreaudio_tap 觸發 macOS 對話框)
- 「打開系統設定」捷徑(x-apple.systempreferences URL scheme)
- 即時 polling 偵測授權狀態(CGPreflight + AVFoundation,不觸發 prompt)
- 麥克風授權後 live 波形預覽
- Step 3 雲端 inline key input + 即時驗證 + 已存的 key 自動 pre-fill

### Settings 介面
- ⚙ 齒輪 icon 入口
- API Key section(填寫 + 即時驗證)
- 本機模型 section(下載 + 進度)
- 說明 section(重新看引導)

### 核心功能
- Local backend 透過 pywhispercpp,自動從 HF Hub 下載模型
- 模型 cache 與 lazy-take-notes 共用路徑
- 中英混合自動使用 Breeze ASR 25
- Silence-aware chunker(取代死板 10s 切片)
- Repetition trim + loop detection(解 Whisper hallucination)
- Voice activity ratio gate
- Prompt chain(跨 chunk 上下文連續)
- 自訂詞彙表 UI(Vocab modal + `.vocab.local`)
- Native macOS save dialog(pywebview JS API bridge)
- 系統音失敗警告 banner(僅 mic recording 時提醒)
- 雲端模式無 key 時即時警告 banner

### UI / UX
- 全 UI 台灣繁中
- 工具列簡化:Pause/Stop 合併成單一錄音 toggle + New session
- 鎖定 selector(錄音中不可改 lang/backend)
- 小型 RAF-driven 音訊波形
- AA 對比度連結色(--link variable)
- (i) tooltip 自製(WKWebView title= 不可靠)
- /pause 進入時 flush 避免漏字
- /stop 等 chunk worker join 完成,New 不再 race 殘留文字
