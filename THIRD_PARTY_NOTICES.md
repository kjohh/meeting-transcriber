# 第三方軟體授權聲明

Meeting Transcriber 使用了下列開放原始碼軟體與模型。感謝這些專案的作者。

## 語音模型

| 名稱 | 用途 | 授權 | 來源 |
| --- | --- | --- | --- |
| Whisper small（ggml 格式） | App 內建的基本語音模型 | MIT | https://huggingface.co/ggerganov/whisper.cpp |
| Whisper large-v3-turbo（ggml 格式） | 多語通用模型（下載） | MIT | https://huggingface.co/ggerganov/whisper.cpp |
| Breeze ASR 25（MediaTek Research） | 中文強化模型（下載） | Apache-2.0 | https://huggingface.co/MediaTek-Research/Breeze-ASR-25 |
| Breeze ASR 25 ggml 轉檔 | 中文強化模型的 whisper.cpp 格式 | Apache-2.0 | https://huggingface.co/alan314159/Breeze-ASR-25-whispercpp |

## 程式元件

| 名稱 | 授權 |
| --- | --- |
| whisper.cpp / ggml | MIT |
| pywhispercpp | MIT |
| Python | PSF-2.0 |
| pywebview、Bottle、proxy_tools | BSD-3-Clause（pywebview）、MIT |
| Flask、Werkzeug、Jinja2、Click、ItsDangerous、MarkupSafe | BSD-3-Clause |
| Blinker | MIT |
| NumPy | BSD-3-Clause（含 0BSD、MIT、Zlib、CC0-1.0 子元件） |
| sounddevice、PortAudio | MIT |
| CFFI | MIT-0 |
| PyObjC | MIT |
| Groq Python SDK | Apache-2.0 |
| httpx、httpcore、idna | BSD-3-Clause |
| anyio、h11、pydantic、pydantic-core、annotated-types、typing-inspection | MIT |
| distro | Apache-2.0 |
| sniffio | MIT 或 Apache-2.0 |
| typing-extensions | PSF-2.0 |
| requests、importlib-metadata | Apache-2.0 |
| urllib3、charset-normalizer、PyYAML、rich、markdown-it-py、mdurl、platformdirs、zipp、tomli、backports.tarfile | MIT |
| Pygments | BSD-2-Clause |
| tqdm | MPL-2.0 與 MIT |
| certifi | MPL-2.0 |
| OpenSSL | Apache-2.0 |
| SQLite | Public Domain |
| XZ Utils（liblzma） | 0BSD |
| mpdecimal | BSD-2-Clause |

## 致謝

即時斷句的做法參考了 [lazy-take-notes](https://github.com/CJHwong/lazy-take-notes)（CJHwong）。

各元件的完整授權條款，請見各專案原始碼中的 LICENSE 檔。
