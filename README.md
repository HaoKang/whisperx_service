# WhisperX 本地语音识别服务

基于 FastAPI 的本地 WhisperX 封装服务，支持多个客户端应用调用。

## 功能特性

- **HTTP REST API**: 支持音频文件上传识别
- **任务队列**: 支持并发处理多个识别请求
- **模型缓存**: 支持指定 WhisperX 模型
- ** diarization**: 支持说话人分离（多人语音播客）
- **结果格式**: 支持 JSON 和 SRT 字幕格式

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

复制并编辑配置文件：

```bash
cp config.example.yaml config.yaml
```

主要配置项：
- `host`: 服务监听地址（默认 0.0.0.0）
- `port`: 服务端口（默认 8000）
- `whisper.model`: Whisper 模型（默认 base）
- `whisper.device`: 运行设备（cuda/cpu，默认 auto）
- `whisper.diarize`: 是否启用说话人分离

### 3. 启动服务

```bash
python app.py
```

或使用 uvicorn：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API 接口

### 1. 语音识别

**POST** `/api/v1/recognize`

请求：multipart/form-data
- `file`: 音频文件（支持 mp3, wav, m4a, flac, ogg）
- `language`: 语言代码（可选，如 zh、en）
- `diarize`: 是否启用说话人分离（true/false）
- `format`: 输出格式（json/srt，默认 json）

响应示例：
```json
{
  "success": true,
  "task_id": "xxx",
  "text": "识别文本内容",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "识别文本内容",
      "speaker": "SPEAKER_01"
    }
  ]
}
```

### 2. 获取任务状态

**GET** `/api/v1/tasks/{task_id}`

### 3. 健康检查

**GET** `/health`

### 4. 模型信息

**GET** `/api/v1/models`

## 使用示例

### cURL

```bash
curl -X POST http://localhost:8000/api/v1/recognize \
  -F "file=@audio.mp3" \
  -F "diarize=true" \
  -F "format=json"
```

### Python

```python
import requests

url = "http://localhost:8000/api/v1/recognize"
files = {"file": open("audio.mp3", "rb")}
data = {"diarize": True, "format": "json"}

response = requests.post(url, files=files, data=data)
result = response.json()
print(result["text"])
```

## 目录结构

```
whisperx_service/
├── app.py              # 主应用入口
├── config.py           # 配置管理
├── config.example.yaml # 配置示例
├── whisper_service.py  # WhisperX 封装
├── models.py           # 数据模型
├── requirements.txt    # 依赖
└── tests/              # 测试
```
