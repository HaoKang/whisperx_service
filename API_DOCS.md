# WhisperX 语音识别服务 API 文档

> 版本: 1.1.0
> 基础 URL: `http://your-server:8001`

## 目录

- [概述](#概述)
- [认证](#认证)
- [通用响应](#通用响应)
- [语音识别接口](#语音识别接口)
  - [异步识别（推荐）](#异步识别推荐)
  - [同步识别](#同步识别)
- [任务管理接口](#任务管理接口)
  - [查询任务状态](#查询任务状态)
  - [获取识别结果](#获取识别结果)
  - [取消任务](#取消任务)
- [队列管理接口](#队列管理接口)
- [缓存管理接口](#缓存管理接口)
- [系统接口](#系统接口)
- [错误码说明](#错误码说明)
- [最佳实践](#最佳实践)
- [SDK 示例](#sdk-示例)

---

## 概述

WhisperX 语音识别服务提供基于 HTTP REST API 的语音转文字能力，支持：

- **多格式输出**: JSON / SRT / VTT / TEXT
- **优先级队列**: 高/中/低三级优先级
- **结果缓存**: 基于音频内容哈希的智能缓存
- **说话人分离**: 可选的多说话人识别（需要配置 HuggingFace Token）

### 支持的音频格式

| 格式 | 扩展名 |
|-----|--------|
| MP3 | .mp3 |
| WAV | .wav |
| M4A/AAC | .m4a |
| FLAC | .flac |
| OGG | .ogg |
| WebM | .webm |

### 限制

- 单文件最大: 100MB（可配置）
- 请求限流: 60次/分钟, 1000次/小时

---

## 认证

当前版本无需认证。生产环境建议通过反向代理添加认证层。

---

## 通用响应

### 成功响应

```json
{
  "success": true,
  "task_id": "uuid-string",
  ...
}
```

### 错误响应

```json
{
  "detail": "错误描述信息"
}
```

### 响应头

| 头部 | 说明 |
|-----|------|
| `X-Request-ID` | 请求追踪 ID，用于日志排查 |

---

## 语音识别接口

### 异步识别（推荐）

适合大文件或批量处理场景，通过队列异步处理。

**请求**

```
POST /api/v1/recognize
Content-Type: multipart/form-data
```

**参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|-----|------|------|--------|------|
| file | File | 是 | - | 音频文件 |
| language | string | 否 | auto | 语言代码（zh/en/ja 等），为空自动检测 |
| diarize | bool | 否 | false | 是否启用说话人分离 |
| format | string | 否 | json | 输出格式: json/srt/vtt/text |
| model | string | 否 | base | 模型: tiny/base/small/medium/large-v3 |
| priority | int | 否 | 2 | 优先级: 1=高, 2=中, 3=低 |

**响应**

```json
{
  "success": true,
  "task_id": "7a2564e7-a094-4e21-a530-af9cb856e0f8",
  "text": "任务已加入队列，优先级: 中",
  "segments": null,
  "language": null,
  "duration": null,
  "error": null,
  "cached": false
}
```

**示例**

```bash
curl -X POST http://localhost:8001/api/v1/recognize \
  -F "file=@audio.m4a" \
  -F "language=zh" \
  -F "priority=1" \
  -F "format=json"
```

---

### 同步识别

适合小文件（<1分钟）实时处理场景，直接返回结果。

**请求**

```
POST /api/v1/recognize/sync
Content-Type: multipart/form-data
```

**参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|-----|------|------|--------|------|
| file | File | 是 | - | 音频文件 |
| language | string | 否 | auto | 语言代码 |
| diarize | bool | 否 | false | 是否启用说话人分离 |
| format | string | 否 | json | 输出格式: json/srt/vtt/text |
| model | string | 否 | base | 模型名称 |

**响应**

```json
{
  "success": true,
  "task_id": "f00fe166-ad06-464b-b9d5-a9f4ffc24c59",
  "text": "大家好 这里是尼达波克...",
  "segments": [
    {
      "start": 9.262,
      "end": 36.45,
      "text": "大家好 这里是尼达波克...",
      "speaker": null,
      "words": null
    }
  ],
  "language": "zh",
  "duration": 141.143,
  "error": null,
  "cached": false
}
```

**示例**

```bash
curl -X POST http://localhost:8001/api/v1/recognize/sync \
  -F "file=@audio.m4a" \
  -F "language=zh" \
  -F "format=srt"
```

---

## 任务管理接口

### 查询任务状态

**请求**

```
GET /api/v1/tasks/{task_id}
```

**响应**

```json
{
  "task_id": "7a2564e7-a094-4e21-a530-af9cb856e0f8",
  "status": "completed",
  "result": {
    "task_id": "b2bb1a32-6c96-492e-8927-91f47da7faeb",
    "text": "识别的文本内容...",
    "segments": [...],
    "language": "zh",
    "duration": 141.143
  },
  "error": null,
  "progress": null,
  "wait_time": null
}
```

**任务状态说明**

| 状态 | 说明 |
|-----|------|
| pending | 等待处理 |
| processing | 正在处理 |
| completed | 处理完成 |
| failed | 处理失败 |

**示例**

```bash
curl http://localhost:8001/api/v1/tasks/7a2564e7-a094-4e21-a530-af9cb856e0f8
```

---

### 获取识别结果

支持按不同格式获取结果。

**请求**

```
GET /api/v1/result/{task_id}?format={format}
```

**参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|-----|------|------|--------|------|
| format | string | 否 | json | 输出格式: json/srt/vtt/text |

**响应**

- `format=json`: 返回 JSON 对象
- `format=srt/vtt/text`: 返回纯文本

**示例**

```bash
# 获取 SRT 字幕
curl "http://localhost:8001/api/v1/result/7a2564e7-a094-4e21-a530-af9cb856e0f8?format=srt"
```

---

### 取消任务

仅可取消 `pending` 状态的任务。

**请求**

```
POST /api/v1/queue/tasks/{task_id}/cancel
```

**响应**

```json
{
  "success": true,
  "message": "任务已取消"
}
```

**示例**

```bash
curl -X POST http://localhost:8001/api/v1/queue/tasks/7a2564e7-a094-4e21-a530-af9cb856e0f8/cancel
```

---

## 队列管理接口

### 获取队列统计

**请求**

```
GET /api/v1/queue/stats
```

**响应**

```json
{
  "max_concurrent": 2,
  "worker_threads": 2,
  "running": 1,
  "pending": 3,
  "completed": 15,
  "total": 19,
  "avg_processing_time": 36.55
}
```

---

### 调整并发数

**请求**

```
POST /api/v1/queue/config
Content-Type: application/json
```

**请求体**

```json
{
  "max_concurrent": 4
}
```

**响应**

```json
{
  "success": true,
  "max_concurrent": 4
}
```

---

### 获取任务列表

**请求**

```
GET /api/v1/queue/tasks?limit=50
```

**响应**

```json
[
  {
    "task_id": "7a2564e7-a094-4e21-a530-af9cb856e0f8",
    "priority": 1,
    "priority_label": "HIGH",
    "status": "completed",
    "created_at": 1708123456.789,
    "started_at": 1708123457.123,
    "completed_at": 1708123493.456,
    "duration": 36.33
  }
]
```

---

## 缓存管理接口

### 获取缓存统计

**请求**

```
GET /api/v1/cache/stats
```

**响应**

```json
{
  "hits": 10,
  "misses": 5,
  "hit_rate": "66.7%",
  "size": 5,
  "max_size": 100,
  "ttl": 21600
}
```

---

### 清空缓存

**请求**

```
POST /api/v1/cache/clear
```

**响应**

```json
{
  "success": true,
  "message": "缓存已清空"
}
```

---

## 系统接口

### 健康检查

**请求**

```
GET /health
```

**响应**

```json
{
  "status": "healthy",
  "version": "1.1.0",
  "model_loaded": true,
  "device": "cpu",
  "is_apple_silicon": true
}
```

---

### 获取模型信息

**请求**

```
GET /api/v1/models
```

**响应**

```json
{
  "name": "base",
  "device": "cpu",
  "diarize_available": false,
  "compute_type": "int8"
}
```

---

### 获取设备信息

**请求**

```
GET /api/v1/device
```

**响应**

```json
{
  "device": "cpu",
  "is_apple_silicon": true,
  "compute_type": "int8",
  "threads": 8,
  "mps_available": true
}
```

---

### 监控面板

**请求**

```
GET /dashboard
```

返回 Web 监控页面。

---

## 错误码说明

| 状态码 | 说明 |
|-------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 任务不存在 |
| 413 | 文件大小超过限制 |
| 429 | 请求过于频繁 |
| 500 | 服务器内部错误 |

**错误响应示例**

```json
{
  "detail": "不支持的文件类型: .exe。支持的类型: {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm'}"
}
```

---

## 最佳实践

### 1. 选择正确的接口

| 场景 | 推荐接口 |
|-----|---------|
| 实时语音（<1分钟） | `/api/v1/recognize/sync` |
| 大文件/批量处理 | `/api/v1/recognize` + 轮询 |
| 高并发场景 | `/api/v1/recognize` + 优先级队列 |

### 2. 使用优先级队列

```python
# 实时对话 - 高优先级
priority = 1

# 普通任务 - 中优先级
priority = 2

# 批处理任务 - 低优先级
priority = 3
```

### 3. 利用缓存

相同音频文件的重复请求会自动命中缓存，响应时间 <1 秒。缓存基于音频内容哈希，与文件名无关。

### 4. 轮询策略

```python
import time
import requests

# 提交任务
resp = requests.post(url, files=files, data=data)
task_id = resp.json()["task_id"]

# 轮询等待
while True:
    status = requests.get(f"{base_url}/api/v1/tasks/{task_id}").json()
    if status["status"] == "completed":
        result = status["result"]
        break
    elif status["status"] == "failed":
        raise Exception(status.get("error"))
    time.sleep(2)  # 建议间隔 2-5 秒
```

### 5. 输出格式选择

| 格式 | 适用场景 |
|-----|---------|
| json | 需要时间戳和分段信息 |
| srt | 视频字幕 |
| vtt | Web 视频字幕 |
| text | 只需要纯文本 |

---

## SDK 示例

### Python

```python
import requests
import time

class WhisperXClient:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url

    def recognize_async(self, file_path, language=None, priority=2, format="json"):
        """异步识别"""
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"language": language, "priority": priority, "format": format}
            resp = requests.post(f"{self.base_url}/api/v1/recognize", files=files, data=data)
        resp.raise_for_status()
        return resp.json()

    def recognize_sync(self, file_path, language=None, format="json"):
        """同步识别"""
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"language": language, "format": format}
            resp = requests.post(f"{self.base_url}/api/v1/recognize/sync", files=files, data=data)
        resp.raise_for_status()
        return resp.json()

    def wait_for_result(self, task_id, timeout=300, interval=2):
        """等待异步任务完成"""
        start = time.time()
        while time.time() - start < timeout:
            resp = requests.get(f"{self.base_url}/api/v1/tasks/{task_id}")
            data = resp.json()
            status = data["status"]
            if status == "completed":
                return data["result"]
            elif status == "failed":
                raise Exception(data.get("error"))
            time.sleep(interval)
        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

    def recognize(self, file_path, language=None, format="json", priority=2):
        """一步完成识别（推荐）"""
        if priority:
            resp = self.recognize_async(file_path, language, priority, format)
            return self.wait_for_result(resp["task_id"])
        else:
            return self.recognize_sync(file_path, language, format)


# 使用示例
client = WhisperXClient("http://localhost:8001")

# 方式1: 一步完成
result = client.recognize("audio.m4a", language="zh", format="json")
print(result["text"])

# 方式2: 异步 + 轮询
task = client.recognize_async("audio.m4a", language="zh", priority=1)
result = client.wait_for_result(task["task_id"])
print(result["text"])
```

### JavaScript / Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const BASE_URL = 'http://localhost:8001';

async function recognize(file_path, options = {}) {
    const form = new FormData();
    form.append('file', fs.createReadStream(file_path));
    if (options.language) form.append('language', options.language);
    if (options.format) form.append('format', options.format);
    if (options.priority) form.append('priority', options.priority);

    const resp = await axios.post(`${BASE_URL}/api/v1/recognize`, form, {
        headers: form.getHeaders()
    });

    const taskId = resp.data.task_id;

    // 轮询等待结果
    while (true) {
        const status = await axios.get(`${BASE_URL}/api/v1/tasks/${taskId}`);
        if (status.data.status === 'completed') {
            return status.data.result;
        } else if (status.data.status === 'failed') {
            throw new Error(status.data.error);
        }
        await new Promise(r => setTimeout(r, 2000));
    }
}

// 使用示例
(async () => {
    const result = await recognize('audio.m4a', { language: 'zh', format: 'json' });
    console.log(result.text);
})();
```

### cURL

```bash
# 同步识别
curl -X POST http://localhost:8001/api/v1/recognize/sync \
  -F "file=@audio.m4a" \
  -F "language=zh" \
  -F "format=json"

# 异步识别
curl -X POST http://localhost:8001/api/v1/recognize \
  -F "file=@audio.m4a" \
  -F "language=zh" \
  -F "priority=1"

# 查询状态
curl http://localhost:8001/api/v1/tasks/{task_id}

# 获取 SRT 字幕
curl "http://localhost:8001/api/v1/result/{task_id}?format=srt"
```

---

## 联系支持

如有问题，请访问: https://github.com/your-repo/whisperx_service/issues
