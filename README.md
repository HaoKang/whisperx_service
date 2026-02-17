# WhisperX 本地语音识别服务

基于 FastAPI 的本地 WhisperX 封装服务，支持多个客户端应用调用。

## 功能特性

- **HTTP REST API**: 支持音频文件上传识别
- **优先级队列**: 支持任务优先级（高/中/低）
- **并发控制**: 可配置最大并发数，动态调整
- **结果缓存**: 基于音频内容哈希的重复请求缓存
- **说话人分离**: 支持多人语音识别
- **多种输出格式**: JSON / SRT / VTT / TEXT
- **可视化监控**: Web 监控页面

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
- `queue.max_concurrent`: 最大并发任务数
- `whisper.model`: Whisper 模型（默认 base）
- `whisper.huggingface_token`: 说话人分离需要

### 3. 启动服务

```bash
python app.py
```

## API 接口

### 1. 语音识别（优先级队列）

**POST** `/api/v1/recognize`

请求：multipart/form-data

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| file | File | 必填 | 音频文件 |
| language | string | auto | 语言代码 |
| diarize | bool | false | 说话人分离 |
| format | string | json | 输出格式 |
| priority | int | 2 | 优先级(1=高,2=中,3=低) |

### 2. 同步识别（适合小文件）

**POST** `/api/v1/recognize/sync`

### 3. 任务管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/tasks/{task_id}` | GET | 获取任务状态 |
| `/api/v1/queue/tasks/{task_id}/cancel` | POST | 取消任务 |

### 4. 队列管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/queue/stats` | GET | 队列统计 |
| `/api/v1/queue/config` | POST | 调整并发数 |
| `/api/v1/queue/tasks` | GET | 任务列表 |

### 5. 缓存管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/cache/stats` | GET | 缓存统计 |
| `/api/v1/cache/clear` | POST | 清空缓存 |

### 6. 其他

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/v1/models` | GET | 模型信息 |
| `/dashboard` | GET | 监控页面 |

## 使用示例

### Python

```python
import requests

url = "http://localhost:8000/api/v1/recognize"
files = {"file": open("audio.m4a", "rb")}
data = {
    "language": "zh",
    "diarize": True,
    "priority": 1,  # 高优先级
    "format": "json"
}

response = requests.post(url, files=files, data=data)
result = response.json()

# 轮询等待结果
while result.get("status") != "completed":
    task_id = result["task_id"]
    result = requests.get(f"http://localhost:8000/api/v1/tasks/{task_id}").json()

print(result["result"]["text"])
```

### cURL

```bash
curl -X POST http://localhost:8000/api/v1/recognize \
  -F "file=@audio.m4a" \
  -F "language=zh" \
  -F "diarize=true" \
  -F "priority=1" \
  -F "format=json"
```

## 监控页面

访问 `http://localhost:8000/dashboard` 可以：
- 查看服务状态
- 查看队列统计（运行中/等待中/已完成）
- 调整最大并发数
- 查看任务列表
- 取消等待中的任务
- 查看缓存状态

## 配置说明

### 优先级说明

| 优先级 | 值 | 适用场景 |
|--------|-----|----------|
| 高 | 1 | 对话语音，实时响应 |
| 中 | 2 | 普通任务 |
| 低 | 3 | 批处理，定时任务 |

### 缓存说明

- 缓存 Key：音频文件 SHA256 哈希
- 有效期：6 小时
- 最大缓存数：100 个

相同音频文件的重复请求会直接返回缓存结果。

## 目录结构

```
whisperx_service/
├── app.py                   # 主应用入口
├── config.py                # 配置管理
├── config.example.yaml       # 配置示例
├── config.yaml              # 运行时配置
├── whisper_service.py        # WhisperX 封装
├── task_queue.py            # 优先级队列管理
├── result_cache.py          # 结果缓存
├── models.py                # 数据模型
├── requirements.txt          # 依赖
└── templates/
    └── dashboard.html       # 监控页面
```
