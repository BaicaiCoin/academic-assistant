# Academic Assistant — 部署与配置指南

## 目录
1. [前置依赖](#1-前置依赖)
2. [获取代码](#2-获取代码)
3. [后端配置](#3-后端配置)
4. [前端配置](#4-前端配置)
5. [快速体验：导入预置 GRPO 数据](#5-快速体验导入预置-grpo-数据)
6. [启动所有服务](#6-启动所有服务)
7. [验证运行正常](#7-验证运行正常)
8. [常见问题](#8-常见问题)

---

## 1. 前置依赖

请确保以下软件已安装：

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Conda | 任意 | 管理 Python 环境 |
| Node.js | ≥ 18 | 运行前端 |
| Redis | ≥ 6 | ARQ 任务队列 |
| CUDA | ≥ 11.8（推荐） | BGE-M3 / Whisper GPU 加速，无 GPU 也可运行但更慢 |
| ffmpeg | 任意 | 视频帧提取（conda 环境自带） |

**外部服务账号**（需自行申请）：

| 服务 | 用途 | 申请地址 |
|------|------|---------|
| DeepSeek API | 对话 LLM | platform.deepseek.com |
| Google Gemini API | 视频 PPT 解析 | aistudio.google.com |
| Neo4j Aura（免费版） | 图数据库 | console.neo4j.io |
| GitHub Personal Access Token | GitHub MCP 工具 | github.com → Settings → Developer settings |

---

## 2. 获取代码

项目分为前后端两个独立文件夹：

```
Academic_assistant/   ← 后端（FastAPI + LangGraph）
Academic_frontend/    ← 前端（React + Vite）
```

---

## 3. 后端配置

### 3.1 创建 Conda 环境

```bash
cd Academic_assistant
conda env create -f environment.yml
conda activate acaAss
```

> `environment.yml` 已包含所有 pip 依赖，包括 `langgraph`、`pymilvus`、`faster-whisper`、`arxiv` 等。

### 3.2 安装额外依赖

`environment.yml` 中未列出的包需手动安装：

```bash
pip install arq fastapi uvicorn python-dotenv
pip install langchain-openai  # DeepSeek 走 OpenAI 兼容接口
pip install langchain-mcp-adapters
pip install docling
pip install neo4j
pip install pymilvus[model]   # 包含 BGE-M3
```

### 3.3 配置 .env 文件

在 `Academic_assistant/` 根目录创建 `.env` 文件：

```dotenv
# LLM
DEEPSEEK_API_KEY="your_deepseek_api_key"

# Gemini（用于视频 PPT 解析）
GEMINI_API_KEY="your_gemini_api_key"
GEMINI_MODEL="gemini-2.0-flash"

# Neo4j Aura
NEO4J_URI="neo4j+s://xxxxxxxx.databases.neo4j.io"
NEO4J_USERNAME="xxxxxxxx"
NEO4J_PASSWORD="your_neo4j_password"
NEO4J_DATABASE="xxxxxxxx"

# Milvus（本地文件路径）
MILVUS_DB_PATH="./data/dataset/GRPO.db"

# GitHub MCP
GITHUB_TOKEN="github_pat_xxxxxxxxxxxx"

# 数据库路径（默认值即可）
SQLITE_DB_PATH="./data/dataset/threads.db"
MEMORY_DB_PATH="./data/dataset/memories.db"

# Redis（默认本地，可不填）
REDIS_URL="redis://localhost:6379"
```

> **Neo4j Aura 注意**：免费版实例若长期未使用会自动暂停，使用前请登录控制台确认实例状态为 Active。

---

## 4. 前端配置

前端代码位于仓库根目录下的 `frontend/` 子目录：

```bash
cd frontend
npm install
```

前端默认连接 `http://localhost:8000`，无需修改配置即可使用。

若后端运行在其他地址，修改 `frontend/.env`：

```dotenv
VITE_API_URL=http://your-backend-address:8000
```

---

## 5. 快速体验：导入预置 GRPO 数据

项目附带了已处理好的 GRPO 论文与视频数据，可跳过耗时的下载和处理步骤，直接建立索引进行问答。

**已提供的数据文件：**

```
data/dataset/GRPO.db          ← Milvus 向量库（含论文 + 视频 embedding）
data/pdf_tmp/GRPO.json        ← 论文分块数据
data/video_tmp/GRPO/group_2.json  ← 视频处理结果（含 PPT 解析 + 字幕）
data/graph_tmp/GRPO/          ← 实体抽取中间结果
```

**第一步：确认 .env 中 Milvus 路径指向预置数据库**

`.env` 中应为：
```dotenv
MILVUS_DB_PATH="./data/dataset/GRPO.db"
```

Milvus 向量数据已在 `GRPO.db` 中，**无需额外操作**。

**第二步：将数据导入 Neo4j**

在 `Academic_assistant/` 目录下，激活 conda 环境后执行：

```bash
conda activate acaAss
python scripts/import_grpo_data.py
```

脚本会依次完成：初始化约束 → 导入论文分块 → 导入视频分块 → 导入实体及关系。看到 `✅ 导入完成！` 即成功。

完成后，Milvus 向量检索 + Neo4j 图查询均已就绪，可直接进行问答。

---

## 6. 启动所有服务

需要开启**四个终端**（或使用 tmux / screen）：

### 终端 1：Redis

```bash
redis-server
```

### 终端 2：ARQ Worker（后台任务处理）

```bash
cd Academic_assistant
conda activate acaAss
arq src.worker.WorkerSettings
```

### 终端 3：FastAPI 后端

```bash
cd Academic_assistant
conda activate acaAss
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

> 首次启动会加载 BGE-M3 模型，约需 30-60 秒，看到 `Application startup complete` 即为就绪。

### 终端 4：前端开发服务器

```bash
cd frontend
npm run dev
```

浏览器访问 `http://localhost:5173` 即可使用。

---

## 7. 验证运行正常

**后端健康检查：**
```bash
curl http://localhost:8000/health
# 期望返回：{"ok": true}
```

**测试对话（示例问题）：**

在前端输入以下问题，验证 GRPO 数据已加载：
- `GRPO 相比 PPO 的核心改进是什么？`
- `视频第 5 页 PPT 说了什么？`
- `论文的实验结果如何？`

---

## 8. 常见问题

**Q: 启动时报 `ConnectionRefusedError` 或 Redis 相关错误**

确认 Redis 已启动：`redis-cli ping` 应返回 `PONG`。

**Q: BGE-M3 加载慢 / OOM**

首次加载会下载模型文件（约 2GB）。无 GPU 时会自动使用 CPU，速度较慢但可运行。

**Q: Neo4j 连接超时 / DNS 解析失败**

登录 [Neo4j Aura Console](https://console.neo4j.io) 确认实例为 Active 状态（免费版会自动暂停）。

**Q: 前端显示空白或无法连接后端**

检查 `Academic_frontend/.env` 中 `VITE_API_URL` 是否与后端地址一致，修改后需重启 `npm run dev`。

**Q: 想处理新的视频或论文**

通过 API 提交：
```bash
# 提交视频（支持 Bilibili / YouTube）
curl -X POST http://localhost:8000/ingest/video \
     -H "Content-Type: application/json" \
     -d '{"video_url": "https://www.bilibili.com/video/BVxxxxxx"}'

# 提交论文 PDF
curl -X POST http://localhost:8000/ingest/paper \
     -H "Content-Type: application/json" \
     -d '{"pdf_url": "https://arxiv.org/pdf/xxxx.xxxx"}'

# 查询进度（替换 job_id）
curl http://localhost:8000/ingest/{job_id}
```

Worker 终端会显示处理进度，完成后数据自动进入检索系统。
