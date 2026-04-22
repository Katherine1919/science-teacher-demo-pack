# Science Teacher Minimal Runnable Demo Pack

这是一个“最小可运行电脑 Demo 整包”。

## 包含
- backend/app.py：最小 FastAPI 服务
- frontend/index.html：最小前端页面（头像 + 状态 + 字幕 + 假嘴型动画）
- backend/prompts.py
- backend/agent.py
- backend/retriever.py
- backend/update_kb.py
- tests/test_agent_output.py
- tests/test_retriever.py
- starter 知识包

## 先运行
```bash
python backend/update_kb.py
pip install -r requirements.txt
uvicorn backend.app:app --reload
```

如果你用 MiniMax 做 LLM，而不是 OpenAI：
```bash
export LLM_PROVIDER=minimax
export MINIMAX_API_KEY=your_minimax_key
export MINIMAX_BASE_URL=https://api.minimaxi.com/v1
export CHAT_MODEL=MiniMax-M2.5
```

如果你用 DeepSeek 做 LLM：
```bash
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=your_deepseek_key
export DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
export CHAT_MODEL=deepseek-chat
```

浏览器打开：
http://127.0.0.1:8000

## 当前版本说明
- 这是“电脑 Demo 版”
- 前端 Demo 按钮会模拟完整状态流
- 还没有接入真实 wake word / ASR / TTS 音频播放
- 但前后端、字幕和嘴型展示路径已经跑通

## 下一步建议
1. 用 Codex 接 Sprint 0 / Sprint 1 任务
2. 把 demo 触发替换成真实 ASR + Agent + TTS
3. 再迁移到 Pi 5
