# Multi-Agent System

基于 Hermes 单 agent 项目改造的多 agent 版本，用于短视频/带货工作流。

## 目录结构

```
backend/
├── agents/
│   ├── product/           # product_agent - 选品/研究
│   │   ├── config.yaml
│   │   └── system_prompt.txt
│   ├── copy/            # copy_agent - 文案改写
│   │   ├── config.yaml
│   │   └── system_prompt.txt
│   ├── storyboard/       # storyboard_agent - 分镜/提示词
│   │   ├── config.yaml
│   │   └── system_prompt.txt
│   ├── launcher.py      # 统一启动器
│   └── README.md
├── data/              # agent 数据（独立避免串味）
│   ├── product/
│   ├── copy/
│   └── storyboard/
```

## 使用方法

### 1. 列出所有 agent
```bash
cd ~/Downloads/science_teacher_demo_pack
python -m agents.launcher --list
```

### 2. 启动 product_agent
```bash
python -m agents.launcher product "帮我分析一下儿童玩具这个品类"
```

### 3. 启动 copy_agent
```bash
python -m agents.launcher copy "把这篇文案改成短视频口播风格"
```

### 4. 启动 storyboard_agent
```bash
python -m agents.launcher storyboard "把这个文案拆成分镜"
```

### 5. 交互模式
```bash
# 选agent
python -m agents.launcher -i
# 或直接指定
python -m agents.launcher product -i
```

### 6. 继续使用原单 agent（旧版）
```bash
# 启动原有的 Science Teacher
uvicorn backend.app:app --Reload --Port 8000
```

## 新增第 4 个 agent

1. 创建目录：`mkdir agents/my_new_agent`
2. 创建配置：`vi agents/my_new_agent/config.yaml`
3. 创建prompt：`vi agents_my_new_agent/system_prompt.txt`
4. 在 launcher.py 添加配置

## 设计原则

- **独立 memory** - 每个agent有独立data目录，历史不串味
- **独立配置** - config.yaml + system_prompt.txt
- **统一启动** - launcher.py 管理所有agent
- **向后兼容** - 原单agent启动方式不变