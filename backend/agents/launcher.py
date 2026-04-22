#!/usr/bin/env python3
"""
Multi-Agent Launcher
用法:
  python -m agents.launcher                    # 列出所有 agent
  python -m agents.launcher product         # 启动 product_agent
  python -m agents.launcher copy            # 启动 copy_agent
  python -m agents.launcher storyboard     # 启动 storyboard_agent
  python -m agents.launcher interactive  # 交互模式
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# 添加 backend 路径
AGENT_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = AGENT_ROOT / "agents"
DATA_DIR = AGENT_ROOT / "data"
LATEST_DIR = AGENT_ROOT / "latest"

# 可用 agents
AVAILABLE_AGENTS = {
    "product": {
        "name": "product_agent",
        "description": "选品/研究 Agent - 分析爆品、需求、人群、平台适配",
        "config": "agents/product/config.yaml",
        "prompt": "agents/product/system_prompt.txt",
        "state": "latest/product_state.json",
        "data_dir": "data/product",
    },
    "copy": {
        "name": "copy_agent",
        "description": "文案改写 Agent - 将文案二创改写，口语化适合口播",
        "config": "agents/copy/config.yaml",
        "prompt": "agents/copy/system_prompt.txt",
        "state": "latest/copy_state.json",
        "data_dir": "data/copy",
    },
    "storyboard": {
        "name": "storyboard_agent",
        "description": "分镜/提示词 Agent - 把文案拆成分镜，生成图片提示词",
        "config": "agents/storyboard/config.yaml",
        "prompt": "agents/storyboard/system_prompt.txt",
        "state": "latest/storyboard_state.json",
        "data_dir": "data/storyboard",
    },
}


def list_agents():
    """列出所有可用 agent"""
    print("\n=== 可用 Agents ===\n")
    for key, agent in AVAILABLE_AGENTS.items():
        print(f"  {key:12} - {agent['description']}")
    print()
    return AVAILABLE_AGENTS


def load_system_prompt(agent_key: str) -> str:
    """加载 agent 的 system prompt"""
    prompt_path = AGENT_ROOT / AVAILABLE_AGENTS[agent_key]["prompt"]
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return "你是一个AI Agent助手。"


def load_config(agent_key: str) -> dict:
    """加载 agent 配置"""
    import yaml
    config_path = AGENT_ROOT / AVAILABLE_AGENTS[agent_key]["config"]
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def save_state(agent_key: str, state: dict):
    """保存 agent 状态"""
    state_path = LATEST_DIR / AVAILABLE_AGENTS[agent_key]["state"]
    state["ts"] = datetime.now().isoformat()
    state["agent"] = agent_key
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_state(agent_key: str) -> dict:
    """加载 agent 状态"""
    state_path = LATEST_DIR / AVAILABLE_AGENTS[agent_key]["state"]
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {"status": "idle", "history": []}


async def chat_with_agent(agent_key: str, user_input: str):
    """与指定 agent 对话"""
    agent = AVAILABLE_AGENTS[agent_key]
    system_prompt = load_system_prompt(agent_key)
    state = load_state(agent_key)
    
    print(f"\n=== {agent['name']} ===")
    print(f"System: {system_prompt[:200]}...")
    print(f"\nUser: {user_input}")
    
    # 记录历史
    history = state.get("history", [])
    history.append({
        "role": "user",
        "content": user_input,
        "ts": datetime.now().isoformat()
    })
    
    # 这里以后可以接入真正的 LLM 调用
    # 现在先做一个简单的 echo 响应示例
    import getpass
    try:
        from openai import OpenAI
        client = OpenAI()
        
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-5:]:  # 最近5条
            messages.append(msg)
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        
    except Exception as e:
        reply = f"[模拟回复] 收到: {user_input}\n\n(当前需要配置 OPENAI_API_KEY 才能真正调用 LLM)\n\n错误: {e}"
    
    print(f"\nAgent: {reply}\n")
    
    # 保存历史
    history.append({
        "role": "assistant",
        "content": reply,
        "ts": datetime.now().isoformat()
    })
    save_state(agent_key, {"status": "idle", "history": history})
    
    return reply


def interactive_mode(agent_key: str = None):
    """交互模式"""
    if agent_key:
        print(f"\n启动 {agent_key} ...")
    
    while True:
        try:
            if agent_key:
                user_input = input("\n> ")
            else:
                print("\n=== 选择 Agent ===")
                for i, key in enumerate(AVAILABLE_AGENTS.keys(), 1):
                    print(f"  {i}. {key}")
                print("  0. 退出")
                choice = input("\n请选择: ").strip()
                
                if choice == "0":
                    break
                try:
                    agent_key = list(AVAILABLE_AGENTS.keys())[int(choice) - 1]
                except:
                    continue
            
            user_input = input(f"\n[{agent_key}] > ").strip()
            if not user_input:
                continue
            if user_input in ["exit", "quit", "q"]:
                break
                
            asyncio.run(chat_with_agent(agent_key, user_input))
            
        except KeyboardInterrupt:
            print("\n\n退出")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Launcher")
    parser.add_argument("agent", nargs="?", help="agent名称: product/copy/storyboard")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    parser.add_argument("-l", "--list", action="store_true", help="列出所有agent")
    parser.add_argument("prompt", nargs="*", help="直接输入prompt")
    
    args = parser.parse_args()
    
    # 列出 agents
    if args.list:
        list_agents()
        return
    
    # 指定 agent
    if args.agent:
        if args.agent not in AVAILABLE_AGENTS:
            print(f"未�� agent: {args.agent}")
            print(f"可用: {', '.join(AVAILABLE_AGENTS.keys())}")
            return
        
        # 直接输入 prompt
        if args.prompt:
            user_input = " ".join(args.prompt)
            asyncio.run(chat_with_agent(args.agent, user_input))
            return
        
        # 交互模式
        interactive_mode(args.agent)
        return
    
    # 默认：列出
    list_agents()


if __name__ == "__main__":
    main()