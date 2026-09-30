"""
DeepSeek API 连通性测试模块
—— 独立运行，用于检测 API 配置和调用是否正常

用法：
    python test_deepseek.py

它会依次检查：
    1. 环境变量 / .env 是否读到 API KEY
    2. 是否能连上 DeepSeek 服务器
    3. 非流式调用是否成功
    4. 流式调用是否成功
    5. 消耗的 token 数
"""

import os
import sys
import time
import json

# ---------- 读取 .env ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[1/5] 已加载 .env 文件")
except ImportError:
    print("[1/5] 未安装 python-dotenv（可选），跳过 .env 加载")

try:
    from openai import OpenAI
except ImportError:
    print("\n[错误] 未安装 openai 库，请先执行：")
    print("    pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple")
    sys.exit(1)


# ---------- 配置 ----------
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()


def mask_key(key: str) -> str:
    """隐藏 key 中间部分，用于安全打印"""
    if len(key) <= 10:
        return "*" * len(key)
    return key[:6] + "*" * (len(key) - 10) + key[-4:]


def print_config():
    print("\n" + "=" * 60)
    print("当前配置")
    print("=" * 60)
    print(f"  API KEY     : {mask_key(API_KEY) if API_KEY else '（未设置）'}")
    print(f"  BASE URL    : {BASE_URL}")
    print(f"  模型        : {MODEL}")
    print("=" * 60 + "\n")


def check_key():
    print("[2/5] 检查 API KEY ...")
    if not API_KEY:
        print("\n[错误] 未找到 DEEPSEEK_API_KEY")
        print("\n请选择以下任意一种方式配置：")
        print("\n  方式 A：在项目根目录创建 .env 文件，内容：")
        print("      DEEPSEEK_API_KEY=sk-你的真实key")
        print("      DEEPSEEK_BASE_URL=https://api.deepseek.com")
        print("      DEEPSEEK_MODEL=deepseek-chat")
        print("\n  方式 B：设置环境变量")
        print("      Windows PowerShell:")
        print("        $env:DEEPSEEK_API_KEY=\"sk-你的真实key\"")
        print("      Linux / macOS:")
        print("        export DEEPSEEK_API_KEY=sk-你的真实key")
        print("\n  方式 C：修改本文件顶部的 API_KEY 变量直接填值（不推荐）")
        print()
        sys.exit(1)

    if not API_KEY.startswith("sk-"):
        print(f"  [警告] API KEY 通常以 'sk-' 开头，当前为：{mask_key(API_KEY)}")
        print("         请确认是否填写正确。\n")
    else:
        print(f"  ✓ API KEY 格式正常：{mask_key(API_KEY)}\n")


def test_non_stream(client: OpenAI):
    """测试非流式调用"""
    print("[3/5] 测试非流式调用 ...")
    prompt = "请用一句话回答：什么是饮酒后驾驶机动车？"

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是《道路交通安全法》助手，回答简洁准确。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            stream=False,
        )
        cost = time.time() - t0
        content = resp.choices[0].message.content
        usage = resp.usage

        print(f"  ✓ 调用成功，耗时 {cost:.2f}s")
        print(f"  问题：{prompt}")
        print(f"  回答：{content}")
        if usage:
            print(f"  Token：prompt={usage.prompt_tokens}  "
                  f"completion={usage.completion_tokens}  "
                  f"total={usage.total_tokens}")
        print()
        return True
    except Exception as e:
        print(f"  ✗ 调用失败：{type(e).__name__}: {e}\n")
        return False


def test_stream(client: OpenAI):
    """测试流式调用"""
    print("[4/5] 测试流式调用（逐字输出）...")
    prompt = "请用两句话说明醉酒驾驶的处罚。"

    t0 = time.time()
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stream=True,
        )
        print("  回答：", end="", flush=True)
        full = ""
        chunk_count = 0
        for ev in stream:
            if not ev.choices:
                continue
            delta = ev.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)
                full += delta.content
                chunk_count += 1
        cost = time.time() - t0
        print(f"\n  ✓ 流式调用成功，耗时 {cost:.2f}s，共 {chunk_count} 个数据块\n")
        return True
    except Exception as e:
        print(f"\n  ✗ 流式调用失败：{type(e).__name__}: {e}\n")
        return False


def diagnose_error(e: Exception):
    """根据异常类型给出排错建议"""
    msg = str(e).lower()
    print("=" * 60)
    print("排错建议")
    print("=" * 60)

    if "401" in msg or "authentication" in msg or "invalid api key" in msg:
        print("  · API KEY 无效或已过期。")
        print("    → 请到 https://platform.deepseek.com/api_keys 重新生成。")
    elif "402" in msg or "insufficient" in msg or "balance" in msg:
        print("  · 账户余额不足。")
        print("    → 请到 DeepSeek 平台充值。")
    elif "429" in msg or "rate limit" in msg:
        print("  · 请求过于频繁，触发限流。")
        print("    → 稍后重试，或降低调用频率。")
    elif "timeout" in msg or "timed out" in msg:
        print("  · 请求超时。")
        print("    → 检查网络，或使用代理。")
    elif "connection" in msg or "connect" in msg:
        print("  · 无法连接到服务器。")
        print("    → 检查网络、防火墙、是否需要代理。")
        print(f"    → 尝试浏览器访问 {BASE_URL} 是否可达。")
    elif "model" in msg and "not found" in msg:
        print(f"  · 模型 {MODEL} 不存在。")
        print("    → 可选：deepseek-chat 或 deepseek-reasoner")
    else:
        print(f"  · 未识别的错误，请把以下信息反馈给开发者：")
        print(f"    {type(e).__name__}: {e}")
    print("=" * 60 + "\n")


def main():
    print("\n" + "=" * 60)
    print("  DeepSeek API 连通性测试")
    print("=" * 60)

    print_config()
    check_key()

    # 创建客户端
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        print("[OK] 客户端创建成功\n")
    except Exception as e:
        print(f"[错误] 创建客户端失败：{e}")
        sys.exit(1)

    # 非流式
    ok1 = test_non_stream(client)
    if not ok1:
        try:
            client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
            )
        except Exception as e:
            diagnose_error(e)
        sys.exit(1)

    # 流式
    ok2 = test_stream(client)
    if not ok2:
        sys.exit(1)

    print("[5/5] 全部测试通过 ✓")
    print("\n结论：API 调用正常，可以启动主程序：")
    print("    python app.py")
    print("    浏览器打开 http://localhost:8000\n")


if __name__ == "__main__":
    main()