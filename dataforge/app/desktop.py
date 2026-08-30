"""桌面外壳 —— pywebview + 本地 FastAPI。

选 pywebview 而不是 Electron 的理由见设计文档 §10：核心是 Python 科学栈，
Electron 会引入双运行时和跨进程序列化开销。
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

APP_NAME = "DataForge"


def _free_port() -> int:
    """让系统分配一个空闲端口。

    写死端口会在用户开着两个实例、或本机别的服务占用时直接起不来。
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_ready(port: int, timeout: float = 25.0) -> bool:
    """等服务真的能连上再开窗口，否则会白屏。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.4):
                return True
        except OSError:
            time.sleep(0.15)
    return False


def _icon_path() -> str | None:
    p = Path(__file__).resolve().parents[2] / "assets" / "icon.ico"
    return str(p) if p.exists() else None


def serve(workspace: str, port: int) -> None:
    import uvicorn

    from .server import create_app

    uvicorn.run(
        create_app(workspace),
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataforge-desktop")
    ap.add_argument("--workspace", default="./.dataforge")
    ap.add_argument("--port", type=int, default=0, help="0 = 自动分配")
    ap.add_argument("--no-window", action="store_true", help="只起服务，不开窗口")
    args = ap.parse_args(argv)

    port = args.port or _free_port()
    url = f"http://127.0.0.1:{port}/"

    if args.no_window:
        print(f"{APP_NAME} 服务：{url}")
        serve(args.workspace, port)
        return 0

    threading.Thread(
        target=serve, args=(args.workspace, port), daemon=True, name="dataforge-server"
    ).start()

    if not _wait_ready(port):
        print(f"服务在 25 秒内没有起来（端口 {port}）", file=sys.stderr)
        return 1

    try:
        import webview
    except ImportError:
        print(
            "未安装 pywebview。可以：\n"
            "  pip install pywebview\n"
            f"或直接用浏览器打开：{url}\n"
            "（先加 --no-window 只起服务）",
            file=sys.stderr,
        )
        return 1

    window = webview.create_window(
        APP_NAME, url=url, width=1440, height=940, min_size=(1100, 720),
    )
    kwargs = {}
    icon = _icon_path()
    if icon:
        kwargs["icon"] = icon
    try:
        webview.start(gui="edgechromium", **kwargs)
    except TypeError:
        # 老版本 pywebview 不认 icon 参数
        webview.start(gui="edgechromium")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
