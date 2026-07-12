#!/usr/bin/env python3
"""
remote_exec.py — 在远程服务器（mig02）上执行命令，供 Code Agent 调用。

跨平台（Windows / Linux / macOS），仅依赖 Python 标准库 + 系统 ssh
(Windows 用 ssh.exe，Linux/macOS 用 ssh)。

服务器账号由他人提供，我们仅此一个固定账号：
    主机 39.107.68.147   用户 mig02   端口 1102   私钥 ./mig02
    等价: ssh -i ./mig02 mig02@39.107.68.147 -p 1102

用法见 --help。

⚠️ 服务器约束（务必遵守）：
    1. 公网流量有限 —— 禁止经服务器上传/下载大文件。
    2. 禁止使用隧道 / 端口转发访问互联网（脚本已强制 ClearAllForwardings=yes）。
"""

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

# 远程服务器固定配置（他人提供，我们仅此一个账号）
HOST = "39.107.68.147"
USER = "mig02"
PORT = "1102"
KEY_NAME = "mig02"   # 当前目录下的私钥文件名

# 让中文等 UTF-8 输出在 Windows 控制台也能正常显示（--json 模式尤其需要）
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

HELP = """\
remote_exec.py — 在远程服务器执行命令（供 Agent 调用，跨平台）

固定连接（他人提供的唯一账号）：
  ssh -i ./mig02 mig02@39.107.68.147 -p 1102

用法:
  python remote_exec.py [--dry-run] [--json] <command...>
  python remote_exec.py --help | -h

  command   要在远程执行的命令。含 && / | / 变量 / 重定向 时整体加引号：
              python remote_exec.py "cd ~/A4S && make -j2"

选项:
  --dry-run   只打印将执行的 ssh 命令，不真正连接（不需要私钥）。
  --json      不实时流式输出，结束后向 stdout 打印一个 JSON：
              {"exit": N, "stdout": "...", "stderr": "...", "command": "..."}
              适合 Agent 程序化解析（短命令）。
  -h, --help  显示本帮助。

示例:
  python remote_exec.py uname -a
  python remote_exec.py "cd ~/A4S && make -j2 2>&1"
  python remote_exec.py --dry-run make
  python remote_exec.py --json python3 grader/public_grade.py

私钥准备:
  脚本先找当前目录 ./mig02；找不到则从 ~/.ssh/mig02 复制（自动 chmod 600）。
  也可手动：把私钥内容写入 ./mig02（Linux/macOS 再 chmod 600）。

输出契约（默认流式模式，方便看编译进度）:
  - 远程 stdout → 本脚本 stdout；远程 stderr → 本脚本 stderr（实时）
  - 本脚本退出码 = 远程命令退出码
  - 末行向 stderr 打印: [remote_exec] exit=<N>

⚠ 约束:
  - 公网流量有限，禁止经服务器上传/下载大文件。
  - 禁止隧道/端口转发访问互联网（已强制 ClearAllForwardings=yes）。
"""


def die(msg: str, code: int = 64) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def parse_args(argv):
    """手动解析，保证对 - 开头的远程参数零误判。"""
    dry_run = False
    json_out = False
    rest = []
    for a in argv:
        if a in ("-h", "--help"):
            print(HELP)
            sys.exit(0)
        elif a == "--dry-run":
            dry_run = True
        elif a == "--json":
            json_out = True
        elif a.startswith("--") and a not in ("--dry-run", "--json"):
            die(f"未知选项 {a!r}")
        else:
            rest.append(a)

    if not rest:
        die("缺少要执行的远程命令。用 --help 查看用法。")
    return dry_run, json_out, rest


def prepare_key() -> Path:
    """确保当前目录下存在 ./mig02 私钥；必要时从 ~/.ssh 复制。"""
    key = Path(KEY_NAME).resolve()
    if key.exists():
        try:
            os.chmod(key, 0o600)
        except OSError:
            pass
        return key

    src = Path.home() / ".ssh" / KEY_NAME
    if src.exists():
        shutil.copy(src, key)
        try:
            os.chmod(key, 0o600)
        except OSError:
            pass
        print(f"[remote_exec] 已复制私钥 {src} -> {key}", file=sys.stderr)
        return key

    die(
        f"找不到私钥 {key}，且 {src} 也不存在。\n"
        f"       请把私钥内容写入 {key}（Linux/macOS 再 chmod 600）后重试。",
        code=65,
    )


def build_ssh_argv(ssh_bin: str, key: Path, remote_cmd: str):
    opts = [
        "-o", "BatchMode=yes",                      # 禁止交互式密码提示
        "-o", "StrictHostKeyChecking=accept-new",  # 首次自动信任，主机 key 变化则失败
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "NumberOfPasswordPrompts=0",
        "-o", "ClearAllForwardings=yes",            # 清除一切本地/远程转发（禁止隧道）
        "-o", "ExitOnForwardFailure=yes",           # 任何转发失败立即退出
        "-o", "ForwardAgent=no",
        "-o", "ForwardX11=no",
        "-o", "PermitLocalCommand=no",
        "-o", "LogLevel=ERROR",
    ]
    return [ssh_bin, "-i", str(key), *opts, "-p", PORT, f"{USER}@{HOST}", remote_cmd]


def main() -> int:
    dry_run, json_out, command = parse_args(sys.argv[1:])
    remote_cmd = " ".join(command)

    ssh_bin = shutil.which("ssh") or shutil.which("ssh.exe")
    if not ssh_bin:
        die("系统未找到 ssh（请安装 OpenSSH 客户端）。", code=66)

    if dry_run:
        # dry-run 不需要真实私钥，只用路径占位即可展示将执行的命令
        argv = build_ssh_argv(ssh_bin, Path(KEY_NAME), remote_cmd)
        shown = " ".join(f'"{a}"' if " " in a else a for a in argv)
        print(f"[dry-run] {shown}")
        return 0

    key = prepare_key()
    argv = build_ssh_argv(ssh_bin, key, remote_cmd)

    if json_out:
        import json
        print(f"[remote_exec] → {USER}@{HOST}:{PORT} : {remote_cmd}", file=sys.stderr)
        proc = subprocess.run(argv, capture_output=True, text=True)
        print(json.dumps({
            "exit": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "command": remote_cmd,
        }, ensure_ascii=False))
        return proc.returncode

    # 默认：实时流式输出（适合长编译，能看到进度）
    print(f"[remote_exec] → {USER}@{HOST}:{PORT} : {remote_cmd}", file=sys.stderr)
    proc = subprocess.run(argv)  # 继承父进程 stdout/stderr，实时透传
    print(f"[remote_exec] exit={proc.returncode}", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
