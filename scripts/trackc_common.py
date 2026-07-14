#!/usr/bin/env python3
"""trackc_common.py — 赛道 C 打包/验证脚本共享层。

集中维护：
  * 提交规范规则（路径、命名、必需文件）
  * 成员信息 -> 压缩包命名
  * zip 条目 Unix 权限位
  * 远程服务器（mig02）连接助手：复用 scripts/remote_exec.py 的连接配置，
    提供 remote_run / remote_push / remote_pull，供 pack（远程构建 libaec.so）
    与 verify（远程运行时校验）共用。

仅依赖 Python 标准库，跨平台（Windows / Linux / macOS）。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# 提交规范常量（压缩包内相对路径，均以提交根目录为基准）
# ---------------------------------------------------------------------------

SUBTASK_DIRS = ("C1", "C2", "C3")

# C1
C1_AEC_CC = "C1/compiler/aec-cc"          # 编译器统一入口（必需，固定路径）
C1_COMPILER_DIR = "C1/compiler"
C1_SRC_DIR = "C1/compiler/src"            # 源码文件夹（compiler 下独立子目录）
C1_MAKEFILE = "C1/compiler/src/Makefile"
C1_CPP_SRC = "C1/compiler/src/src"        # C++ 源码目录

# C2
C2_LIBAEC = "C2/libaec.so"                # Runtime 动态库（必需，固定路径）
C2_DEVICE_LIB_FORBIDDEN = "C2/lib/libaec_device.so"  # 禁止提交；评测框架在包外经 RTLD_GLOBAL 提供
C2_AGENTS_DIR = "C2/agents"
C2_DMA_AGENT = "C2/agents/dma_agent.py"   # 可选
C2_KERNEL_AGENT = "C2/agents/kernel_agent.py"  # 可选

# C3（框架源码位于 C3/src/ 子文件夹；readme.md 在 C3/ 根，命令为 src/tools/...）
C3_README = "C3/readme.md"                # 小写！大小写敏感
C3_SRC_DIR = "C3/src"                     # 框架源码独立子文件夹
C3_SCHEDULER = "C3/src/scheduler"
C3_RUNTIME = "C3/src/runtime"
C3_TOOLS = "C3/src/tools"

# readme.md 必须包含的命令线索
C31_ONNX_TOKEN = "--onnx"
C31_OUTPUT_TOKEN = "--output"
C35_WORKER_HINTS = ("infer_worker.py", "python")

# 体积上限（GitHub 单文件 100MB；赛道每个压缩包 ≤ 100MB）
SIZE_LIMIT_BYTES = 100 * 1024 * 1024

# 打包时应剔除的目录/后缀（开发机产物）
EXCLUDE_DIRS = {
    "__pycache__", ".pytest_cache", ".git", ".jj", ".zcode", ".idea", ".vscode",
    "obj", "bin", "node_modules",
}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".pyd", ".DS_Store", ".o", ".d", ".exe")

# ---------------------------------------------------------------------------
# 成员信息 / 命名
# ---------------------------------------------------------------------------

DEFAULT_MEMBERS = ["00000000成员1", "00000001成员2", "00000002成员3"]
ZIP_NAME_RE = re.compile(r"^TrackC-(.+)-(.+)-(.+)\.zip$")


def parse_members(members_str: str | None) -> list[str]:
    """把 "ID1姓名1,ID2姓名2,ID3姓名3" 解析成 3 元素列表。"""
    if not members_str:
        return list(DEFAULT_MEMBERS)
    parts = [m.strip() for m in members_str.split(",") if m.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"--members 需要恰好 3 个成员（逗号分隔），收到 {len(parts)} 个：{members_str!r}"
        )
    return parts


def root_dir_name(members: list[str]) -> str:
    return "TrackC-" + "-".join(members)


def zip_name(members: list[str]) -> str:
    return root_dir_name(members) + ".zip"


# ---------------------------------------------------------------------------
# 通用文件/构建产物助手
# ---------------------------------------------------------------------------

def is_excluded(name: str, is_dir: bool = False) -> bool:
    if is_dir and name in EXCLUDE_DIRS:
        return True
    if not is_dir and name in EXCLUDE_DIRS:
        return True
    return any(name.endswith(suf) for suf in EXCLUDE_SUFFIXES)


def is_elf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def is_elf_bytes(data: bytes) -> bool:
    return data[:4] == b"\x7fELF"


# ---------------------------------------------------------------------------
# zip 权限位
# ---------------------------------------------------------------------------

def _zinfo_for(path_in_zip: str, mode: int, is_dir: bool = False) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(path_in_zip)
    zi.create_system = 3                               # Unix origin → unzip restores +x
    if is_dir:
        zi.external_attr = ((0o040000 | (mode & 0o7777)) << 16) | 0x10
    else:
        zi.external_attr = (0o100000 | (mode & 0o7777)) << 16
    zi.file_size = 0
    return zi


def mode_for(relname: str) -> int:
    """按文件类型决定 Unix 权限位。"""
    base = os.path.basename(relname)
    if base == "aec-cc" or relname.endswith(".so"):
        return 0o755          # 入口脚本 / 动态库：评测机需可执行/可加载
    if relname.endswith(".py"):
        return 0o755          # 带 shebang 的 Python：可直接执行也兼容
    return 0o644


def add_dir_entry(zf: zipfile.ZipFile, dir_in_zip: str, mode: int = 0o755) -> None:
    name = dir_in_zip if dir_in_zip.endswith("/") else dir_in_zip + "/"
    zi = _zinfo_for(name, mode, is_dir=True)
    zf.writestr(zi, "")


def add_file_entry(zf: zipfile.ZipFile, src: Path, arcname: str) -> None:
    with open(src, "rb") as f:
        data = f.read()
    zi = _zinfo_for(arcname, mode_for(arcname), is_dir=False)
    zi.file_size = len(data)
    zf.writestr(zi, data)


def zip_mode(zi: zipfile.ZipInfo) -> int:
    """从 ZipInfo 还原 Unix 权限位（低 12 位）。"""
    return (zi.external_attr >> 16) & 0o7777


# ---------------------------------------------------------------------------
# 远程服务器助手（复用 remote_exec.py 连接配置）
# ---------------------------------------------------------------------------

# 远程固定账号（与 scripts/remote_exec.py 一致；此处为单一事实源的本地镜像，
# 实际运行时优先 import remote_exec 的常量，保证不漂移）。
_FALLBACK_HOST = "39.107.68.147"
_FALLBACK_USER = "mig02"
_FALLBACK_PORT = "1102"
_FALLBACK_KEY_NAME = "mig02"


def _remote_config() -> dict:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    cfg = {
        "host": _FALLBACK_HOST,
        "user": _FALLBACK_USER,
        "port": _FALLBACK_PORT,
        "key_name": _FALLBACK_KEY_NAME,
        "prepare_key": None,
    }
    try:
        import remote_exec as re  # type: ignore
        cfg["host"] = re.HOST
        cfg["user"] = re.USER
        cfg["port"] = re.PORT
        cfg["key_name"] = re.KEY_NAME
        cfg["prepare_key"] = getattr(re, "prepare_key", None)
    except Exception:
        pass
    return cfg


def _resolve_key() -> Path:
    """确保 ./mig02 私钥就绪；优先借用 remote_exec.prepare_key。"""
    cfg = _remote_config()
    if cfg["prepare_key"]:
        try:
            return Path(cfg["prepare_key"]()).resolve()
        except SystemExit:
            raise
        except Exception:
            pass
    key = Path(cfg["key_name"]).resolve()
    if key.exists():
        try:
            os.chmod(key, 0o600)
        except OSError:
            pass
        return key
    src = Path.home() / ".ssh" / cfg["key_name"]
    if src.exists():
        shutil.copy(src, key)
        try:
            os.chmod(key, 0o600)
        except OSError:
            pass
        return key
    raise FileNotFoundError(
        f"找不到私钥 {key}，且 {src} 也不存在。请把私钥写入 {key}（Linux/macOS chmod 600）。"
    )


def _ssh_opts() -> list[str]:
    return [
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "NumberOfPasswordPrompts=0",
        "-o", "ClearAllForwardings=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ForwardAgent=no",
        "-o", "ForwardX11=no",
        "-o", "PermitLocalCommand=no",
        "-o", "LogLevel=ERROR",
    ]


def _ssh_target() -> str:
    cfg = _remote_config()
    return f"{cfg['user']}@{cfg['host']}"


def _scp_bin() -> str:
    return shutil.which("scp") or shutil.which("scp.exe") or ""


def _ssh_bin() -> str:
    return shutil.which("ssh") or shutil.which("ssh.exe") or ""


def remote_run(cmd: str, *, capture: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    """在 mig02 上执行一条 shell 命令（经 bash -lic 加载 profile）。

    capture=True 时捕获 stdout/stderr（程序化解析）；False 时实时透传。
    """
    ssh = _ssh_bin()
    if not ssh:
        raise RuntimeError("系统未找到 ssh（请安装 OpenSSH 客户端）。")
    key = _resolve_key()
    cfg = _remote_config()
    wrapped = f"bash -lic {shell_quote(cmd)}"
    argv = [ssh, "-i", str(key), *_ssh_opts(), "-p", cfg["port"], _ssh_target(), wrapped]
    return subprocess.run(argv, capture_output=capture, text=True, timeout=timeout)


def _tar_filter(info):
    """tarfile.add 过滤器：剔除开发产物。"""
    seg = info.name.rstrip("/").split("/")[-1]
    if seg in EXCLUDE_DIRS:
        return None
    if any(seg.endswith(s) for s in EXCLUDE_SUFFIXES):
        return None
    return info


def _ssh_pipe(cmd: str, input_bytes: bytes | None = None, timeout: int | None = None):
    """ssh 执行一条 bash -lc 命令；input_bytes 经 stdin 透传给远端命令。

    stdout 解码为文本（适合控制输出）；二进制内容请用 remote_pull。
    """
    ssh = _ssh_bin()
    if not ssh:
        raise RuntimeError("系统未找到 ssh（请安装 OpenSSH 客户端）。")
    key = _resolve_key()
    cfg = _remote_config()
    wrapped = f"bash -lc {shell_quote(cmd)}"
    argv = [ssh, "-i", str(key), *_ssh_opts(), "-p", cfg["port"], _ssh_target(), wrapped]
    proc = subprocess.run(argv, input=input_bytes, capture_output=True, timeout=timeout)
    return subprocess.CompletedProcess(
        argv, proc.returncode,
        proc.stdout.decode("utf-8", "replace") if proc.stdout else "",
        proc.stderr.decode("utf-8", "replace") if proc.stderr else "",
    )


def remote_push(local_paths: Iterable[str | Path], remote_dir: str) -> subprocess.CompletedProcess:
    """把本地若干路径打包（tarfile），经 ssh 管道解包到远端目录（自动 mkdir -p）。

    用 stdlib tarfile + ssh，规避 Windows/msys2 下 scp 的路径转换问题。
    每个顶层项以 basename 作为远端 arcname（保留其内部结构）。
    """
    import io as _io
    import tarfile as _tarfile
    buf = _io.BytesIO()
    with _tarfile.open(fileobj=buf, mode="w") as tar:
        for p in local_paths:
            p = Path(p)
            tar.add(p, arcname=p.name, filter=_tar_filter)
    data = buf.getvalue()
    cmd = (f"mkdir -p {shell_quote(remote_dir)} && tar -xf - -C {shell_quote(remote_dir)}"
           f" && echo PUSH_OK")
    return _ssh_pipe(cmd, data)


def remote_pull(remote_path: str, local_path: str | Path) -> subprocess.CompletedProcess:
    """经 ssh cat 取远端文件到本地（二进制安全，保留 .so 字节）。"""
    ssh = _ssh_bin()
    if not ssh:
        raise RuntimeError("系统未找到 ssh（请安装 OpenSSH 客户端）。")
    key = _resolve_key()
    cfg = _remote_config()
    wrapped = f"bash -lc {shell_quote('cat ' + shell_quote(remote_path))}"
    argv = [ssh, "-i", str(key), *_ssh_opts(), "-p", cfg["port"], _ssh_target(), wrapped]
    proc = subprocess.run(argv, capture_output=True)
    if proc.returncode == 0:
        lp = Path(local_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_bytes(proc.stdout)
    return subprocess.CompletedProcess(
        argv, proc.returncode, "",
        proc.stderr.decode("utf-8", "replace") if proc.stderr else "",
    )


def shell_quote(s: str) -> str:
    """POSIX 单引号转义（与 remote_exec 的 shlex.quote 行为一致）。"""
    import shlex
    return shlex.quote(s)


def remote_mkdir(path: str) -> subprocess.CompletedProcess:
    return remote_run(f"mkdir -p {shell_quote(path)}")


def remote_rmtree(path: str) -> subprocess.CompletedProcess:
    return remote_run(f"rm -rf {shell_quote(path)}")


# ---------------------------------------------------------------------------
# 终端着色（可选，便于汇总表阅读）
# ---------------------------------------------------------------------------

def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if _supports_color() else s


def red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if _supports_color() else s


def yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if _supports_color() else s


def cyan(s: str) -> str:
    return f"\033[36m{s}\033[0m" if _supports_color() else s
