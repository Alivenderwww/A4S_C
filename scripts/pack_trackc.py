#!/usr/bin/env python3
"""pack_trackc.py — 把赛道 C 开发布局装配成官方要求的提交压缩包。

产出结构（详见仓库内 plan）：

    TrackC-<m1>-<m2>-<m3>.zip
    └── TrackC-<m1>-<m2>-<m3>/
        ├── C1/compiler/{aec-cc, src/{Makefile,src,include,tools}}
        ├── C2/{libaec.so, agents/}
        └── C3/{scheduler,runtime,tools,benchmarks,requirements.txt,readme.md}

C1 wrapper 的 root 路径在暂存副本里改写为 ./src（源码已收入 compiler/src/）。
C2/libaec.so 缺失时默认走远程服务器（mig02）构建并回传。
注意：libaec_device.so 不随提交打包；评测框架在包外经 RTLD_GLOBAL 提供。

用法见 --help。仅依赖标准库，跨平台。
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# 让 Windows 控制台正常显示中文
for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trackc_common as tc  # noqa: E402


# ---------------------------------------------------------------------------
# 仓库根定位与子任务源
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    return HERE.parent


def _copy_tree(src: Path, dst: Path) -> None:
    """复制目录树，过滤开发产物（EXCLUDE_DIRS / EXCLUDE_SUFFIXES）。"""
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not tc.is_excluded(d, is_dir=True)]
        rel = os.path.relpath(root, src)
        target = dst if rel == "." else dst / rel
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            if tc.is_excluded(f):
                continue
            shutil.copy2(os.path.join(root, f), target / f)


def _copy_file(src: Path, dst: Path) -> None:
    if tc.is_excluded(src.name):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# C1 暂存
# ---------------------------------------------------------------------------

def stage_c1(c1_src: Path, dst: Path) -> None:
    """C1 -> C1/compiler/{aec-cc(改写), src/{Makefile,src,include,tools}}。"""
    compiler = dst / "compiler"
    compiler.mkdir(parents=True, exist_ok=True)

    # wrapper：改写 root 路径（源码已移入 compiler/src）
    wrapper_path = c1_src / "compiler" / "aec-cc"
    if not wrapper_path.exists():
        raise FileNotFoundError(f"找不到 C1 入口 wrapper：{wrapper_path}")
    text = wrapper_path.read_text(encoding="utf-8")
    # '$(dirname -- "$0")/..'  ->  '$(dirname -- "$0")/src'
    old = '"$(dirname -- "$0")/.."'
    new = '"$(dirname -- "$0")/src"'
    if old not in text:
        print(f"[pack][C1] 警告：wrapper 未匹配到 root 占位 {old!r}，按原样拷贝", file=sys.stderr)
    else:
        text = text.replace(old, new)
    text = text.replace("\r\n", "\n")  # 强制 LF，避免 #!/bin/sh\r
    (compiler / "aec-cc").write_text(text, encoding="utf-8", newline="\n")

    # 源码文件夹：Makefile + src + include + tools
    srcdst = compiler / "src"
    srcdst.mkdir(parents=True, exist_ok=True)
    for item in ("Makefile", "src", "include", "tools"):
        s = c1_src / item
        if not s.exists():
            print(f"[pack][C1] 警告：缺少 {item}（{s}）", file=sys.stderr)
            continue
        d = srcdst / item
        if s.is_dir():
            _copy_tree(s, d)
        else:
            _copy_file(s, d)


# ---------------------------------------------------------------------------
# C2 暂存
# ---------------------------------------------------------------------------

def stage_c2(c2_src: Path, dst: Path) -> None:
    """C2 -> {libaec.so, agents/}。

    注意：libaec_device.so / lib/ 不随提交打包。
    """
    libaec = c2_src / "libaec.so"
    if not libaec.exists() or not tc.is_elf(libaec):
        raise FileNotFoundError(
            f"C2/libaec.so 缺失或非 ELF（{libaec}）。请先用 --build-c2 构建并回传。"
        )
    _copy_file(libaec, dst / "libaec.so")

    agents = c2_src / "agents"
    if agents.is_dir():
        for name in ("dma_agent.py", "kernel_agent.py"):
            a = agents / name
            if a.exists():
                _copy_file(a, dst / "agents" / name)


# ---------------------------------------------------------------------------
# C3 暂存（平铺）
# ---------------------------------------------------------------------------

def stage_c3(c3_src: Path, dst: Path) -> None:
    """C3 -> {src/<框架源码>, requirements.txt, readme.md}。

    仓库布局：C3/src/{scheduler,runtime,tools,benchmarks,tests} + C3/{README.md,requirements.txt}。
    提交同构保留 src/：整树拷入 C3/src/；README.md → 小写 readme.md（命令已为 src/tools/...，
    评测以 C3/ 为工作目录，脚本内部 sys.path 自注入 src/ 使 `from scheduler import` 生效）。
    """
    repo_src = c3_src / "src"
    if repo_src.is_dir():
        _copy_tree(repo_src, dst / "src")
    else:
        print(f"[pack][C3] 警告：缺少 {repo_src}（框架源码目录）", file=sys.stderr)
    req = c3_src / "requirements.txt"
    if req.exists():
        _copy_file(req, dst / "requirements.txt")
    # README.md -> readme.md（小写，大小写敏感）
    readme = c3_src / "README.md"
    if readme.exists():
        data = readme.read_bytes().replace(b"\r\n", b"\n")
        (dst / "readme.md").write_bytes(data)
    else:
        print(f"[pack][C3] 警告：缺少 {readme}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 远程构建 libaec.so（mig02）
# ---------------------------------------------------------------------------

def build_c2_remote(c2_src: Path) -> None:
    """在 mig02 上构建 libaec.so 并回传到 C2/libaec.so。

    流程：推送构建输入（Makefile src/ include/）到服务器临时目录 ->
    ssh make -> scp 回传 libaec.so -> 清理临时目录。
    仅传 KB 级源码 + 小体积 .so，流量极小。

    注意：lib/（设备库）不随提交打包，构建时也不需上传。
    """
    print("[pack][C2] 远程构建 libaec.so ...", file=sys.stderr)
    ts = os.getpid()
    remote_base = f"~/trackc_build_{ts}"
    remote_c2 = f"{remote_base}/C2"

    inputs = []
    for item in ("Makefile", "src", "include"):
        s = c2_src / item
        if s.exists():
            inputs.append(s)
        else:
            print(f"[pack][C2] 警告：远程构建缺少输入 {s}", file=sys.stderr)

    # 1) 建远端目录
    r = tc.remote_mkdir(remote_c2)
    if r.returncode != 0:
        raise RuntimeError(f"远端 mkdir 失败：{r.stderr.strip()}")

    # 2) 推送构建输入
    r = tc.remote_push(inputs, remote_c2 + "/")
    if r.returncode != 0:
        tc.remote_rmtree(remote_base)
        raise RuntimeError(f"推送 C2 构建输入失败：{r.stderr.strip()}")

    # 3) 远端 make
    r = tc.remote_run(f"cd {tc.shell_quote(remote_c2)} && make -j2 2>&1", capture=True, timeout=300)
    print(r.stdout, file=sys.stderr)
    if r.returncode != 0:
        tc.remote_rmtree(remote_base)
        raise RuntimeError(f"远端构建 libaec.so 失败（exit={r.returncode}）")

    # 4) 回传
    local_libaec = c2_src / "libaec.so"
    r = tc.remote_pull(f"{remote_c2}/libaec.so", local_libaec)
    if r.returncode != 0 or not local_libaec.exists():
        tc.remote_rmtree(remote_base)
        raise RuntimeError(f"回传 libaec.so 失败：{r.stderr.strip()}")

    # 5) 清理
    tc.remote_rmtree(remote_base)
    if not tc.is_elf(local_libaec):
        raise RuntimeError(f"回传的 {local_libaec} 非 ELF")
    print(f"[pack][C2] libaec.so 已构建并回传：{local_libaec}", file=sys.stderr)


def build_c2_wsl(c2_src: Path) -> None:
    """备选：本地 WSL 构建 libaec.so。"""
    wsl = shutil.which("wsl") or shutil.which("wsl.exe")
    if not wsl:
        raise RuntimeError("未找到 wsl，无法本地构建 libaec.so。")
    win = c2_src.resolve()
    # 转 wsl 路径
    wslpath = subprocess.run(
        [wsl, "wslpath", "-a", str(win)], capture_output=True, text=True
    )
    if wslpath.returncode != 0:
        raise RuntimeError(f"wslpath 转换失败：{wslpath.stderr.strip()}")
    c2_linux = wslpath.stdout.strip()
    r = subprocess.run([wsl, "bash", "-lc", f"cd {tc.shell_quote(c2_linux)} && make -j2"], text=True)
    if r.returncode != 0 or not (c2_src / "libaec.so").exists():
        raise RuntimeError("WSL 构建 libaec.so 失败")
    print(f"[pack][C2] libaec.so 已由 WSL 构建", file=sys.stderr)


def ensure_libaec(c2_src: Path, mode: str) -> None:
    libaec = c2_src / "libaec.so"
    if libaec.exists() and tc.is_elf(libaec):
        return
    if mode == "none":
        raise RuntimeError("libaec.so 缺失且 --build-c2=none")
    tried = []
    if mode in ("auto", "remote"):
        try:
            build_c2_remote(c2_src)
            return
        except Exception as e:
            tried.append(f"remote: {e}")
            if mode == "remote":
                raise
    if mode in ("auto", "wsl"):
        try:
            build_c2_wsl(c2_src)
            return
        except Exception as e:
            tried.append(f"wsl: {e}")
            if mode == "wsl":
                raise
    raise RuntimeError(
        "libaec.so 构建失败。尝试：\n  " + "\n  ".join(tried)
        + "\n请手动在 Linux 构建后放到 C2/libaec.so 再重跑。"
    )


# ---------------------------------------------------------------------------
# 打包
# ---------------------------------------------------------------------------

def _cleanup_artifacts(root: Path) -> None:
    """清除暂存区可能残留的开发产物。"""
    for dp, dirs, files in os.walk(root):
        for d in list(dirs):
            if d in tc.EXCLUDE_DIRS:
                shutil.rmtree(os.path.join(dp, d), ignore_errors=True)
                dirs.remove(d)
        for f in files:
            if any(f.endswith(s) for s in tc.EXCLUDE_SUFFIXES):
                try:
                    os.remove(os.path.join(dp, f))
                except OSError:
                    pass


def write_zip(stage_root: Path, root_name: str, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    added_dirs: set[str] = set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dp, dirs, files in os.walk(stage_root):
            dirs.sort()
            rel_dir = os.path.relpath(dp, stage_root).replace(os.sep, "/")
            # 显式写入目录条目（保留目录权限位）
            dir_arc = root_name if rel_dir == "." else f"{root_name}/{rel_dir}"
            if dir_arc not in added_dirs:
                tc.add_dir_entry(zf, dir_arc, mode=0o755)
                added_dirs.add(dir_arc)
            for f in sorted(files):
                fp = os.path.join(dp, f)
                arc = f"{root_name}/{rel_dir}/{f}" if rel_dir != "." else f"{root_name}/{f}"
                tc.add_file_entry(zf, Path(fp), arc)
    print(f"[pack] 已写出 {zip_path}", file=sys.stderr)


def cmd_pack(args: argparse.Namespace) -> int:
    members = tc.parse_members(args.members)
    root_name = tc.root_dir_name(members)
    zname = tc.zip_name(members)

    repo = repo_root()
    c1, c2, c3 = repo / "C1", repo / "C2", repo / "C3"
    for p, n in ((c1, "C1"), (c2, "C2"), (c3, "C3")):
        if not p.is_dir():
            print(f"错误：找不到 {n} 源目录 {p}", file=sys.stderr)
            return 2

    # 1) 确保 libaec.so
    ensure_libaec(c2, args.build_c2)

    # 2) 暂存
    with tempfile.TemporaryDirectory(prefix="trackc_pack_") as tmp:
        stage = Path(tmp) / root_name
        stage.mkdir(parents=True)
        print(f"[pack] 暂存装配到 {stage}", file=sys.stderr)
        stage_c1(c1, stage / "C1")
        stage_c2(c2, stage / "C2")
        stage_c3(c3, stage / "C3")
        _cleanup_artifacts(stage)

        # 3) 打包
        out_dir = Path(args.out).resolve()
        zip_path = out_dir / zname
        write_zip(stage, root_name, zip_path)

    size = zip_path.stat().st_size
    over = size > tc.SIZE_LIMIT_BYTES
    print()
    print(f"压缩包：{zip_path}")
    print(f"体  积：{size/1024/1024:.2f} MB {'（超过 100MB 限制！）' if over else ''}")
    print(f"下一步：python scripts/verify_trackc.py --zip \"{zip_path}\"")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="赛道 C 提交打包（产出官方规范压缩包）")
    ap.add_argument("--members", default=None,
                    help='三位成员，逗号分隔："编号1姓名1,编号2姓名2,编号3姓名3"（默认占位）')
    ap.add_argument("--out", default=str(repo_root()), help="输出目录（默认仓库根）")
    ap.add_argument("--build-c2", choices=["auto", "remote", "wsl", "none"], default="auto",
                    help="libaec.so 缺失时的构建方式（默认 auto：先 remote 再 wsl）")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_pack(args)


if __name__ == "__main__":
    sys.exit(main())
