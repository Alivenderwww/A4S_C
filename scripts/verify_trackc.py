#!/usr/bin/env python3
"""verify_trackc.py — 校验赛道 C 提交压缩包是否符合官方规范。

两层校验：

  Layer A（静态，本地，默认即跑）：仅读 zip 字节，校验命名/结构/路径/权限/ELF/readme/体积/洁净度。
  Layer B（运行时，--remote，在 mig02 Linux 上跑）：把 zip 本身上传解压，
           对 C1/C2/C3 各跑一次冒烟测试，验证 build-on-first-use、ELF 加载、Python import 链路。

任一硬性 FAIL 退出码非 0。

用法见 --help。仅依赖标准库，跨平台。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

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
# 校验结果收集
# ---------------------------------------------------------------------------

class Check:
    """单条校验结果。severity: FAIL(硬)/ WARN / PASS / INFO。"""

    def __init__(self, name: str, ok: bool, severity: str, detail: str = ""):
        self.name = name
        self.ok = ok
        self.severity = severity
        self.detail = detail

    def fmt(self) -> str:
        tag = {"PASS": tc.green("PASS"), "FAIL": tc.red("FAIL"),
               "WARN": tc.yellow("WARN"), "INFO": tc.cyan("INFO")}[self.severity]
        line = f"  [{tag}] {self.name}"
        if self.detail:
            line += f"  — {self.detail}"
        return line


class Reporter:
    def __init__(self):
        self.items: list[Check] = []

    def add(self, name: str, ok: bool, severity: str, detail: str = "") -> None:
        self.items.append(Check(name, ok, severity, detail))

    def pass_(self, name, detail=""):
        self.add(name, True, "PASS", detail)

    def fail(self, name, detail=""):
        self.add(name, False, "FAIL", detail)

    def warn(self, name, detail=""):
        self.add(name, ok=True, severity="WARN", detail=detail)

    def info(self, name, detail=""):
        self.add(name, ok=True, severity="INFO", detail=detail)

    @property
    def hard_fail(self) -> bool:
        return any(it.severity == "FAIL" for it in self.items)

    def print(self) -> None:
        print()
        cur = None
        for it in self.items:
            if it.name and it.name.startswith("== "):
                print(it.fmt())
                cur = it
                continue
            print(it.fmt())
        fails = sum(1 for it in self.items if it.severity == "FAIL")
        warns = sum(1 for it in self.items if it.severity == "WARN")
        print()
        verdict = tc.red("FAIL") if fails else tc.green("PASS")
        print(f"汇总：{verdict}  硬性失败 {fails}，警告 {warns}")


# ---------------------------------------------------------------------------
# zip 读取辅助
# ---------------------------------------------------------------------------

class ZipView:
    def __init__(self, zip_path: Path):
        self.path = zip_path
        self.zf = zipfile.ZipFile(zip_path)
        self.names = set(self.zf.namelist())
        # 顶层目录（第一个路径段）
        self.tops = {n.split("/")[0] for n in self.zf.namelist() if n and not n.startswith("/")}

    def close(self):
        self.zf.close()

    def has(self, name: str) -> bool:
        # 同时容忍带尾斜杠的目录条目与文件条目
        return name in self.names or name.rstrip("/") in self.names or (name + "/") in self.names

    def member(self, name: str) -> zipfile.ZipInfo | None:
        for cand in (name, name.rstrip("/")):
            if cand in self.zf.NameToInfo:
                return self.zf.NameToInfo[cand]
        return None

    def exists_file(self, name: str) -> bool:
        zi = self.member(name)
        return zi is not None and not zi.is_dir()

    def read(self, name: str, limit: int | None = None) -> bytes:
        data = self.zf.read(name)
        return data if limit is None else data[:limit]

    def any_under(self, prefix: str) -> bool:
        pfx = prefix.rstrip("/") + "/"
        return any(n.startswith(pfx) for n in self.names)


# ---------------------------------------------------------------------------
# Layer A：静态校验
# ---------------------------------------------------------------------------

def layer_a(zip_path: Path, rep: Reporter) -> None:
    rep.add("== Layer A：静态校验（本地） ==", True, "INFO")
    name = zip_path.name
    stem = zip_path.stem

    # 命名
    m = tc.ZIP_NAME_RE.match(name)
    rep.add("压缩包命名 TrackC-...-...-...zip", bool(m), "FAIL" if not m else "PASS",
            name if m else f"{name} 不匹配规范")

    zv = ZipView(zip_path)
    try:
        # 顶层结构
        tops = {n.split("/")[0] for n in zv.names if n and not n.startswith("/")}
        # 只允许一个顶层目录，且与 zip stem 一致
        if len(tops) == 1 and stem in tops:
            rep.pass_("唯一顶层目录与 zip 同名", stem)
        else:
            rep.fail("顶层目录", f"期望单个 {stem}，实际 {sorted(tops)}")

        root = stem

        def rp(p: str) -> str:
            return f"{root}/{p}"

        # 三个子任务目录
        for sub in tc.SUBTASK_DIRS:
            rep.add(f"存在 {sub}/", zv.has(rp(sub)),
                    "FAIL" if not zv.has(rp(sub)) else "PASS")

        # ---- C1 ----
        rep.add("== C1 ==", True, "INFO")
        aec = rp(tc.C1_AEC_CC)
        if zv.exists_file(aec):
            zi = zv.member(aec)
            mode = tc.zip_mode(zi) if zi is not None else 0
            head = zv.read(aec, limit=2)
            rep.add("aec-cc 存在", True, "PASS")
            rep.add("aec-cc 可执行位 (+x)", bool(mode & 0o111),
                    "PASS" if mode & 0o111 else "FAIL", f"mode={oct(mode)}")
            rep.add("aec-cc ZIP 来源 Unix（create_system==3）",
                    zi is not None and zi.create_system == 3,
                    "PASS" if zi is not None and zi.create_system == 3 else "FAIL",
                    f"create_system={zi.create_system if zi else 'N/A'}")
            rep.add("aec-cc shebang (#!)", head[:2] == b"#!",
                    "PASS" if head[:2] == b"#!" else "FAIL")
            # wrapper root 指向 ./src
            body = zv.read(aec).decode("utf-8", "replace")
            rep.add("wrapper root 指向 ./src", '"$(dirname -- "$0")/src"' in body,
                    "PASS" if '"$(dirname -- "$0")/src"' in body else "FAIL")
        else:
            rep.fail("C1/compiler/aec-cc 存在", aec)

        rep.add("C1/compiler/src/Makefile", zv.exists_file(rp(tc.C1_MAKEFILE)),
                "PASS" if zv.exists_file(rp(tc.C1_MAKEFILE)) else "FAIL")
        rep.add("C1/compiler/src/src/ (C++ 源码)", zv.has(rp(tc.C1_CPP_SRC)),
                "PASS" if zv.has(rp(tc.C1_CPP_SRC)) else "FAIL")
        rep.add("C1/compiler/src/include/", zv.has(rp(tc.C1_SRC_DIR + "/include")),
                "WARN" if not zv.has(rp(tc.C1_SRC_DIR + "/include")) else "PASS")

        # ---- C2 ----
        rep.add("== C2 ==", True, "INFO")
        lib = rp(tc.C2_LIBAEC)
        if zv.exists_file(lib):
            magic = zv.read(lib, limit=4)
            zi = zv.member(lib)
            mode = tc.zip_mode(zi) if zi is not None else 0
            rep.add("libaec.so 存在", True, "PASS")
            rep.add("libaec.so ELF", tc.is_elf_bytes(magic),
                    "PASS" if tc.is_elf_bytes(magic) else "FAIL")
            rep.add("libaec.so 可执行位 (+x)", bool(mode & 0o111),
                    "PASS" if mode & 0o111 else "FAIL", f"mode={oct(mode)}")
        else:
            rep.fail("C2/libaec.so 存在", lib)

        dev = rp(tc.C2_DEVICE_LIB_FORBIDDEN)
        if zv.exists_file(dev):
            rep.fail("C2/lib/libaec_device.so（禁止提交）",
                     "设备库由评测框架在包外注入；提交中不得包含")
        else:
            rep.pass_("C2/lib/libaec_device.so（未提交）")

        dma = rp(tc.C2_DMA_AGENT)
        ker = rp(tc.C2_KERNEL_AGENT)
        rep.add("C2/agents/dma_agent.py", zv.exists_file(dma),
                "INFO" if not zv.exists_file(dma) else "PASS",
                "可选项" if not zv.exists_file(dma) else "")
        rep.add("C2/agents/kernel_agent.py", zv.exists_file(ker),
                "INFO" if not zv.exists_file(ker) else "PASS",
                "可选项" if not zv.exists_file(ker) else "")

        # ---- C3 ----
        rep.add("== C3 ==", True, "INFO")
        readme = rp(tc.C3_README)
        readme_upper = rp("C3/README.md")
        if zv.exists_file(readme):
            rep.add("C3/readme.md (小写)", True, "PASS")
            body = zv.read(readme).decode("utf-8", "replace")
            rep.add("readme 含 C3.1 模板 (--onnx/--output)",
                    (tc.C31_ONNX_TOKEN in body and tc.C31_OUTPUT_TOKEN in body),
                    "PASS" if (tc.C31_ONNX_TOKEN in body and tc.C31_OUTPUT_TOKEN in body) else "FAIL")
            c35 = ("infer_worker.py" in body)
            rep.add("readme 含 C3.5 启动命令", c35,
                    "PASS" if c35 else "FAIL")
        else:
            rep.fail("C3/readme.md (小写)", f"未找到 {readme}" +
                     ("（注意有 README.md 大写）" if zv.exists_file(readme_upper) else ""))

        for label, path in (("scheduler", tc.C3_SCHEDULER), ("runtime", tc.C3_RUNTIME), ("tools", tc.C3_TOOLS)):
            rep.add(f"C3/{label}/", zv.has(rp(path)),
                    "FAIL" if not zv.has(rp(path)) else "PASS")

        # ---- 体积 / 洁净度 ----
        rep.add("== 体积 / 洁净度 ==", True, "INFO")
        size = zip_path.stat().st_size
        rep.add(f"体积 ≤ 100MB ({size/1024/1024:.2f} MB)", size <= tc.SIZE_LIMIT_BYTES,
                "PASS" if size <= tc.SIZE_LIMIT_BYTES else "WARN")

        banned = []
        for n in zv.names:
            low = n.lower()
            seg = low.rsplit("/", 1)[-1]
            if seg in ("__pycache__",) or seg.endswith((".pyc", ".ds_store", ".o", ".d")):
                banned.append(n)
            elif "/bin/" in low or low.endswith("/bin") or "/obj/" in low:
                if any(low.endswith(s) for s in ("/aec-cc", ".so")):
                    continue
                banned.append(n)
        rep.add("无开发产物 (__pycache__/*.pyc/.DS_Store/obj/bin)",
                not banned, "PASS" if not banned else "WARN",
                ("命中: " + ", ".join(banned[:8])) if banned else "")
    finally:
        zv.close()


# ---------------------------------------------------------------------------
# Layer B：远程运行时校验（mig02 / Linux）
# ---------------------------------------------------------------------------

def _find_remote_zip_target(rep: Reporter) -> str | None:
    """确认 ssh/scp 可用并返回 ssh target 字符串；不可用则记 FAIL。"""
    if not tc._ssh_bin():
        rep.fail("远程运行时校验", "未找到 ssh 客户端")
        return None
    try:
        tc._resolve_key()
    except Exception as e:
        rep.fail("私钥就绪", str(e))
        return None
    return tc._ssh_target()


def layer_b(zip_path: Path, rep: Reporter, remote_public: str | None) -> None:
    rep.add("== Layer B：远程运行时校验（mig02 / Linux） ==", True, "INFO")
    if _find_remote_zip_target(rep) is None:
        return

    ts = os.getpid()
    remote_base = f"/tmp/trackc_verify_{ts}"
    r = tc.remote_mkdir(remote_base)
    if r.returncode != 0:
        rep.fail("远端工作目录创建", r.stderr.strip() or f"exit={r.returncode}")
        return

    # 1) 上传 zip
    r = tc.remote_push([zip_path], remote_base + "/")
    if r.returncode != 0:
        rep.fail("上传 zip 到服务器", r.stderr.strip())
        tc.remote_rmtree(remote_base)
        return
    rep.pass_("上传 zip 到服务器", f"{zip_path.name} -> {remote_base}")

    remote_zip = f"{remote_base}/{zip_path.name}"
    # 2) 检查 unzip 可用（必须用 unzip 才能从 ZIP 条目中恢复 Unix 权限位）
    r = tc.remote_run("unzip -v >/dev/null 2>&1", capture=True)
    if r.returncode != 0:
        rep.fail("远端解压 zip", "系统缺少 unzip，无法还原 Unix 文件权限")
        tc.remote_rmtree(remote_base)
        return

    # 3) 用 unzip 解压（自动恢复 create_system=3 的 Unix 权限位）
    r = tc.remote_run(
        f"cd {tc.shell_quote(remote_base)} && unzip -q {tc.shell_quote(zip_path.name)} 2>&1",
        capture=True, timeout=120,
    )
    if r.returncode != 0:
        rep.fail("远端解压 zip", r.stderr.strip() or f"exit={r.returncode}")
        tc.remote_rmtree(remote_base)
        return
    # 找到解压根
    rls = tc.remote_run(f"cd {tc.shell_quote(remote_base)} && ls -1", capture=True)
    roots = [x for x in rls.stdout.splitlines() if x.strip() and x.strip() != zip_path.name]
    if len(roots) != 1:
        rep.fail("解压根目录唯一", f"ls: {rls.stdout.strip()}")
        tc.remote_rmtree(remote_base)
        return
    root_remote = f"{remote_base}/{roots[0]}"
    rep.pass_("远端解压", roots[0])

    # unzip 已从 ZIP 中恢复 create_system=3 条目的 Unix 权限
    # 对提交的 libaec.so 额外 chmod +x 以防 umask 过滤
    tc.remote_run(
        " ".join([
            "chmod", "+x",
            tc.shell_quote(root_remote + "/C2/libaec.so"),
            "2>/dev/null", ";", "true",
        ])
    )

    # ---- 上传官方参考 libaec_device.so（包外提供，非提交内容） ----
    official_dev = HERE.parent / "public" / "Track-C" / "C2-runtime" / "starter-kit" / "lib" / "libaec_device.so"
    if not official_dev.exists():
        rep.fail("C2 官方参考设备库",
                 f"本地缺失 {official_dev}，请确认 public/Track-C/ 完整")
        tc.remote_rmtree(remote_base)
        return
    r = tc.remote_push([official_dev], remote_base + "/official-reference/")
    if r.returncode != 0:
        rep.fail("上传官方设备库", r.stderr.strip() or f"exit={r.returncode}")
        tc.remote_rmtree(remote_base)
        return
    official_dev_remote = f"{remote_base}/official-reference/libaec_device.so"

    # ---- C1 冒烟：build-on-first-use + selftest ----
    rep.add("== C1 运行时（compiler/aec-cc --selftest） ==", True, "INFO")
    r = tc.remote_run(
        f"cd {tc.shell_quote(root_remote + '/C1/compiler')} && ./aec-cc --selftest 2>&1",
        capture=True, timeout=600,
    )
    rep.add("C1 aec-cc 构建+selftest", r.returncode == 0,
            "PASS" if r.returncode == 0 else "FAIL",
            (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"exit={r.returncode}"))

    # ---- C2 冒烟：libaec.so 可加载（先 RTLD_GLOBAL 加载官方设备库） ----
    rep.add("== C2 运行时（libaec.so 加载） ==", True, "INFO")
    c2root = f"{root_remote}/C2"
    r = tc.remote_run(
        f"python3 -c \"import ctypes,sys; "
        f"ctypes.CDLL(sys.argv[1],mode=ctypes.RTLD_GLOBAL); "
        f"ctypes.CDLL(sys.argv[2]); print('loaded')\" "
        f"{tc.shell_quote(official_dev_remote)} {tc.shell_quote(c2root + '/libaec.so')}",
        capture=True, timeout=120,
    )
    rep.add("C2 libaec.so dlopen（官方设备库 + RTLD_GLOBAL）", r.returncode == 0,
            "PASS" if r.returncode == 0 else "FAIL",
            (r.stdout.strip() or r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"exit={r.returncode}"))

    # 可选：公共 grader（在公共资料里，不在 zip 内）
    grader = _resolve_remote_grader(remote_public)
    if grader:
        r = tc.remote_run(
            f"python3 {tc.shell_quote(grader)} --submission {tc.shell_quote(c2root)} --profile public 2>&1",
            capture=True, timeout=600,
        )
        rep.add("C2 public_grade.py (可选)", r.returncode == 0,
                "PASS" if r.returncode == 0 else "WARN",
                (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"exit={r.returncode}"))
    else:
        rep.warn("C2 public_grade.py (可选)", "服务器未定位到公共 grader，已跳过")

    # ---- C3 冒烟：export_dag + worker READY ----
    rep.add("== C3 运行时（export_dag / worker） ==", True, "INFO")
    c3root = f"{root_remote}/C3"
    rdag = f"{remote_base}/dag.json"
    model = _resolve_remote_model(remote_public)
    model_src = "公共模型"
    if not model:
        if _gen_probe_onnx(f"{remote_base}/probe.onnx"):
            model = f"{remote_base}/probe.onnx"
            model_src = "生成的探针模型"
    if model:
        r = tc.remote_run(
            f"cd {tc.shell_quote(c3root)} && python3 src/tools/export_dag.py --onnx {tc.shell_quote(model)} "
            f"--output {tc.shell_quote(rdag)} 2>&1",
            capture=True, timeout=300,
        )
        ok = r.returncode == 0
        jdetail = ""
        if ok:
            jok, jdetail = _check_json_valid(rdag)
            ok = jok
        rep.add(f"C3.1 export_dag [{model_src}]", ok, "PASS" if ok else "FAIL",
                (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"exit={r.returncode}")
                + (f" | json: {jdetail}" if jdetail else ""))
    else:
        rep.warn("C3.1 export_dag (可选)", "服务器无 onnx 模型且无法生成探针模型，已跳过")

    r = tc.remote_run(
        f"cd {tc.shell_quote(c3root)} && printf '%s\\n' '{{\"cmd\":\"exit\"}}' | python3 src/tools/infer_worker.py 2>&1",
        capture=True, timeout=120,
    )
    ready = r.returncode == 0 and "READY" in r.stdout
    rep.add("C3.5 worker READY", ready, "PASS" if ready else "FAIL",
            (r.stdout.strip().splitlines()[0] if r.stdout.strip() else f"exit={r.returncode}"))

    # 清理
    tc.remote_rmtree(remote_base)


def _resolve_remote_model(remote_public: str | None) -> str | None:
    base = remote_public or "~/A4S/public"
    r = tc.remote_run(
        f"find {tc.shell_quote(base)} -maxdepth 6 -name '*.onnx' 2>/dev/null | head -1",
        capture=True, timeout=60,
    )
    m = (r.stdout or "").strip().splitlines()
    return m[0] if m else None


def _gen_probe_onnx(remote_path: str) -> bool:
    """在服务器上用 onnx 生成一个最小合法模型（单 Relu 节点），供 C3.1 端到端验证。

    脚本经 base64 管道送入远端 python，规避嵌套引号问题。
    """
    import base64
    script = (
        "import sys,onnx;"
        "from onnx import helper,TensorProto,checker;"
        "X=helper.make_tensor_value_info('X',TensorProto.FLOAT,[1,4]);"
        "Y=helper.make_tensor_value_info('Y',TensorProto.FLOAT,[1,4]);"
        "n=helper.make_node('Relu',['X'],['Y']);"
        "g=helper.make_graph([n],'g',[X],[Y]);"
        "m=helper.make_model(g,producer_name='probe');"
        "checker.check_model(m);"
        "onnx.save(m,sys.argv[1]);print('GEN_OK')"
    )
    b64 = base64.b64encode(script.encode()).decode()
    r = tc.remote_run(
        f"echo {b64} | base64 -d | python3 - {tc.shell_quote(remote_path)}",
        capture=True, timeout=60,
    )
    return r.returncode == 0 and "GEN_OK" in (r.stdout or "")


def _check_json_valid(path: str) -> tuple[bool, str]:
    """远端校验 JSON 合法（base64 管道，规避嵌套引号）。返回 (ok, detail)。"""
    import base64
    script = "import json,sys;json.load(open(sys.argv[1]));print('JSON_OK')"
    b64 = base64.b64encode(script.encode()).decode()
    r = tc.remote_run(f"echo {b64} | base64 -d | python3 - {tc.shell_quote(path)}", capture=True, timeout=60)
    ok = r.returncode == 0 and "JSON_OK" in (r.stdout or "")
    src = (r.stdout or "") or (r.stderr or "")
    detail = src.strip().splitlines()[-1] if src.strip() else f"exit={r.returncode}"
    return ok, detail


def _resolve_remote_grader(remote_public: str | None) -> str | None:
    base = remote_public or "~/A4S/public"
    r = tc.remote_run(
        f"find {tc.shell_quote(base)} -maxdepth 7 -path '*grader/public_grade.py' 2>/dev/null | head -1",
        capture=True, timeout=60,
    )
    m = (r.stdout or "").strip().splitlines()
    return m[0] if m else None


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def find_latest_zip(out_dir: Path) -> Path | None:
    zips = sorted(out_dir.glob("TrackC-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="校验赛道 C 提交压缩包（静态 + 远程运行时）")
    ap.add_argument("--zip", help="待校验的 zip 路径（缺省自动找仓库内最新 TrackC-*.zip）")
    ap.add_argument("--remote", action="store_true", help="追加 Layer B 远程 Linux 运行时校验")
    ap.add_argument("--remote-public", default=None,
                    help="服务器公共资料路径（含 onnx 模型 / grader），默认 ~/A4S/public")
    args = ap.parse_args(argv)

    zip_path = Path(args.zip) if args.zip else find_latest_zip(Path.cwd())
    if not zip_path or not zip_path.exists():
        print(f"错误：找不到 zip（{args.zip or '自动查找'}）", file=sys.stderr)
        return 2

    print(f"校验对象：{zip_path}  ({zip_path.stat().st_size/1024/1024:.2f} MB)")

    rep = Reporter()
    layer_a(zip_path, rep)

    do_remote = args.remote
    if do_remote and rep.hard_fail:
        print("\n静态校验存在硬性 FAIL，跳过远程运行时校验（先修复结构）。", file=sys.stderr)
        do_remote = False
    if do_remote:
        try:
            layer_b(zip_path, rep, args.remote_public)
        except Exception as e:
            rep.fail("远程运行时校验", f"异常：{type(e).__name__}: {e}")

    rep.print()
    return 1 if rep.hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
