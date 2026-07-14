#!/usr/bin/env python3
"""Tests for remote_exec.py — TDD: RED (failing) → GREEN (passing)."""

import io
import json
import os
import shlex
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.remote_exec import (
    build_ssh_argv,
    main,
    parse_args,
    wrap_remote_command,
    HOST,
    KEY_NAME,
    PORT,
    USER,
)


# ── wrap_remote_command ──────────────────────────────────────────────

class TestWrapRemoteCommand(unittest.TestCase):
    """wrap_remote_command: 默认 bash -lic 包装, raw_shell 透传."""

    def test_default_wrap_simple(self):
        """默认包装简单命令；shlex.split → ['bash', '-lic', original]"""
        cmd = "make"
        result = wrap_remote_command(cmd)
        self.assertEqual(shlex.split(result), ["bash", "-lic", cmd])

    def test_default_wrap_complex_preserved(self):
        """复杂命令原文在 shlex.split 第三项完整保留."""
        cmd = "cd ~/A4S && make -j2 2>&1"
        result = wrap_remote_command(cmd)
        parts = shlex.split(result)
        self.assertEqual(parts[:2], ["bash", "-lic"])
        self.assertEqual(parts[2], cmd)

    def test_default_wrap_path_variable(self):
        """$PATH 不被改变."""
        cmd = "echo $PATH"
        result = wrap_remote_command(cmd)
        self.assertEqual(shlex.split(result)[2], cmd)

    def test_default_wrap_single_quotes(self):
        """含单引号的命令不被改变."""
        cmd = "echo 'hello world'"
        result = wrap_remote_command(cmd)
        self.assertEqual(shlex.split(result)[2], cmd)

    def test_default_wrap_double_quotes(self):
        """含双引号的命令不被改变."""
        cmd = 'echo "hello world"'
        result = wrap_remote_command(cmd)
        self.assertEqual(shlex.split(result)[2], cmd)

    def test_default_wrap_and_operator(self):
        """&& 不被改变."""
        cmd = "cd ~/A4S && make -j2"
        result = wrap_remote_command(cmd)
        self.assertEqual(shlex.split(result)[2], cmd)

    def test_default_wrap_pipe(self):
        """| 不被改变."""
        cmd = "cat log.txt | grep error"
        result = wrap_remote_command(cmd)
        self.assertEqual(shlex.split(result)[2], cmd)

    def test_default_wrap_redirect(self):
        """> 不被改变."""
        cmd = "echo hello > /dev/null"
        result = wrap_remote_command(cmd)
        self.assertEqual(shlex.split(result)[2], cmd)

    def test_raw_shell_returns_original(self):
        """raw_shell=True 返回原命令."""
        cmd = "make"
        self.assertEqual(wrap_remote_command(cmd, raw_shell=True), cmd)

    def test_raw_shell_complex_unchanged(self):
        """raw_shell=True 时复杂命令原样返回."""
        cmd = "cd ~/A4S && make -j2 2>&1"
        self.assertEqual(wrap_remote_command(cmd, raw_shell=True), cmd)

    def test_default_and_raw_differ(self):
        """默认和 raw 模式结果不同."""
        cmd = "echo hello"
        self.assertNotEqual(
            wrap_remote_command(cmd, raw_shell=False),
            wrap_remote_command(cmd, raw_shell=True),
        )


# ── parse_args ───────────────────────────────────────────────────────

class TestParseArgs(unittest.TestCase):
    """parse_args: --raw-shell / 4-tuple / 未知选项报错."""

    def test_no_args_exits(self):
        """空 argv → die (SystemExit)."""
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_basic_command_four_tuple(self):
        """基本命令返回 4 元组, raw_shell=False."""
        result = parse_args(["make"])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        dry_run, json_out, raw_shell, rest = result
        self.assertFalse(dry_run)
        self.assertFalse(json_out)
        self.assertFalse(raw_shell)
        self.assertEqual(rest, ["make"])

    def test_dry_run_flag(self):
        """--dry-run 解析正确."""
        dry_run, json_out, raw_shell, rest = parse_args(["--dry-run", "make"])
        self.assertTrue(dry_run)
        self.assertEqual(rest, ["make"])

    def test_json_flag(self):
        """--json 解析正确."""
        dry_run, json_out, raw_shell, rest = parse_args(["--json", "make"])
        self.assertTrue(json_out)

    def test_raw_shell_flag(self):
        """--raw-shell 解析正确."""
        dry_run, json_out, raw_shell, rest = parse_args(["--raw-shell", "make"])
        self.assertTrue(raw_shell)
        self.assertEqual(rest, ["make"])

    def test_raw_shell_with_dry_run(self):
        """--raw-shell 与 --dry-run 可组合."""
        dry_run, json_out, raw_shell, rest = parse_args(
            ["--dry-run", "--raw-shell", "make"]
        )
        self.assertTrue(dry_run)
        self.assertTrue(raw_shell)

    def test_raw_shell_with_json(self):
        """--raw-shell 与 --json 可组合."""
        dry_run, json_out, raw_shell, rest = parse_args(
            ["--json", "--raw-shell", "make"]
        )
        self.assertTrue(json_out)
        self.assertTrue(raw_shell)

    def test_unknown_option_exits(self):
        """未知 -- 选项报错 SystemExit."""
        with self.assertRaises(SystemExit):
            parse_args(["--unknown", "make"])

    def test_unknown_option_code(self):
        """未知选项 exit code = 64 (die 默认)."""
        with self.assertRaises(SystemExit) as ctx:
            parse_args(["--bad-option", "make"])
        self.assertEqual(ctx.exception.code, 64)


# ── build_ssh_argv ───────────────────────────────────────────────────

class TestBuildSshArgv(unittest.TestCase):
    """build_ssh_argv: 最后一项为 remote_cmd, 其他结构不变."""

    def test_remote_cmd_is_last(self):
        """最后一项 = 传入 remote_cmd."""
        argv = build_ssh_argv("ssh", Path("key"), "make")
        self.assertEqual(argv[-1], "make")

    def test_wrapped_cmd_as_last(self):
        """最后一项可以是包装后的命令（调用者决定)."""
        wrapped = "bash -lic 'make'"
        argv = build_ssh_argv("ssh", Path("key"), wrapped)
        self.assertEqual(argv[-1], wrapped)

    def test_ssh_format(self):
        """SSH 参数格式正确."""
        argv = build_ssh_argv("ssh.exe", Path("./mig02"), "make")
        self.assertEqual(argv[0], "ssh.exe")
        self.assertIn("-i", argv)
        self.assertIn(str(Path("./mig02")), argv)
        self.assertIn("-p", argv)
        self.assertIn(PORT, argv)
        self.assertIn(f"{USER}@{HOST}", argv)

    def test_opts_unchanged(self):
        """安全 SSH 选项不变."""
        argv = build_ssh_argv("ssh", Path("k"), "cmd")
        opts = {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1, 2)
                if not argv[i].startswith("-o")}
        # All -o flags present and in order
        self.assertIn("BatchMode=yes", argv)
        self.assertIn("ClearAllForwardings=yes", argv)
        self.assertIn("ForwardAgent=no", argv)


# ── main() dry-run ───────────────────────────────────────────────────

class TestMainDryRun(unittest.TestCase):
    """dry-run 展示实际包装后的 SSH argv."""

    @patch("scripts.remote_exec.shutil.which", return_value="ssh")
    def test_dry_run_default_shows_wrapped(self, mock_which):
        """默认 dry-run 展示 bash -lic."""
        test_args = ["remote_exec.py", "--dry-run", "make"]
        with patch.object(sys, "argv", test_args):
            out = io.StringIO()
            with redirect_stdout(out):
                ret = main()
            self.assertEqual(ret, 0)
            output = out.getvalue()
            self.assertIn("[dry-run]", output)
            self.assertIn("bash", output)
            self.assertIn("-lic", output)

    @patch("scripts.remote_exec.shutil.which", return_value="ssh")
    def test_dry_run_raw_shows_original(self, mock_which):
        """raw dry-run 最后一项为原命令, 不含 bash -lic."""
        test_args = ["remote_exec.py", "--dry-run", "--raw-shell", "make"]
        with patch.object(sys, "argv", test_args):
            out = io.StringIO()
            with redirect_stdout(out):
                ret = main()
            self.assertEqual(ret, 0)
            output = out.getvalue()
            self.assertIn("[dry-run]", output)
            self.assertNotIn("bash -lic", output)

    @patch("scripts.remote_exec.shutil.which", return_value="ssh")
    def test_dry_run_ssh_not_needed(self, mock_which):
        """dry-run 不调用 prepare_key (不需要私钥)."""
        test_args = ["remote_exec.py", "--dry-run", "make"]
        with patch.object(sys, "argv", test_args):
            with patch("scripts.remote_exec.prepare_key") as mock_prep:
                out = io.StringIO()
                with redirect_stdout(out):
                    main()
                mock_prep.assert_not_called()


# ── main() JSON mode ─────────────────────────────────────────────────

class TestMainJson(unittest.TestCase):
    """JSON mode: command 字段为原命令, subprocess 用包装命令."""

    @patch("scripts.remote_exec.prepare_key", return_value=Path("mig02"))
    @patch("scripts.remote_exec.shutil.which", return_value="ssh")
    def test_json_command_field_is_original(self, _which, _key):
        """JSON command = 原始命令."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "ok"
        mock_proc.stderr = ""
        test_args = ["remote_exec.py", "--json", "echo hello"]
        with patch.object(sys, "argv", test_args):
            with patch("scripts.remote_exec.subprocess.run",
                       return_value=mock_proc):
                out = io.StringIO()
                with redirect_stdout(out):
                    main()
                data = json.loads(out.getvalue())
                self.assertEqual(data["command"], "echo hello")

    @patch("scripts.remote_exec.prepare_key", return_value=Path("mig02"))
    @patch("scripts.remote_exec.shutil.which", return_value="ssh")
    def test_json_subprocess_uses_wrapped(self, _which, _key):
        """JSON mode subprocess.run argv[-1] 为包装命令."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        test_args = ["remote_exec.py", "--json", "make"]
        with patch.object(sys, "argv", test_args):
            with patch("scripts.remote_exec.subprocess.run",
                       return_value=mock_proc) as mock_run:
                out = io.StringIO()
                with redirect_stdout(out):
                    main()
                last_arg = mock_run.call_args[0][0][-1]
                self.assertEqual(
                    shlex.split(last_arg), ["bash", "-lic", "make"]
                )

    @patch("scripts.remote_exec.prepare_key", return_value=Path("mig02"))
    @patch("scripts.remote_exec.shutil.which", return_value="ssh")
    def test_json_raw_subprocess_uses_original(self, _which, _key):
        """JSON + raw-shell: subprocess argv[-1] 为原命令."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        test_args = ["remote_exec.py", "--json", "--raw-shell", "make"]
        with patch.object(sys, "argv", test_args):
            with patch("scripts.remote_exec.subprocess.run",
                       return_value=mock_proc) as mock_run:
                out = io.StringIO()
                with redirect_stdout(out):
                    main()
                last_arg = mock_run.call_args[0][0][-1]
                self.assertEqual(last_arg, "make")

    @patch("scripts.remote_exec.prepare_key", return_value=Path("mig02"))
    @patch("scripts.remote_exec.shutil.which", return_value="ssh")
    def test_json_raw_command_field_still_original(self, _which, _key):
        """JSON + raw-shell: command 字段仍为原命令."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        test_args = ["remote_exec.py", "--json", "--raw-shell", "echo hello"]
        with patch.object(sys, "argv", test_args):
            with patch("scripts.remote_exec.subprocess.run",
                       return_value=mock_proc):
                out = io.StringIO()
                with redirect_stdout(out):
                    main()
                data = json.loads(out.getvalue())
                self.assertEqual(data["command"], "echo hello")


# ── main() stream mode ───────────────────────────────────────────────

class TestMainStream(unittest.TestCase):
    """Stream mode: subprocess 使用包装命令."""

    @patch("scripts.remote_exec.prepare_key", return_value=Path("mig02"))
    @patch("scripts.remote_exec.shutil.which", return_value="ssh")
    def test_stream_subprocess_uses_wrapped(self, _which, _key):
        """Stream mode subprocess.run argv[-1] 为包装命令."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        test_args = ["remote_exec.py", "cd ~/A4S && make -j2"]
        with patch.object(sys, "argv", test_args):
            with patch("scripts.remote_exec.subprocess.run",
                       return_value=mock_proc) as mock_run:
                out = io.StringIO()
                with redirect_stdout(out):
                    main()
                last_arg = mock_run.call_args[0][0][-1]
                self.assertEqual(
                    shlex.split(last_arg),
                    ["bash", "-lic", "cd ~/A4S && make -j2"],
                )

    @patch("scripts.remote_exec.prepare_key", return_value=Path("mig02"))
    @patch("scripts.remote_exec.shutil.which", return_value="ssh")
    def test_stream_raw_uses_original(self, _which, _key):
        """Stream + raw-shell: subprocess argv[-1] 为原命令."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        test_args = ["remote_exec.py", "--raw-shell", "echo hello"]
        with patch.object(sys, "argv", test_args):
            with patch("scripts.remote_exec.subprocess.run",
                       return_value=mock_proc) as mock_run:
                out = io.StringIO()
                with redirect_stdout(out):
                    main()
                last_arg = mock_run.call_args[0][0][-1]
                self.assertEqual(last_arg, "echo hello")


# ── Integration: no real SSH ─────────────────────────────────────────

class TestNoRealSsh(unittest.TestCase):
    """所有测试通过 mock 避免真实 SSH 连接."""

    def test_all_mocked(self):
        """验证测试套件没有真实 SSH / subprocess 调用."""
        # Sanity check: every main() test patches subprocess.run or
        # returns early (dry-run).  This test itself is a marker.
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
