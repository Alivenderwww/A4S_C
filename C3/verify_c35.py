import numpy as np, subprocess, os, sys

MODELS = [
    ("MLP", "mlp_v1.onnx", "testdata_mlp", "logits", "input", 0.98),
    ("ResNet", "resnet_v1.onnx", "testdata_resnet", "logits", "input", 0.85),
    ("Transformer", "transformer_v1.onnx", "testdata_tf", "logits", "input_ids", None),
]

print("=" * 60)
results = []
for name, onnx, td, out_name, in_name, gate in MODELS:
    print(f"\n验证 {name} (batch=256, backend=cupy)...")
    out_dir = f"out_v_{name}"
    cmd = ["python", "tools/infer.py", "--onnx", onnx,
           "--input", f"{td}/input", "--output", out_dir,
           "--batch-size", "256", "--backend", "cupy"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [FAIL] 推理失败: {r.stderr.strip()[-200:]}")
        results.append((name, False, None, None)); continue
    out = np.load(f"{out_dir}/{out_name}.npy")
    gold = np.load(f"{td}/golden/{out_name}.npy")
    ac = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
    md = float(np.abs(out - gold).max())
    acc_str, passed = "N/A", ac
    if gate is not None:
        labels = np.load(f"{td}/labels.npy")
        acc = float((out.argmax(-1) == labels).mean())
        acc_str = f"{acc:.4f}/{gate}"
        passed = ac and acc >= gate
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] allclose(1e-3): {ac} | max_diff={md:.2e} | top1={acc_str}")
    results.append((name, passed, f"{md:.2e}", acc_str))

print("\n" + "=" * 60)
print("汇总")
print("=" * 60)
print(f"{'模型':<14}{'精度门槛':<16}{'准确率门槛':<16}{'结果'}")
for name, passed, md, acc in results:
    s = "PASS" if passed else "FAIL"
    print(f"{name:<14}{md or 'FAIL':<16}{acc:<16}{s}")
all_pass = all(p for _, p, _, _ in results)
print(f"\n精度门槛(15分): {'全部通过' if all_pass else '存在失败'}")
sys.exit(0 if all_pass else 1)
