import argparse
import time

import torch
import torch.nn as nn

from common import build_model, load_config, parse_hw, repo_path


def parse_args():
    parser = argparse.ArgumentParser(description="Profile ETMNet parameters, GFLOPs, FPS, and latency.")
    parser.add_argument("--config", required=True, help="Path to ETMNet YAML config.")
    parser.add_argument("--input-size", nargs=2, type=int, default=None, metavar=("H", "W"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", default=None, help="Optional output txt path.")
    return parser.parse_args()


def count_parameters(model):
    return sum(param.numel() for param in model.parameters()) / 1e6


def fallback_flops(model, dummy):
    total = {"flops": 0.0}
    hooks = []

    def conv_hook(module, inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        if out.dim() != 4:
            return
        batch, out_channels, out_h, out_w = out.shape
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels / module.groups
        bias_ops = 1 if module.bias is not None else 0
        total["flops"] += batch * out_channels * out_h * out_w * (kernel_ops + bias_ops)

    def linear_hook(module, inputs, output):
        inp = inputs[0]
        batch = int(inp.shape[0]) if inp.dim() > 1 else 1
        total["flops"] += batch * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(dummy)
    for hook in hooks:
        hook.remove()
    return total["flops"] / 1e9, "conv_linear_hooks"


def compute_gflops(model, dummy):
    try:
        from thop import profile

        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9, "thop"
    except Exception:
        return fallback_flops(model, dummy)


@torch.no_grad()
def measure_speed(model, dummy, warmup, iters, device):
    model.eval()
    for _ in range(int(warmup)):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(int(iters)):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    fps = float(iters) / max(elapsed, 1e-12)
    latency_ms = 1000.0 / max(fps, 1e-12)
    return fps, latency_ms


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_h, input_w = tuple(args.input_size) if args.input_size else parse_hw(config.get("input_size", [512, 512]))
    model = build_model(config).to(device).eval()
    dummy = torch.randn(1, 3, int(input_h), int(input_w), device=device)

    params_m = count_parameters(model)
    gflops, flops_method = compute_gflops(model, dummy)
    fps, latency_ms = measure_speed(model, dummy, args.warmup, args.iters, device)
    lines = [
        "model: ETMNet",
        "device: {}".format(device),
        "input_size: 1x3x{}x{}".format(input_h, input_w),
        "Params(M): {:.6f}".format(params_m),
        "GFLOPs: {:.6f}".format(gflops),
        "GFLOPs_method: {}".format(flops_method),
        "FPS: {:.6f}".format(fps),
        "latency_ms: {:.6f}".format(latency_ms),
        "warmup_iters: {}".format(int(args.warmup)),
        "test_iters: {}".format(int(args.iters)),
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.output:
        output_path = repo_path(args.output)
    else:
        output_path = repo_path(config.get("save_dir", "runs/etmnet")) / "profile_model.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print("Saved profile to {}".format(output_path))


if __name__ == "__main__":
    main()

