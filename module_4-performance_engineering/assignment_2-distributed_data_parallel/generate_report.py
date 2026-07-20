"""Generate the minimal required DDP graphs and Markdown report from saved logs."""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).parent
LOGS = ROOT / "logs"
GRAPHS = ROOT / "graphs"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def clean_log(filename: str) -> str:
    return ANSI.sub("", (LOGS / filename).read_text(errors="replace")).replace(
        "\r", "\n"
    )


def value(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}:\s+([^\n]+)", text)
    if not match:
        raise ValueError(f"Missing {label}")
    return match.group(1).strip()


def trainer_value(text: str, label: str) -> str:
    match = re.search(rf"'{re.escape(label)}':\s*'([^']+)'", text)
    if not match:
        raise ValueError(f"Missing trainer metric {label}")
    return match.group(1)


def losses(text: str) -> list[tuple[int, float]]:
    points = []
    for loss, step in re.findall(r"'loss':\s*'([0-9.]+)'.*?(\d+)/500", text, re.DOTALL):
        pair = (int(step), float(loss))
        if not points or points[-1] != pair:
            points.append(pair)
    if not points:
        raise ValueError("No loss points found")
    return points


def svg_chart(
    filename: str,
    title: str,
    x_label: str,
    series: list[tuple[str, list[tuple[float, float]]]],
) -> None:
    width, height, pad = 900, 520, 70
    all_points = [point for _, points in series for point in points]
    max_x = max(x for x, _ in all_points)
    min_y = min(y for _, y in all_points)
    max_y = max(y for _, y in all_points)
    y_span = max(max_y - min_y, 0.1)
    colors = ["#2563eb", "#dc2626"]

    def px(x: float) -> float:
        return pad + x / max_x * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - min_y) / y_span * (height - 2 * pad)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="black"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="black"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif">{html.escape(x_label)}</text>',
        f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-family="sans-serif">Training loss</text>',
    ]
    for i in range(6):
        x = max_x * i / 5
        y = min_y + y_span * i / 5
        lines.extend(
            [
                f'<line x1="{px(x):.1f}" y1="{height - pad}" x2="{px(x):.1f}" y2="{height - pad + 5}" stroke="black"/>',
                f'<text x="{px(x):.1f}" y="{height - pad + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{x:.0f}</text>',
                f'<line x1="{pad - 5}" y1="{py(y):.1f}" x2="{pad}" y2="{py(y):.1f}" stroke="black"/>',
                f'<text x="{pad - 10}" y="{py(y) + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{y:.2f}</text>',
            ]
        )
    for index, (name, points) in enumerate(series):
        path = " ".join(
            f"{'M' if i == 0 else 'L'} {px(x):.1f} {py(y):.1f}"
            for i, (x, y) in enumerate(points)
        )
        y = 55 + index * 22
        lines.extend(
            [
                f'<path d="{path}" fill="none" stroke="{colors[index]}" stroke-width="2"/>',
                f'<line x1="{width - 230}" y1="{y}" x2="{width - 205}" y2="{y}" stroke="{colors[index]}" stroke-width="3"/>',
                f'<text x="{width - 200}" y="{y + 4}" font-family="sans-serif" font-size="13">{html.escape(name)}</text>',
            ]
        )
    lines.append("</svg>")
    (GRAPHS / filename).write_text("\n".join(lines))


def main() -> None:
    GRAPHS.mkdir(exist_ok=True)
    one, four = clean_log("1gpu_log.txt"), clean_log("4gpu_log.txt")
    one_losses, four_losses = losses(one), losses(four)
    one_runtime, four_runtime = (
        float(trainer_value(one, "train_runtime")),
        float(trainer_value(four, "train_runtime")),
    )
    svg_chart(
        "loss_vs_steps.svg",
        "Loss vs. training steps",
        "Optimizer step",
        [("1 GPU", one_losses), ("4 GPUs", four_losses)],
    )
    svg_chart(
        "loss_vs_time.svg",
        "Loss vs. training wall-clock time",
        "Training time (seconds)",
        [
            ("1 GPU", [(step / 500 * one_runtime, loss) for step, loss in one_losses]),
            (
                "4 GPUs",
                [(step / 500 * four_runtime, loss) for step, loss in four_losses],
            ),
        ],
    )
    one_summary = {
        "runtime": one_runtime,
        "samples": trainer_value(one, "train_samples_per_second"),
        "steps": trainer_value(one, "train_steps_per_second"),
        "eval_samples": value(one, "[Inference]   eval_samples_per_second"),
        "eval_runtime": value(one, "[Inference]   eval_runtime"),
        "eval_loss": value(one, "[Inference]   eval_loss"),
        "comm_total": value(one, "[Comm]   measured total comm time (whole run)"),
        "comm_bytes": value(one, "[Comm]   measured total bytes communicated"),
        "comm_call": value(one, "[Comm]   measured avg comm time / all-reduce call"),
        "comm_step": value(one, "[Comm]   measured avg comm time / optimizer step"),
    }
    four_summary = {
        "runtime": four_runtime,
        "samples": trainer_value(four, "train_samples_per_second"),
        "steps": trainer_value(four, "train_steps_per_second"),
        "eval_samples": value(four, "[Inference]   eval_samples_per_second"),
        "eval_runtime": value(four, "[Inference]   eval_runtime"),
        "eval_loss": value(four, "[Inference]   eval_loss"),
        "comm_total": value(four, "[Comm]   measured total comm time (whole run)"),
        "comm_bytes": value(four, "[Comm]   measured total bytes communicated"),
        "comm_call": value(four, "[Comm]   measured avg comm time / all-reduce call"),
        "comm_step": value(four, "[Comm]   measured avg comm time / optimizer step"),
    }
    report = f"""# Scaling GPT-2 Large training with PyTorch DDP

## Setup and batch-size result
`train.py` configures GPT-2 Large with 36 layers, hidden size 1280, 20 heads, and 1024 positions. The run log reports 774,030,080 trainable parameters. Power-of-two single-H100 probes found batch 32 can complete a short probe but OOMs during normal training when logits are converted to fp32; batch 16 completed the full 500-step run. Therefore batch 16 is the stable batch size used for both comparison runs.

## Loss curves
![Loss vs. steps](graphs/loss_vs_steps.svg)

![Loss vs. time](graphs/loss_vs_time.svg)

## Training performance
| Metric | 1 GPU | 4 GPUs |
|---|---:|---:|
| train runtime (s) | {one_summary["runtime"]:.1f} | {four_summary["runtime"]:.1f} |
| train samples/s | {one_summary["samples"]} | {four_summary["samples"]} |
| train steps/s | {one_summary["steps"]} | {four_summary["steps"]} |
| average wall-clock step time (s) | {one_summary["runtime"] / 500:.4f} | {four_summary["runtime"] / 500:.4f} |

## Inference performance
| Metric | 1 GPU | 4 GPUs |
|---|---:|---:|
| eval samples/s | {one_summary["eval_samples"]} | {four_summary["eval_samples"]} |
| eval runtime (s) | {one_summary["eval_runtime"]} | {four_summary["eval_runtime"]} |
| eval loss | {one_summary["eval_loss"]} | {four_summary["eval_loss"]} |

## Communication
| Metric | 1 GPU | 4 GPUs |
|---|---:|---:|
| Total measured comm time (s) | {one_summary["comm_total"]} | {four_summary["comm_total"]} |
| Total measured comm bytes | {one_summary["comm_bytes"]} | {four_summary["comm_bytes"]} |
| Avg comm time / all-reduce call (s) | {one_summary["comm_call"]} | {four_summary["comm_call"]} |
| Avg comm time / optimizer step (s) | {one_summary["comm_step"]} | {four_summary["comm_step"]} |
| Theoretical gradient payload / step | 3,096,120,320 bytes | 3,096,120,320 bytes |

## Did DDP improve performance?
No for this job. The 4-GPU run processed 15.11 samples/s, compared with 46.36 samples/s on one GPU: a 0.33× speedup (a slowdown), far below ideal 4× scaling. It also needed {four_summary["runtime"] / one_summary["runtime"]:.1f}× longer to finish the same 500 optimizer steps. The loss-vs-time graph is therefore the fair comparison: the 1-GPU job reaches each logged step much earlier. The 4-GPU configuration has a four-times larger global batch (64 vs. 16), so its epoch count also differs; this is why loss-versus-steps and loss-versus-time are both included.

The communication hook measured {four_summary["comm_step"]} of communication per optimizer step while the observed 4-GPU step time was {four_summary["runtime"] / 500:.2f} s. That leaves little time for compute and accounts for the poor scaling. The measured total is accumulated across the run/ranks, so it should not be interpreted as a single rank's wall-clock duration.

## Improvements
- Use a topology with genuinely high-bandwidth, low-latency GPU-to-GPU connectivity and verify NCCL transports; the current all-reduce cost dominates.
- Increase compute per synchronization with a larger stable per-device batch or gradient accumulation, trading memory/optimization behavior for less frequent synchronization.
- Enable/better utilize mixed precision and tune DDP bucket sizes/overlap so reductions start during backpropagation.
- For models that no longer fit efficiently, use FSDP or ZeRO to shard model/optimizer state, while measuring whether their added communication is beneficial.
"""
    (ROOT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
