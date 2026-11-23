#!/usr/bin/env python3
"""Generate prosody-optimized Kokoro-82M-v1.1-zh voice packs (.pt for Python, .bin for kokoro-js).

Preserves 100% of each speaker's [:128] acoustic/timbre subspace while optimizing only [128:]
(duration/prosody half) on GPU via differentiable Kokoro ProsodyPredictor duration alignment
against clean-rhythm donor zf_001 + zf_023 and comma-frame clamping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from kokoro_engine import OptimizedZhG2P  # noqa: E402

DEFAULT_VOICES = ("zf_032", "zf_046", "zf_021", "zf_022", "zf_023", "zf_024", "zf_059", "zf_079")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out-pt", type=Path, default=HERE / "voices_zh")
    ap.add_argument("--out-bin", type=Path, default=HERE / "voices_zh_bin")
    args = ap.parse_args()

    from kokoro import KModel
    from misaki import en as misaki_en

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = Path(snapshot_download("hexgrad/Kokoro-82M-v1.1-zh", revision="01e7505bd6a7a2ac4975463114c3a7650a9f7218"))
    model = KModel(repo_id="hexgrad/Kokoro-82M-v1.1-zh", config=str(local / "config.json"),
                   model=str(next(local.glob("*.pth")))).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    en_g2p = misaki_en.G2P(trf=False, british=False, fallback=None, unk="❓")
    opt_g2p = OptimizedZhG2P(en_callable=lambda t: en_g2p(t)[0], unk="❓")

    v001 = torch.load(str(local / "voices/zf_001.pt"), weights_only=True).to(device)

    corpus_rows = [l.split("\t")[2] for l in (ROOT / "corpus" / "zh.tsv").read_text(encoding="utf-8").splitlines() if l]
    cal_texts = [
        "你好，今天的天气很好。",
        "在光线充足的情况下，诺斯费拉图与普通吸血鬼很容易区分。",
        "我们生活中的朋友有时会让我们觉得无聊，而我们书中的朋友永远不会这样。",
        "国际妇女争取和平与自由联盟（Women's International League for Peace and Freedom）",
        "他说：“这件事并不复杂——只要你愿意仔细检查，就能发现问题所在。”",
    ] + [t for t in corpus_rows[20:160] if "，" in t][:7]

    base_targets = []
    with torch.no_grad():
        for txt in cal_texts:
            ps, _ = opt_g2p(txt)
            ids = [i for i in (model.vocab.get(ch) for ch in ps) if i is not None]
            if len(ids) + 2 > 510 or len(ps) < 8:
                continue
            t_ids = torch.LongTensor([[0, *ids, 0]]).to(device)
            row_idx = min(len(ps) - 1, 509)
            in_len = torch.LongTensor([t_ids.shape[1]]).to(device)
            mask = torch.zeros((1, t_ids.shape[1]), dtype=torch.bool, device=device)
            bd = model.bert(t_ids, attention_mask=(~mask).int())
            d_en = model.bert_encoder(bd).transpose(-1, -2)
            s_d = v001[row_idx, :, 128:]
            d_d = model.predictor.text_encoder(d_en, s_d, in_len, mask)
            x_d, _ = model.predictor.lstm(d_d)
            dur_d = torch.sigmoid(model.predictor.duration_proj(x_d)).sum(dim=-1)
            punct_idx = [idx for idx, ch in enumerate(ps, 1) if ch in ",.;:!? "]
            base_targets.append((row_idx, in_len, mask, d_en, dur_d, punct_idx))

    args.out_pt.mkdir(parents=True, exist_ok=True)
    args.out_bin.mkdir(parents=True, exist_ok=True)

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    summary = {}

    for voice in voices:
        orig_pack = torch.load(str(local / "voices" / f"{voice}.pt"), weights_only=True).to(device)
        v_targets = []
        with torch.no_grad():
            for row_idx, in_len, mask, d_en, dur_d, punct_idx in base_targets:
                s_o = orig_pack[row_idx, :, 128:]
                d_o = model.predictor.text_encoder(d_en, s_o, in_len, mask)
                en_o = d_o.transpose(-1, -2)
                f0_o, n_o = model.predictor.F0Ntrain(en_o, s_o)
                v_targets.append((row_idx, in_len, mask, d_en, dur_d, en_o, f0_o, n_o, punct_idx))

        # Keep [128:] close to each speaker's natural pitch while letting duration align cleanly
        init_w = -2.2 if voice in ("zf_021", "zf_023", "zf_032", "zf_059") else -1.0
        w_blend = torch.tensor([init_w], device=device, requires_grad=True)
        knots = torch.zeros((16, 1, 128), device=device, requires_grad=True)
        opt = torch.optim.Adam([w_blend, knots], lr=0.045)

        def build_s_all():
            interp = torch.nn.functional.interpolate(
                knots.permute(1, 2, 0), size=510, mode="linear", align_corners=True
            ).permute(2, 0, 1)
            w0 = torch.sigmoid(w_blend[0]) * 0.45
            return (1.0 - w0) * orig_pack[:, :, 128:] + w0 * v001[:, :, 128:] + interp

        with torch.backends.cudnn.flags(enabled=False):
            for _step in range(args.steps):
                opt.zero_grad()
                s_all = build_s_all()
                loss = torch.zeros((), device=device)
                for row_idx, in_len, mask, d_en, dur_d, en_o, f0_o, n_o, punct_idx in v_targets:
                    s_cur = s_all[row_idx]
                    d = model.predictor.text_encoder(d_en, s_cur, in_len, mask)
                    x, _ = model.predictor.lstm(d)
                    dur = torch.sigmoid(model.predictor.duration_proj(x)).sum(dim=-1)
                    f0_c, n_c = model.predictor.F0Ntrain(en_o, s_cur)
                    loss_dur = torch.nn.functional.mse_loss(dur, dur_d)
                    loss_out = 0.6 * torch.relu(dur - 6.5).mean()
                    loss_f0 = 0.005 * torch.nn.functional.mse_loss(f0_c, f0_o)
                    loss_n = 0.1 * torch.nn.functional.mse_loss(n_c, n_o)
                    loss = loss + loss_dur + loss_out + loss_f0 + loss_n
                    if punct_idx:
                        loss = loss + 0.8 * torch.relu(dur[:, punct_idx] - 6.5).mean()
                loss = loss + 0.05 * knots.pow(2).mean()
                loss.backward()
                opt.step()

        tuned = orig_pack.clone().detach()
        with torch.no_grad():
            tuned[:, :, 128:] = build_s_all()
        tuned_cpu = tuned.cpu().to(torch.float32).contiguous()

        pt_path = args.out_pt / f"{voice}.pt"
        bin_path = args.out_bin / f"{voice}.bin"
        torch.save(tuned_cpu, str(pt_path))
        bin_path.write_bytes(tuned_cpu.numpy().tobytes())

        with torch.no_grad():
            w0 = float(torch.sigmoid(w_blend[0]) * 0.45)
        summary[voice] = {
            "weights_orig_zf001": [round(1.0 - w0, 3), round(w0, 3)],
            "final_loss": round(float(loss.detach()), 4),
            "pt_bytes": pt_path.stat().st_size,
            "bin_bytes": bin_path.stat().st_size,
        }
        print(f"Saved {voice}: w={summary[voice]['weights_orig_zf001']} loss={summary[voice]['final_loss']}", flush=True)

    (HERE / "voices_opt_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
