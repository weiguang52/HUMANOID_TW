"""Fit-only paired action-response loss; never load validation transitions."""

import hashlib
import json

import numpy as np
import torch


class AntitheticResponseLoss:
    def __init__(self, root, integrity_path, device="cuda:0"):
        audit = json.loads(integrity_path.read_text())
        assert audit["passed"] and audit["fit_conditions"] == 240
        hashes = {r["file"]: r["sha256"] for r in audit["source_files"]}
        rows = json.loads((root / "results.json").read_text())["rows"]
        data = {}
        for r in rows:
            if r["split"] != "fit":
                continue
            p = (root / r["file"]).resolve()
            assert p.is_relative_to(root.resolve())
            assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[r["file"]]
            with np.load(p) as z:
                d = {
                    k: z[k].copy() for k in ["state", "action", "context", "next_state"]
                }
            data[(r["warmup"], r["kind"], r.get("epsilon"), r.get("assignment"))] = d
        plan = json.loads((root / "plan.json").read_text())
        arrays = {k: [] for k in ["state", "action", "context", "next_state"]}
        den = []
        for warmup in plan["fit_warmups"]:
            for eps in plan["epsilons"]:
                a, b = [data[(warmup, "branch", eps, i)] for i in [0, 1]]
                for k, values in arrays.items():
                    values.append(
                        np.stack([a[k][::2], a[k][1::2], b[k][::2], b[k][1::2]], axis=1)
                    )
                den.extend([4 * eps] * 30)
        self.data = {
            k: torch.as_tensor(np.concatenate(v), device=device, dtype=torch.float32)
            for k, v in arrays.items()
        }
        self.den = torch.tensor(den, device=device)[:, None]
        assert len(self.den) == 240
        self.generator = torch.Generator(device=device).manual_seed(44)
        self.target = self.derivative(self.data["next_state"]).detach()
        assert torch.isfinite(self.target).all()

    def derivative(self, v):
        return (v[:, 1] - v[:, 0] + v[:, 2] - v[:, 3]) / self.den

    def __call__(self, model, batch=16):
        idx = torch.randint(
            240, (batch,), generator=self.generator, device=self.den.device
        )
        d = self.data
        pred = model(
            d["state"][idx].reshape(-1, 73),
            d["action"][idx].reshape(-1, 30),
            d["context"][idx].reshape(-1, 34),
        ).reshape(batch, 4, 73)
        response = (pred[:, 1] - pred[:, 0] + pred[:, 2] - pred[:, 3]) / self.den[idx]
        scale = (model.state_norm.var + model.state_norm.eps).sqrt()
        scale = torch.where(model.state_norm.frozen_mask, torch.ones_like(scale), scale)
        error = (response - self.target[idx]) / scale
        root = error[:, 10:13].square().mean()
        joint = error[:, 43:73].square().mean()
        loss = 0.5 * (root + joint)
        assert torch.isfinite(loss)
        return loss, root, joint
