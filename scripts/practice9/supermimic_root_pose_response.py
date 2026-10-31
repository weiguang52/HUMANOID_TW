"""Fit-only native root-pose response at declared perturbation scales; output normalization frozen."""

import hashlib
import json

import numpy as np
import torch


class RootPoseResponseLoss:
    def __init__(self, root, cfg, device="cuda:0"):
        audit = json.loads((root / "integrity.json").read_text())
        assert audit["passed"]
        plan = json.loads((root / "plan.json").read_text())
        assert plan["contract"] == "root_pose_state_branches_v1"
        assert set(plan["fit_warmups"]).isdisjoint(plan["validation_warmups"])
        mapping = json.loads((root / "control_mapping.json").read_text())
        assert mapping["state_joint_names"] == cfg["extra"]["tw_state_joint_names"]
        assert mapping["action_joint_names"] == cfg["extra"]["tw_action_joint_names"]
        hashes = {x["file"]: x["sha256"] for x in audit["files"]}
        loaded = {}
        self.files = []
        for row in json.loads((root / "results.json").read_text())["rows"]:
            if row["split"] != "fit":
                continue
            p = (root / row["file"]).resolve()
            assert p.is_relative_to(root.resolve())
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            assert digest == hashes[row["file"]]
            with np.load(p) as z:
                d = {
                    k: z[k].copy() for k in ["state", "action", "context", "next_state"]
                }
            loaded[
                (row["warmup"], row["kind"], row.get("level"), row.get("assignment"))
            ] = d
            self.files.append({"file": row["file"], "sha256": digest})
        assert len(loaded) == 20
        arrays = {k: [] for k in ["state", "action", "context", "next_state"]}
        den = []
        axes = []
        for warmup in plan["fit_warmups"]:
            for level in plan["levels"]:
                a, b = [loaded[(warmup, "branch", level, i)] for i in [0, 1]]
                for k, values in arrays.items():
                    values.append(
                        np.stack([a[k][::2], a[k][1::2], b[k][::2], b[k][1::2]], axis=1)
                    )
                den.extend(4 * level * np.array(plan["base_delta"]))
                axes.extend(range(6))
        self.data = {
            k: torch.as_tensor(np.concatenate(v), device=device, dtype=torch.float32)
            for k, v in arrays.items()
        }
        self.den = torch.tensor(den, device=device, dtype=torch.float32)[:, None]
        self.axes = torch.tensor(axes, device=device)
        self.input_scale = torch.tensor(plan["base_delta"],device=device)
        assert len(self.den) == 48
        self.target = self.derivative(self.data["next_state"]).detach()
        assert torch.isfinite(self.target).all()
        self.generator = torch.Generator(device=device).manual_seed(46)

    def derivative(self, v):
        return (
            v[:, 1, 7:13] - v[:, 0, 7:13] + v[:, 2, 7:13] - v[:, 3, 7:13]
        ) / self.den

    def __call__(self, model, batch=16):
        idx = torch.randint(
            48, (batch,), generator=self.generator, device=self.den.device
        )
        d = self.data
        pred = model(
            d["state"][idx].reshape(-1, 73),
            d["action"][idx].reshape(-1, 30),
            d["context"][idx].reshape(-1, 34),
        ).reshape(batch, 4, 73)
        response = (
            pred[:, 1, 7:13] - pred[:, 0, 7:13] + pred[:, 2, 7:13] - pred[:, 3, 7:13]
        ) / self.den[idx]
        std = (model.state_norm.var[7:13] + model.state_norm.eps).sqrt()
        std = torch.where(model.state_norm.frozen_mask[7:13], torch.ones_like(std), std)
        error = (response - self.target[idx]) * self.input_scale[self.axes[idx], None] / std[None, :]
        loss = error.square().mean()
        assert torch.isfinite(loss)
        return loss
