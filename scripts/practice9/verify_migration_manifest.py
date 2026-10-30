"""Verify a source-generated JSONL file manifest on the destination data volume."""

import argparse, hashlib, json, os, stat, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("manifest", type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--relocate-known-links", action="store_true")
a = p.parse_args()
root = Path("/root/gpufree-data")
failures = []
rows = 0
total = 0
start = time.time()
relocations = []
for line in a.manifest.open():
    r = json.loads(line)
    f = Path(r["path"])
    rows += 1
    if not f.is_relative_to(root) or ".." in f.parts:
        raise ValueError("Manifest path outside data root")
    try:
        s = f.lstat()
        if r["kind"] == "symlink":
            expected = r["target"]
            relocated = expected
            if a.relocate_known_links:
                if expected.startswith("/root/twrl_runs/"):
                    relocated = (
                        "/root/gpufree-data/twrl_runs/"
                        + expected[len("/root/twrl_runs/") :]
                    )
                elif expected.startswith("/root/.local/share/ov/data/exts/v2/"):
                    relocated = (
                        "/root/gpufree-data/share/ov/data/exts/v2/"
                        + expected[len("/root/.local/share/ov/data/exts/v2/") :]
                    )
            if not stat.S_ISLNK(s.st_mode):
                raise ValueError("expected symlink")
            current = os.readlink(f)
            if relocated != expected:
                if current not in (expected, relocated):
                    raise ValueError("unexpected pre-relocation link target")
                if not Path(relocated).exists():
                    raise ValueError("relocation destination missing")
                if current == expected:
                    f.unlink()
                    f.symlink_to(relocated)
                relocations.append(
                    {
                        "path": str(f),
                        "source_target": expected,
                        "destination_target": relocated,
                    }
                )
            if os.readlink(f) != relocated:
                raise ValueError("symlink target mismatch")
            if not f.exists():
                raise ValueError("unresolved symlink")
        elif r["kind"] == "file":
            if not stat.S_ISREG(s.st_mode) or s.st_size != r["bytes"]:
                raise ValueError("file type/size mismatch")
            h = hashlib.sha256()
            with f.open("rb") as stream:
                while b := stream.read(4 * 1024 * 1024):
                    h.update(b)
            if h.hexdigest() != r["sha256"]:
                raise ValueError("sha256 mismatch")
            total += s.st_size
        else:
            raise ValueError("unknown manifest kind")
        if stat.S_IMODE(s.st_mode) != r["mode"]:
            raise ValueError("mode mismatch")
    except (OSError, ValueError) as e:
        failures.append({"path": str(f), "error": str(e)})
    if rows % 10000 == 0:
        print(rows, total, len(failures), flush=True)
result = {
    "passed": not failures,
    "rows": rows,
    "verified_bytes": total,
    "elapsed_s": time.time() - start,
    "failures": failures,
    "explicit_link_relocations": relocations,
    "scope": "source manifest entries only; does not prove simulator runtime or detect unrelated extra target files",
}
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "failures"}))
raise SystemExit(0 if result["passed"] else 1)
