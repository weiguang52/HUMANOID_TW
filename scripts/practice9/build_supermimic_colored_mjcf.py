"""Bake existing precolored-USD palette into a separate MuJoCo asset offline."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--colored-usd", type=Path, required=True)
    p.add_argument("--palette", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), "Refusing to overwrite colored asset"
    assert a.colored_usd.is_file()
    palette = json.loads(a.palette.read_text())
    colors = {row["link"]: row["rgb"] for row in palette}
    assert len(colors) == 30
    tree = ET.parse(a.source)
    original = ET.tostring(tree.getroot())
    changed = []

    def visit(body, inherited=(0.65, 0.65, 0.65)):
        rgb = colors.get(body.get("name"), inherited)
        for geom in body.findall("geom"):
            geom.set("rgba", " ".join(format(float(v), ".9g") for v in (*rgb, 1.0)))
            changed.append(geom.get("name"))
        for child in body.findall("body"):
            visit(child, rgb)

    for body in tree.getroot().find("worldbody").findall("body"):
        visit(body)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(a.output, encoding="unicode")
    # Structural proof: the only changed XML attributes are geom rgba.
    check = ET.parse(a.output).getroot()
    before = ET.fromstring(original)
    for root in (check, before):
        for geom in root.iter("geom"):
            geom.attrib.pop("rgba", None)
    assert ET.tostring(check) == ET.tostring(before)
    models = [mujoco.MjModel.from_xml_path(str(path)) for path in (a.source, a.output)]
    checked = []
    for name in dir(models[0]):
        old, new = getattr(models[0], name), getattr(models[1], name)
        if isinstance(old, np.ndarray) and name != "geom_rgba":
            assert np.array_equal(old, new), name
            checked.append(name)
    assert len(checked) > 100
    report = {
        "passed": True,
        "source": str(a.source),
        "colored_mjcf": str(a.output),
        "source_mjcf_sha256": hashlib.sha256(a.source.read_bytes()).hexdigest(),
        "colored_mjcf_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
        "colored_usd": str(a.colored_usd),
        "colored_usd_sha256": hashlib.sha256(a.colored_usd.read_bytes()).hexdigest(),
        "palette": str(a.palette),
        "palette_sha256": hashlib.sha256(a.palette.read_bytes()).hexdigest(),
        "changed_geoms": changed,
        "identical_compiled_fields": checked,
        "note": "MuJoCo cannot load USD directly. Colors baked offline from existing precolored USD palette; no runtime material binding. Only geom_rgba changes.",
    }
    a.output.with_suffix(".color_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    a.output.with_suffix(".colors.json").write_text(
        json.dumps(palette, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "identical_compiled_fields"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
