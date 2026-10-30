"""Lossless binary-STL triangle conversion for MuJoCo's STL face-count limit."""

import hashlib, math, struct
from pathlib import Path


def convert_stl(source: Path, dest: Path):
    raw = source.read_bytes()
    assert len(raw) >= 84
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError("Expected exact binary STL records")
    vertices = []
    lookup = {}
    faces = []
    expected = hashlib.sha256()
    for i in range(count):
        triangle = []
        for j in range(3):
            key = raw[84 + 50 * i + 12 + 12 * j : 84 + 50 * i + 24 + 12 * j]
            xyz = struct.unpack("<3f", key)
            if not all(math.isfinite(v) for v in xyz):
                raise ValueError("Nonfinite mesh vertex")
            expected.update(key)
            if key not in lookup:
                lookup[key] = len(vertices) + 1
                vertices.append(xyz)
            triangle.append(lookup[key])
        faces.append(triangle)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as out:
        for xyz in vertices:
            out.write("v " + " ".join(format(v, ".9g") for v in xyz) + "\n")
        for face in faces:
            out.write("f " + " ".join(map(str, face)) + "\n")
    decoded = []
    actual = hashlib.sha256()
    nfaces = 0
    for line in dest.open():
        parts = line.split()
        if parts[0] == "v":
            decoded.append(struct.pack("<3f", *map(float, parts[1:])))
        elif parts[0] == "f":
            for index in map(int, parts[1:]):
                actual.update(decoded[index - 1])
            nfaces += 1
    assert nfaces == count and actual.digest() == expected.digest(), (
        "Triangle geometry changed"
    )
    return {
        "source": str(source),
        "obj": str(dest),
        "triangles": count,
        "unique_vertices": len(vertices),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "obj_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "oriented_float32_triangle_sha256": actual.hexdigest(),
        "all_triangle_coordinates_exact": True,
    }
