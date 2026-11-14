# Lossless source restoration

All small files retain their exact bytes. Most files larger than4MiB are stored in verified parts under .chunks/. Directly linked MP4,PNG,CSV andJSON deliveries up to90MiB retain their original paths for playback or raw download. Text/USD files are gzip-compressed with deterministic headers; arrays/videos are split without recompression. lossless_chunk_manifest.json binds original byte counts/SHA256 and every stored part.

On the remote Linux server, restore a complete source tree into a new directory with:

    python scripts/locomotion/restore_issue21_archive.py --archive validation_artifacts/stage1_issue21_20261114 --output /root/tw_stage1/issue21_restored

No local copy is required. Existing_reference_bindings.json binds the same five final checkpoints and historical phase-test states; input_bindings.json binds the exact issue20 strict USD and baseline by exact Git commit,path,blob and SHA256; those existing Git objects are not duplicated here. The exact strict USD and baseline traces are existing issue20 Git objects listed in existing_issue20_input_objects.json. If prior server paths are unavailable, restore the previous issue20 archive with restore_issue20_archive.py first; its source/chunk manifests bind exact bytes. The colored layer references that same strict USD. Instrumentation-v1 data is retained separately and excluded from formal counts. Metadata keeps original absolute paths for provenance. Rebase /root/tw_stage1/issue21_20261114 to the restored root for inspection; do not alter hashed source files.
