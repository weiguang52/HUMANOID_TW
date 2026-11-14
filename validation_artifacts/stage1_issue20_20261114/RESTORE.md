# Lossless source restoration

All small files retain their exact bytes. Most files larger than4MiB are stored in verified parts under .chunks/. MP4 deliveries up to90MiB retain their original paths for direct playback. Text/USD files are gzip-compressed with deterministic headers; arrays/videos are split without recompression. lossless_chunk_manifest.json binds original byte counts/SHA256 and every stored part.

On the remote Linux server, restore a complete source tree into a new directory with:

    python scripts/locomotion/restore_issue20_archive.py --archive validation_artifacts/stage1_issue20_20261114 --output /root/tw_stage1/issue20_restored

No local copy is required. Existing_reference_bindings.json identifies all five unchanged final checkpoints and102400 fixed states by exact Git commit,path,blob and SHA256; those existing Git objects are not duplicated here. input_assets/original preserves the complete original USD layer structure; the strict USD is self-contained. Metadata keeps original absolute paths for provenance. Rebase /root/tw_stage1/issue20_20261114 to the restored root for inspection; do not alter hashed source files.
