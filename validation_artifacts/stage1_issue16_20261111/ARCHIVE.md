# Issue16 evaluation and heading-source analysis

Remote Linux evidence from issue16_20261111. Start with results.md. This archive preserves exact source bytes, including preflight corrections and raw logs. It contains no new training weights: cohort.json binds the existing eleven immutable checkpoints and their SHA-256 digests.

Large files are stored losslessly in lossless_chunks/. Reconstruct each original file by concatenating its part paths in the order listed in lossless_chunk_manifest.json; verify the resulting byte count and original SHA-256. The original frozen source files remain unchanged on the server. All other source_files_manifest.json entries retain their original relative paths and exact bytes.

Preflight_phase_definition_superseded is forensic validation evidence, excluded from formal trial denominators. Formal denominators are 34 jobs,199 command records,203776 initial trials. Original lineage and the representative trial selection reuse committed issue15 evidence. Runtime readback and replay reconstruction are separate validations, not additional formal trials.

Frozen absolute paths in provenance describe the original remote experiment. Resolve archived equivalents by relative suffix under this directory, using lossless chunk reconstruction where listed. Source shaders/USD/checkpoints already bound by frozen provenance remain remote source assets rather than being copied to the local client.
