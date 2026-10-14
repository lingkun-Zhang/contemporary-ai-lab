# Historical experimental records

Every existing run directory is retained, including repeated runs, failed improvements, and cached baselines.
The private raw archive is byte-verified against the source files. It is not intended for public publication.
The public raw export is transformed: every row is retained, raw-text columns are removed, feature dictionaries are excluded, and local lab paths are normalized. See public_export_manifest.json.
Teacher-provided datasets were not copied. Dataset hashes identify the input; public redistribution permission is unconfirmed.
The current_source_snapshot is the current code, not retroactively the exact training source for old runs. Missing original logs remain marked as gaps.
A rerun must live separately and be labeled as a new reproduction. Nothing in this archive publishes, refits a final model, or creates an official prediction submission.
