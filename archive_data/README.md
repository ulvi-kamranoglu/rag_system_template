# Stage archives

Only `01_raw`, `02_clean`, `03_chunks`, `04_embeddings` and `05_indexes` belong here.
After the 2026-10-31 cleanup, only the active v1.5 canonical base, current batch
stage archives, its raw upload and the latest collector archive are retained.
Historical test versions and runs have been removed explicitly.

A complete recovery vector snapshot (22,988 vectors plus mapping/metadata) is in
`04_embeddings/run_20261031_184650_826a065da2ce42089b3d86a08721cc62`.
It consolidates the current index, including rector records and corrected legal
dates, without changing shared_data. The other retained embedding run contains
only the latest batch. The retained index archive contains only active v1.5
files and has updated inventory hashes; obsolete carried-forward versions were
removed from that archive. A subsequent local publication also compacted
shared_data/05_indexes to the same active files. Live Qdrant data was unchanged.

Historical reports retain original machine paths, not deployment configuration.
Old development jobs are not transferred as pending jobs. New checkpoints go in
`runtime_data/pipeline_runs`. Back up artifacts, collector state and Qdrant together.
