# Test fixtures

## `cpjump1/metadata/`

A verbatim copy of the CPJUMP1 platemap and annotation tree that `MetadataIndex` parses. 145 KB,
8 files, committed so the metadata tests run on a fresh clone with no download.

Source: the public Cell Painting Gallery S3 bucket, as configured in `configs/dataset.yml`:

```
s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/metadata/platemaps/2020_11_04_CPJUMP1
s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/metadata/external_metadata
```

Refresh it with `uv run morphoclip data fetch`, which writes the same tree to `data/metadata/`.

These bytes are real, not synthesized. That matters: the tests assert on real-format details the
parser has to handle (tab quoting, an empty `broad_sample` meaning negcon, the `target` vs
`target_list` fallback, the nested `platemap/` subdirectory). A synthesized fixture would only
assert that the fixture says what the fixture says.

`tests/data/test_metadata_realdata.py` checks this copy against `data/metadata/` when the latter
is present, so it cannot silently drift. That test is marked `realdata` and is excluded from the
default run.

`data/reference/cpjump1/` sets the precedent for committing small first-party CPJUMP1 metadata.
