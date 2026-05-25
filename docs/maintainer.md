# Maintainer notes

## Publishing Apple artifacts for a release

Phase 14 ships zero-config Apple speedup via pre-built artifacts hosted as
**GitHub Release assets** on the esm-cpp repo. The maintainer-side flow:

1. **Tag the release** in the normal way (`git tag v0.2.0`, push, draft a
   release on GitHub). Don't upload artifacts manually — the publisher
   tool will.

2. **Set up the convert-time env once** (~3 GB install; only the
   maintainer needs this):

   ```bash
   # Python 3.12 venv + the convert deps coremltools needs
   python3.12 -m venv /tmp/ct312
   /tmp/ct312/bin/pip install coremltools "torch>=2.5,<2.8" transformers \
       safetensors numpy
   ```

3. **Build + upload artifacts** for each model size:

   ```bash
   # 8M is fastest to verify the pipeline (~30 s)
   /tmp/ct312/bin/python tools/publish_apple_artifacts.py \
       --model esm2_t6_8M \
       --out /tmp/esm-cpp-publish-8m \
       --push --tag v0.2.0

   # Then the rest (150M, 650M — each takes ~5-30 min for the full sweep)
   /tmp/ct312/bin/python tools/publish_apple_artifacts.py \
       --model esm2_t30_150M \
       --out /tmp/esm-cpp-publish-150m \
       --push --tag v0.2.0

   /tmp/ct312/bin/python tools/publish_apple_artifacts.py \
       --model esm2_t33_650M \
       --out /tmp/esm-cpp-publish-650m \
       --push --tag v0.2.0
   ```

4. **Sanity-check the consumer flow** on a fresh checkout:

   ```bash
   pip install esm-cpp
   esm-cpp-fetch-artifacts --model esm2_t6_8M --list
   esm-cpp-fetch-artifacts --model esm2_t6_8M
   python -c "
   import esm_cpp, numpy as np
   m = esm_cpp.Model.load_from_safetensors('<path to esm2_t6_8M.safetensors>')
   print('auto-loaded amx :', m.amx_artifacts_path)
   print('auto-loaded WG  :', m.whole_graph_shapes)
   "
   ```

## Requirements

- `gh` CLI installed + `gh auth login` for `--push`.
- Apple Silicon Mac (the artifacts only build on Apple).
- ~20 GB free disk for the staging dirs across all model sizes.

## Without an upload (dry-run)

Drop `--push --tag` to build + tarball locally without touching GitHub:

```bash
/tmp/ct312/bin/python tools/publish_apple_artifacts.py \
    --model esm2_t6_8M --out /tmp/esm-cpp-publish-8m
ls -la /tmp/esm-cpp-publish-8m/
# manifest.json + esm2-8m-amx-fp16.tar.gz + esm2-8m-whole-graph-B*.tar.gz
```

The local round-trip test points the fetch CLI at the staged tree:

```bash
esm-cpp-fetch-artifacts --model esm2_t6_8M \
    --repo file:///tmp/esm-cpp-publish-8m \
    --tag '(dry-run)' \
    --out /tmp/esm-cpp-cache
```

## Tarball naming convention

The fetch CLI knows the names — keep these stable across releases:

```
esm2-<size>-amx-fp16.tar.gz
esm2-<size>-whole-graph-B<B>-L<L>.tar.gz
manifest.json
```

`<size>` is the lowercased trailing chunk of the shorthand
(`esm2_t33_650M` → `650m`).

## Versioning

The `--tag` value should match the engine version that pins this artifact
set. The fetch CLI defaults to `v<esm_cpp.__version__>` so an artifact
republish requires a patch version bump on the engine side. Each
`esm_cpp_artifact.json` inside the tarball carries a `trace_sha` that
must match what the engine was built against, or `Model::Load*` prints a
"refresh via esm-cpp-fetch-artifacts" warning at load time.

## Adding a new model size

1. Add the shorthand → HF id mapping in two places (keep in sync):
   - `python/esm_cpp/fetch_artifacts.py::_HF_ID_FOR_SHORTHAND`
   - `tools/publish_apple_artifacts.py::_HF_ID_FOR_SHORTHAND`
2. Decide the default shape sweep for `_DEFAULT_SHAPES` in the publisher.
3. Run the publisher in dry-run, verify tarball sizes are reasonable
   (each must be < 2 GB for the GitHub Releases asset cap).
4. Push + sanity-check via the consumer flow above.
