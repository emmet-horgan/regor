# regor-rs

Idiomatic Rust bindings to ARM's **regor** compiler — the C/C++ backend of
[Vela](https://pypi.org/project/ethos-u-vela/), which compiles neural-network
models for the Ethos-U family of NPUs (U55, U65, U85).

## Crates

| Crate | Purpose |
|---|---|
| `regor-sys` | Raw `unsafe` FFI matching `regor.h` |
| `regor` | Safe, idiomatic wrapper |

## Quick start

```rust
use regor::{Architecture, Compiler, InputFormat};

let model = std::fs::read("model.tflite")?;

let mut compiler = Compiler::new(Architecture::EthosU55)?;
let output = compiler.compile(InputFormat::TfLite, &model)?;

std::fs::write("output.tflite", output.as_bytes())?;

let report = compiler.perf_report()?;
println!("NPU cycles: {}", report.npu_cycles);
```

## Building

`cargo build` works out of the box. `regor-sys` resolves the native library in
exactly one of three ways, in order:

1. **`REGOR_LIB_DIR`** — link a library you already have.
2. **`ETHOS_U_VELA_PATH`** — build from a local `ethos-u-vela` checkout. Needs
   `cmake` and a C++ compiler. The build is out-of-tree, so your checkout is
   never modified.
3. **Otherwise** — download the pre-built artifact for your target from this
   repository's `native-*` release. Needs `curl` or `wget` on PATH.

There is no fetch-the-source fallback. Downloads are verified against the
digests pinned in `regor-sys/artifacts.sha256`; a tarball whose digest is not
listed there is never linked. If no artifact is published for your target, the
build fails with instructions rather than quietly doing something slower.

Artifacts are produced by the **Build native regor** workflow
(`.github/workflows/build-native.yml`), dispatched manually. Each one is the
cmake *install* tree — the static library plus the headers — for one target and
configuration. Publishing a new set means updating `artifacts.sha256` from the
release's `SHA256SUMS` and bumping `ARTIFACT_TAG` in `regor-sys/build.rs` in the
same commit.

### Patches

`regor-sys/patches/` carries fixes for upstream regor bugs — currently a
context-id collision that can destroy a live compiler when one context is
released while another is in use.

They are applied **only** when building the published artifacts. The build
script never patches, so if you build from source you get your source, exactly
as supplied. Apply them yourself if you want the fixes:

```sh
cd ethos-u-vela
git apply /path/to/regor-sys/patches/*.patch
```

### Environment variables

| Variable | Purpose |
|---|---|
| `REGOR_LIB_DIR` | Link a library you built yourself; skips everything else |
| `REGOR_INCLUDE_DIR` | Directory containing `regor.h`, used with `REGOR_LIB_DIR` |
| `ETHOS_U_VELA_PATH` | Build from a local `ethos-u-vela` checkout |
| `REGOR_BUILD_TYPE` | `Release` (default) or `Debug` |
| `REGOR_ARTIFACT_TAG` | Override the release tag to download from |
| `REGOR_ARTIFACT_BASE_URL` | Override the artifact base URL entirely |
| `REGOR_CACHE_DIR` | Where downloads and native build output are cached |
| `REGOR_FORCE_REBUILD` | Force a source rebuild, ignoring the cache |
| `REGOR_CXX_LIB` | C++ standard library to link (e.g. `c++` on macOS) |

Building against a local Vela checkout:

```sh
git clone --branch 5.2.0 \
  https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela.git
ETHOS_U_VELA_PATH=$PWD/ethos-u-vela cargo build
```

## Concurrency

regor reaches process-global state that its own locking does not cover, so
every call into the C library is serialised on a single lock. `Compiler` is
`Send` but not `Sync`: contexts move between threads freely, but only one
compilation runs at a time no matter how many contexts exist. See the `sync`
module docs for the reasoning.

## Logging

regor's diagnostics can be forwarded to [`log`] or [`tracing`] under the target
`regor`. Enable the matching feature and call `logging::init` once at start-up:

```toml
regor = { version = "0.1", features = ["tracing"] }
```

```rust
regor::logging::init(regor::LogFilter::ERROR | regor::LogFilter::WARNING)?;
```

The underlying regor callback is process-global and unsynchronised, so it is
not exposed; this crate installs a single writer of its own and routes
everything through it.

[`log`]: https://docs.rs/log
[`tracing`]: https://docs.rs/tracing

## Examples

```sh
cargo run --example compile_tflite -- model.tflite output.tflite
cargo run --example check_constraints -- model.tflite
```

## License

The Rust bindings are dual-licensed under MIT / Apache-2.0. The underlying
regor library is licensed under Apache-2.0 by Arm Limited.
