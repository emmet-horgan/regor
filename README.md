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

The `regor-sys` build script looks for the native library via environment
variables:

| Variable | Default | Purpose |
|---|---|---|
| `REGOR_DIR` | `regor-sys/regor/` | Root of the regor build tree |
| `REGOR_LIB_DIR` | `$REGOR_DIR/lib` | Directory containing `libregor-static.a` |
| `REGOR_INCLUDE_DIR` | `$REGOR_DIR/include` | Directory containing `regor.h` |
| `REGOR_CXX_LIB` | `stdc++` | C++ standard library to link (e.g. `c++` on macOS) |

Build regor from the Vela source tree first, then point these variables at
the output:

```sh
# Inside the ethos-u-vela checkout:
cmake -B build ethosu/regor -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Then build the Rust crate:
REGOR_LIB_DIR=build REGOR_INCLUDE_DIR=ethosu/regor/include cargo build
```

## Examples

```sh
cargo run --example compile_tflite -- model.tflite output.tflite
cargo run --example check_constraints -- model.tflite
```

## License

The Rust bindings are dual-licensed under MIT / Apache-2.0. The underlying
regor library is licensed under Apache-2.0 by Arm Limited.
