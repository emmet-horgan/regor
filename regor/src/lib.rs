//! Safe, idiomatic Rust API for ARM's regor compiler.
//!
//! Regor is the C/C++ backend of ARM's [Vela](https://pypi.org/project/ethos-u-vela/)
//! neural-network compiler targeting the Ethos-U family of NPUs.
//!
//! # Quick start
//!
//! ```no_run
//! use regor::{Compiler, InputFormat};
//! use regor::options::*;
//!
//! let model_bytes = std::fs::read("model.tflite")?;
//! let accelerator = AcceleratorConfig::EthosU55_256;
//!
//! // The accelerator is described by the system configuration, not by a
//! // compiler option — see [`options`].
//! let system = SystemConfig::new(accelerator)
//!     .system_config_name("Ethos_U55_High_End_Embedded")
//!     .memory_mode_name("Shared_Sram")
//!     .vela_ini(std::fs::read_to_string("vela.ini")?)
//!     .build();
//!
//! let options = CompilerOptions::new()
//!     .optimise(Optimise::Performance)
//!     .build()?;
//!
//! let mut compiler = Compiler::new(accelerator.architecture())?;
//! compiler.set_system_config(&system)?;
//! compiler.set_options(&options)?;
//! let output = compiler.compile(InputFormat::TfLite, &model_bytes)?;
//!
//! std::fs::write("output.tflite", output.as_bytes())?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

//! # Concurrency
//!
//! Every call into the C library is serialised on a single process-wide lock,
//! because regor reaches shared global state that its own locking does not
//! cover. [`Compiler`] is `Send` but not `Sync`: move a context between threads
//! freely, but expect compilations to run one at a time regardless of how many
//! contexts exist. See [`sync`] for the full reasoning.
//!
//! # Logging
//!
//! regor's diagnostics are forwarded to the [`log`] and [`tracing`] facades
//! under the target `regor`; enable the matching feature and call
//! [`logging::init`]. The underlying callback API is process-global and is not
//! exposed. See [`logging`].

mod compiler;
mod config;
mod constraints;
mod error;
mod format;
pub mod logging;
pub mod options;
mod output;
mod perf;
pub mod sync;

pub use compiler::Compiler;
pub use config::Architecture;
pub use constraints::{ConstraintsReport, OperatorConstraints};
pub use error::Error;
pub use format::InputFormat;
pub use logging::{LogFilter, LogFormat};
pub use options::{AcceleratorConfig, CompilerOptions, OptionsError, SystemConfig};
pub use output::{Blob, Output};
pub use perf::{MemoryAccessPerf, PeakMemoryUsage, PerfReport};

pub type Result<T> = std::result::Result<T, Error>;
