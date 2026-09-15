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

mod compiler;
mod config;
mod constraints;
mod error;
mod format;
mod logging;
pub mod options;
mod output;
mod perf;

pub use compiler::Compiler;
pub use config::Architecture;
pub use constraints::{ConstraintsReport, OperatorConstraints};
pub use error::Error;
pub use format::InputFormat;
pub use logging::{set_log_callback, set_log_callback_ex, LogFilter, LogFormat};
pub use options::{AcceleratorConfig, CompilerOptions, OptionsError, SystemConfig};
pub use output::{Blob, Output};
pub use perf::{MemoryAccessPerf, PeakMemoryUsage, PerfReport};

pub type Result<T> = std::result::Result<T, Error>;
