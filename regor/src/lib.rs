//! Safe, idiomatic Rust API for ARM's regor compiler.
//!
//! Regor is the C/C++ backend of ARM's [Vela](https://pypi.org/project/ethos-u-vela/)
//! neural-network compiler targeting the Ethos-U family of NPUs.
//!
//! # Quick start
//!
//! ```no_run
//! use regor::{Compiler, Architecture, InputFormat};
//!
//! let model_bytes = std::fs::read("model.tflite").unwrap();
//!
//! let output = Compiler::new(Architecture::EthosU55)?
//!     .system_config("Ethos_U55_High_End_Embedded")?
//!     .compiler_options("optimise=Performance")?
//!     .compile(InputFormat::TfLite, &model_bytes)?;
//!
//! std::fs::write("output.tflite", output.as_bytes()).unwrap();
//! # Ok::<(), regor::Error>(())
//! ```

mod compiler;
mod config;
mod error;
mod format;
mod logging;
pub mod options;
mod output;
mod perf;
mod constraints;

pub use compiler::Compiler;
pub use config::Architecture;
pub use error::Error;
pub use format::InputFormat;
pub use logging::{LogFormat, LogFilter, set_log_callback, set_log_callback_ex};
pub use output::{Output, Blob};
pub use perf::{
    PerfReport, PeakMemoryUsage, MemoryAccessPerf,
};
pub use constraints::{ConstraintsReport, OperatorConstraints};

pub type Result<T> = std::result::Result<T, Error>;
