//! Routing regor's diagnostics into [`log`] and [`tracing`].
//!
//! regor writes its diagnostics through a single, process-global writer
//! function installed with `regor_set_logging`. There is one writer for the
//! whole process, it is not tied to a context, and replacing it while a
//! compilation is running races with the code reading it.
//!
//! That API is therefore **not exposed**. Handing out `set_log_callback` would
//! mean handing out a way for one part of a program to silently redirect or
//! disable another part's diagnostics — and to do so unsynchronised. Instead
//! this module installs one writer of its own, once, and forwards whatever
//! regor emits to the logging facades the rest of the program already uses.
//!
//! # Usage
//!
//! Enable the `log` and/or `tracing` feature, then call [`init`] once during
//! start-up:
//!
//! ```no_run
//! regor::logging::init(regor::LogFilter::ERROR | regor::LogFilter::WARNING)?;
//! # Ok::<(), regor::Error>(())
//! ```
//!
//! Records are emitted under the target `regor`. Without a call to [`init`],
//! regor stays silent: a writer is still installed (the C library asserts on a
//! null one) but with an empty filter mask.
//!
//! # Thread safety
//!
//! The writer is called by regor from whichever thread is compiling, while that
//! thread holds the global FFI lock described in [`crate::sync`]. It therefore
//! takes only its own line buffer mutex; reaching for the FFI lock would
//! deadlock immediately. Panics are caught at the boundary, because unwinding
//! out of an `extern "C"` function into C++ frames is undefined behaviour.

use std::os::raw::{c_uint, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, PoisonError};

use regor_sys as ffi;

use crate::error::check_global;
use crate::sync;

/// Logging output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LogFormat {
    /// Plain text (no ANSI escapes).
    #[default]
    Text,
    /// Terminal-friendly with ANSI colour codes.
    ///
    /// Rarely what you want when forwarding to `log` or `tracing`: the escape
    /// sequences end up embedded in the record's message and will be written
    /// verbatim into log files.
    Terminal,
}

impl LogFormat {
    fn to_raw(self) -> c_uint {
        match self {
            LogFormat::Text => ffi::regor_logging_format_t::REGOR_LOG_FORMAT_TEXT as c_uint,
            LogFormat::Terminal => ffi::regor_logging_format_t::REGOR_LOG_FORMAT_TERMINAL as c_uint,
        }
    }
}

bitflags::bitflags! {
    /// Bitmask selecting which log categories regor emits.
    ///
    /// The bit assignments are defined by the regor C library.
    /// Pass `LogFilter::all()` to see everything.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct LogFilter: u32 {
        const ERROR   = 1 << 0;
        const WARNING = 1 << 1;
        const INFO    = 1 << 2;
        const DEBUG   = 1 << 3;
    }
}

/// Severity assigned to a forwarded line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    Error,
    Warning,
    Info,
    Debug,
}

/// Start forwarding regor diagnostics, in plain-text format.
///
/// Safe to call more than once; the most recent call wins. Calling it while
/// another thread is compiling is safe but the change may not take effect until
/// that compilation finishes, since installing the writer takes the same
/// process-wide lock.
///
/// With neither the `log` nor the `tracing` feature enabled this still installs
/// the writer, but the lines are discarded — there is nowhere to send them.
pub fn init(filter: LogFilter) -> crate::Result<()> {
    init_with_format(filter, LogFormat::Text)
}

/// Start forwarding regor diagnostics with an explicit output format.
pub fn init_with_format(filter: LogFilter, format: LogFormat) -> crate::Result<()> {
    install(filter, format, true)
}

/// Stop forwarding. regor keeps a writer installed but emits nothing.
pub fn disable() -> crate::Result<()> {
    install(LogFilter::empty(), LogFormat::Text, true)
}

/// Tracks whether the writer is installed and whether the caller chose the
/// filter, so that the implicit installation done on
/// [`Compiler`](crate::Compiler) creation cannot silently undo an explicit
/// [`init`].
struct InstallState {
    installed: bool,
    explicit: bool,
}

static INSTALL_STATE: Mutex<InstallState> = Mutex::new(InstallState {
    installed: false,
    explicit: false,
});

/// Install a writer with the given filter.
///
/// Lock order is `INSTALL_STATE` then the FFI lock, and nothing acquires them
/// the other way round.
fn install(filter: LogFilter, format: LogFormat, explicit: bool) -> crate::Result<()> {
    let mut state = INSTALL_STATE.lock().unwrap_or_else(PoisonError::into_inner);

    if !explicit && (state.installed || state.explicit) {
        return Ok(());
    }

    let rc = {
        let _guard = sync::lock();
        unsafe { ffi::regor_set_logging_ex(Some(write_log), filter.bits(), format.to_raw()) }
    };
    check_global(rc)?;

    state.installed = true;
    state.explicit |= explicit;
    Ok(())
}

/// Install a silent writer if none is installed yet.
///
/// regor asserts on a null writer during compilation, so one must exist before
/// the first compile. Using the same bridge rather than a separate no-op keeps
/// a single writer in play for the process lifetime, which means a later
/// [`init`] only has to change the filter mask.
pub(crate) fn ensure_installed() {
    // A failure here means regor rejected the writer, which the subsequent
    // compile will report with a usable error message. Nothing is gained by
    // failing compiler construction over it.
    let _ = install(LogFilter::empty(), LogFormat::Text, false);
}

/// Buffers a partial line between writer calls.
///
/// regor writes in arbitrary chunks that do not align with line boundaries, so
/// emitting one record per call would shred messages across records. This is
/// deliberately not the FFI lock: see the module docs on lock ordering.
static LINE_BUFFER: Mutex<String> = Mutex::new(String::new());

/// The writer handed to regor.
unsafe extern "C" fn write_log(data: *const c_void, size: usize) {
    if data.is_null() || size == 0 {
        return;
    }

    // Unwinding from here would cross C++ frames, which is undefined
    // behaviour. Anything that goes wrong is dropped instead.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let bytes = std::slice::from_raw_parts(data.cast::<u8>(), size);
        // regor emits UTF-8, but a chunk boundary can split a multi-byte
        // sequence. Lossy conversion keeps the rest of the line readable.
        let text = String::from_utf8_lossy(bytes);

        let mut buffer = LINE_BUFFER.lock().unwrap_or_else(PoisonError::into_inner);
        buffer.push_str(&text);

        while let Some(end) = buffer.find('\n') {
            let line = buffer[..end].trim_end().to_owned();
            buffer.drain(..=end);
            if !line.is_empty() {
                emit(classify(&line), &line);
            }
        }
    }));
}

/// Guess a severity for a line.
///
/// regor's writer receives already-formatted text and no severity alongside it;
/// the filter mask controls what is produced, not what each chunk was. The
/// prefix is the only signal available, so lines that do not announce
/// themselves are reported at info.
fn classify(line: &str) -> Level {
    let head = line.trim_start().trim_start_matches(['[', '(']);
    let lower = head.to_ascii_lowercase();

    if lower.starts_with("error") || lower.starts_with("fatal") {
        Level::Error
    } else if lower.starts_with("warn") {
        Level::Warning
    } else if lower.starts_with("debug") || lower.starts_with("trace") {
        Level::Debug
    } else {
        Level::Info
    }
}

#[cfg_attr(
    not(any(feature = "log", feature = "tracing")),
    allow(unused_variables)
)]
fn emit(level: Level, message: &str) {
    #[cfg(feature = "log")]
    {
        let level = match level {
            Level::Error => log::Level::Error,
            Level::Warning => log::Level::Warn,
            Level::Info => log::Level::Info,
            Level::Debug => log::Level::Debug,
        };
        log::log!(target: "regor", level, "{message}");
    }

    #[cfg(feature = "tracing")]
    {
        match level {
            Level::Error => tracing::error!(target: "regor", "{message}"),
            Level::Warning => tracing::warn!(target: "regor", "{message}"),
            Level::Info => tracing::info!(target: "regor", "{message}"),
            Level::Debug => tracing::debug!(target: "regor", "{message}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_is_taken_from_the_line_prefix() {
        assert_eq!(classify("Error: bad operator"), Level::Error);
        assert_eq!(classify("  [warning] unsupported"), Level::Warning);
        assert_eq!(classify("Debug: scheduling"), Level::Debug);
    }

    #[test]
    fn unlabelled_lines_report_as_info() {
        assert_eq!(classify("compiling subgraph 3"), Level::Info);
    }
}
