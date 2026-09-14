use std::os::raw::c_uint;

use regor_sys as ffi;

use crate::error::check_global;

/// Logging output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogFormat {
    /// Plain text (no ANSI escapes).
    Text,
    /// Terminal-friendly with ANSI color codes.
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
    /// Bitmask selecting which log categories to emit.
    ///
    /// The exact bit assignments are defined by the regor C library.
    /// Pass `LogFilter::all()` to see everything.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct LogFilter: u32 {
        const ERROR   = 1 << 0;
        const WARNING = 1 << 1;
        const INFO    = 1 << 2;
        const DEBUG   = 1 << 3;
    }
}

/// Install a global log callback (plain-text format).
///
/// # Safety
///
/// `callback` must be safe to call from any thread at any time while the
/// regor library is loaded.
pub fn set_log_callback(
    callback: unsafe extern "C" fn(*const std::os::raw::c_void, usize),
    filter: LogFilter,
) -> crate::Result<()> {
    let rc = unsafe { ffi::regor_set_logging(Some(callback), filter.bits()) };
    check_global(rc)
}

/// Install a global log callback with explicit format control.
pub fn set_log_callback_ex(
    callback: unsafe extern "C" fn(*const std::os::raw::c_void, usize),
    filter: LogFilter,
    format: LogFormat,
) -> crate::Result<()> {
    let rc = unsafe { ffi::regor_set_logging_ex(Some(callback), filter.bits(), format.to_raw()) };
    check_global(rc)
}
