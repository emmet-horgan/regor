use std::ffi::CStr;
use std::fmt;
use std::os::raw::c_char;

use regor_sys as ffi;

#[derive(Debug)]
pub enum Error {
    /// The C library returned a non-zero status code.
    RegorError { code: i32, message: String },
    /// An argument contained an interior NUL byte.
    NulError(std::ffi::NulError),
    /// Option Misconfiguration.
    OptionError(crate::OptionsError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::RegorError { code, message } => {
                write!(f, "regor error {code}: {message}")
            }
            Error::NulError(e) => write!(f, "interior NUL byte: {e}"),
            Error::OptionError(e) => write!(f, "option misconfiguration: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::NulError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(e: std::ffi::NulError) -> Self {
        Error::NulError(e)
    }
}

impl From<crate::OptionsError> for Error {
    fn from(e: crate::OptionsError) -> Self {
        Error::OptionError(e)
    }
}

/// Check a regor return code. The regor C API returns non-zero (typically 1)
/// on success and 0 on failure.
pub(crate) fn check(ctx: ffi::regor_context_t, code: i32) -> crate::Result<()> {
    if code != 0 {
        return Ok(());
    }
    let mut len: usize = 0;
    // Note that we do not attempt to get the lock here because the assumption is 
    // that we are checking an error internally within the regor bindings in which
    // case we already hold the lock, thus locking here would cause a deadlock.
    unsafe { ffi::regor_get_error(ctx, std::ptr::null_mut(), &mut len) };
    if len == 0 {
        return Err(Error::RegorError {
            code,
            message: String::new(),
        });
    }
    let mut buf: Vec<c_char> = vec![0; len + 1];
    let mut cap = buf.len();
    unsafe { ffi::regor_get_error(ctx, buf.as_mut_ptr(), &mut cap) };
    let msg = unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Err(Error::RegorError { code, message: msg })
}

/// Check a return code for global (context-free) functions like logging.
pub(crate) fn check_global(code: i32) -> crate::Result<()> {
    if code != 0 {
        return Ok(());
    }
    Err(Error::RegorError {
        code,
        message: "regor global function failed".into(),
    })
}
