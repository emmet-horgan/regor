use std::ffi::CString;
use std::mem::MaybeUninit;
use std::os::raw::c_void;
use std::sync::Once;

use regor_sys as ffi;

use crate::config::Architecture;
use crate::constraints::ConstraintsReport;
use crate::error::check;
use crate::format::InputFormat;
use crate::output::{Blob, Output};
use crate::perf::PerfReport;

static LOGGING_INIT: Once = Once::new();

/// No-op log writer registered as a fallback so regor never asserts on a null
/// writer during compilation.
unsafe extern "C" fn noop_log_writer(_data: *const c_void, _size: usize) {}

fn ensure_logging_initialized() {
    LOGGING_INIT.call_once(|| {
        unsafe { ffi::regor_set_logging(Some(noop_log_writer), 0) };
    });
}

/// A regor compiler instance.
///
/// Wraps a `regor_context_t` with RAII semantics — the underlying context is
/// destroyed when this value is dropped.
///
/// The typical workflow is:
///
/// 1. Create a compiler for a target architecture.
/// 2. Optionally configure system settings and compiler options.
/// 3. Call [`compile`](Compiler::compile) with model bytes.
/// 4. Read the compiled [`Output`] and optionally the [`PerfReport`].
pub struct Compiler {
    ctx: ffi::regor_context_t,
}

// The C library documents that each context is independent.
unsafe impl Send for Compiler {}

impl Drop for Compiler {
    fn drop(&mut self) {
        unsafe { ffi::regor_destroy(self.ctx) };
    }
}

impl Compiler {
    /// Create a new compiler targeting `arch`.
    pub fn new(arch: Architecture) -> crate::Result<Self> {
        ensure_logging_initialized();
        let mut ctx: ffi::regor_context_t = 0;
        let rc = unsafe { ffi::regor_create(&mut ctx, arch.as_cstr().as_ptr()) };
        check(ctx, rc)?;
        Ok(Compiler { ctx })
    }

    /// Apply a system configuration string.
    ///
    /// The format is the INI/TOML dialect used by Vela's `.ini` config files
    /// (e.g. `"Ethos_U55_High_End_Embedded"`).
    pub fn system_config(&mut self, config: &str) -> crate::Result<&mut Self> {
        let rc =
            unsafe { ffi::regor_set_system_config(self.ctx, config.as_ptr().cast(), config.len()) };
        check(self.ctx, rc)?;
        Ok(self)
    }

    /// Apply typed compiler options built with [`crate::options::CompilerOptions`].
    ///
    /// ```no_run
    /// # use regor::{Compiler, Architecture};
    /// # use regor::options::*;
    /// let opts = CompilerOptions::new()
    ///     .optimise(Optimise::Performance)
    ///     .arena_cache_size(2 * 1024 * 1024)
    ///     .build()
    ///     .unwrap();
    ///
    /// let mut c = Compiler::new(Architecture::EthosU55).unwrap();
    /// c.set_options(&opts).unwrap();
    /// ```
    pub fn set_options(&mut self, serialised: &str) -> crate::Result<&mut Self> {
        self.compiler_options(serialised)
    }

    /// Apply compiler option overrides from a raw key=value string.
    ///
    /// Prefer [`set_options`](Self::set_options) with
    /// [`CompilerOptions`](crate::options::CompilerOptions) for type safety.
    pub fn compiler_options(&mut self, options: &str) -> crate::Result<&mut Self> {
        let rc = unsafe {
            ffi::regor_set_compiler_options(self.ctx, options.as_ptr().cast(), options.len())
        };
        check(self.ctx, rc)?;
        Ok(self)
    }

    /// Compile a model, collecting the output via an internal buffer.
    ///
    /// Returns the compiled bytes as an [`Output`].
    pub fn compile(&mut self, format: InputFormat, model: &[u8]) -> crate::Result<Output> {
        let mut buf: Vec<u8> = Vec::new();
        let buf_ptr: *mut Vec<u8> = &mut buf;

        unsafe {
            ffi::regor_set_callback_arg(self.ctx, buf_ptr as *mut c_void);
        }

        let rc = unsafe {
            ffi::regor_compile(
                self.ctx,
                format.to_ffi(),
                model.as_ptr() as *const c_void,
                model.len(),
                Some(write_to_vec),
            )
        };
        check(self.ctx, rc)?;
        Ok(Output::new(buf))
    }

    /// Compile a model, delivering output through a caller-supplied writer.
    ///
    /// `user_arg` is forwarded opaquely to every `writer` invocation.
    ///
    /// # Safety
    ///
    /// `writer` must be safe to call with the provided `user_arg` for the
    /// duration of the compilation.
    pub unsafe fn compile_with_writer(
        &mut self,
        format: InputFormat,
        model: &[u8],
        user_arg: *mut c_void,
        writer: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> usize,
    ) -> crate::Result<()> {
        ffi::regor_set_callback_arg(self.ctx, user_arg);
        let rc = ffi::regor_compile(
            self.ctx,
            format.to_ffi(),
            model.as_ptr() as *const c_void,
            model.len(),
            Some(writer),
        );
        check(self.ctx, rc)
    }

    /// Retrieve the compiled output as a reference-counted blob.
    ///
    /// This is the C++ `IRegorBlob` interface — useful when you need to pass
    /// the output back into other regor APIs without copying.
    pub fn get_output_blob(&mut self) -> crate::Result<Blob> {
        let mut ptr: *mut ffi::IRegorBlob = std::ptr::null_mut();
        let rc = unsafe { ffi::regor_get_output(self.ctx, &mut ptr) };
        check(self.ctx, rc)?;
        Ok(unsafe { Blob::from_raw(self.ctx, ptr) })
    }

    /// Retrieve the performance report from the most recent compilation.
    pub fn perf_report(&self) -> crate::Result<PerfReport> {
        let mut raw = MaybeUninit::<ffi::regor_perf_report_t>::zeroed();
        let rc = unsafe { ffi::regor_get_perf_report(self.ctx, raw.as_mut_ptr()) };
        check(self.ctx, rc)?;
        Ok(unsafe { PerfReport::from_ffi(&raw.assume_init()) })
    }

    /// Retrieve TFLite operator constraints.
    ///
    /// Reports which operators cannot be accelerated and why.
    pub fn tflite_constraints(&self) -> crate::Result<ConstraintsReport> {
        let mut raw = MaybeUninit::<ffi::regor_operator_constraints_report_t>::zeroed();
        let rc = unsafe { ffi::regor_get_tflite_constraints(self.ctx, raw.as_mut_ptr()) };
        check(self.ctx, rc)?;
        Ok(unsafe { ConstraintsReport::from_ffi(&raw.assume_init()) })
    }

    /// Obtain the raw reporting interface pointer.
    ///
    /// Returns `None` if no reporting data is available.
    pub fn reporting_interface(&self) -> Option<*mut ffi::IRegorReporting> {
        let ptr = unsafe { ffi::regor_get_reporting_interface(self.ctx) };
        if ptr.is_null() {
            None
        } else {
            Some(ptr)
        }
    }

    /// Obtain a graph builder for constructing models programmatically.
    ///
    /// Returns `None` if the graph builder is not available.
    pub fn graph_builder(&mut self, name: &str) -> crate::Result<Option<*mut ffi::IGraphBuilder>> {
        let c_name = CString::new(name)?;
        let ptr = unsafe { ffi::regor_get_graph_builder(self.ctx, c_name.as_ptr()) };
        if ptr.is_null() {
            Ok(None)
        } else {
            Ok(Some(ptr))
        }
    }

    /// The raw FFI context handle, for advanced interop.
    pub fn raw_context(&self) -> ffi::regor_context_t {
        self.ctx
    }
}

/// Writer callback that appends data to a `Vec<u8>`.
unsafe extern "C" fn write_to_vec(
    user_arg: *mut c_void,
    data: *const c_void,
    size: usize,
) -> usize {
    let buf = &mut *(user_arg as *mut Vec<u8>);
    let slice = std::slice::from_raw_parts(data as *const u8, size);
    buf.extend_from_slice(slice);
    size
}
