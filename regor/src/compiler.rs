use std::ffi::CString;
use std::mem::MaybeUninit;
use std::os::raw::c_void;
use std::sync::{Mutex, MutexGuard, Once, PoisonError};

use regor_sys as ffi;

use crate::config::Architecture;
use crate::constraints::ConstraintsReport;
use crate::error::check;
use crate::format::InputFormat;
use crate::output::{Blob, Output};
use crate::perf::PerfReport;

static LOGGING_INIT: Once = Once::new();

/// Serialises **all** regor FFI calls across the process.
///
/// The C library is not safe to drive from more than one thread, even through
/// separate contexts. Its own mutex covers the context registry — creation,
/// lookup and destruction — but the compiler, scheduler and architecture code
/// underneath reach global state that nothing guards, and that state is shared
/// between contexts for as long as they exist. Holding a lock only around
/// `regor_compile` is therefore not enough: configuring one context while
/// another compiles corrupts it just the same.
///
/// So every entry point takes this lock. Concurrency is lost, which is a real
/// cost, but a safe wrapper cannot offer a faster contract than the library
/// underneath actually honours — and the failure mode being prevented is a
/// segfault, not a wrong answer.
///
/// The lock is poisoned only if a call panics while holding it, which for this
/// FFI means the library is in an unknown state. The guard is recovered rather
/// than propagating a poison error, since the alternative is that every later
/// call fails for a reason unrelated to its own input.
static REGOR_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    REGOR_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

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
    /// Configuration is buffered rather than pushed straight through to the C
    /// library, and applied inside [`Compiler::compile`] while `COMPILE_LOCK`
    /// is held. Applying it eagerly would let one thread reconfigure the shared
    /// state another thread is midway through compiling against.
    pending_system_config: Option<String>,
    pending_options: Option<String>,
}

// Each context owns its own C++ `Compiler`, so a context can be moved between
// threads. It is deliberately not `Sync`, and compilation additionally takes
// `COMPILE_LOCK` because the work underneath touches shared global state.
unsafe impl Send for Compiler {}

impl std::fmt::Debug for Compiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The context is an opaque handle, so there is nothing to show beyond
        // its identity. Implemented so `Compiler` can sit inside a `Result` that
        // callers `unwrap`.
        f.debug_struct("Compiler").field("ctx", &self.ctx).finish()
    }
}

impl Drop for Compiler {
    fn drop(&mut self) {
        let _guard = lock();
        unsafe { ffi::regor_destroy(self.ctx) };
    }
}

impl Compiler {
    /// Create a new compiler targeting `arch`.
    pub fn new(arch: Architecture) -> crate::Result<Self> {
        ensure_logging_initialized();
        let _guard = lock();
        let mut ctx: ffi::regor_context_t = 0;
        let rc = unsafe { ffi::regor_create(&mut ctx, arch.as_cstr().as_ptr()) };
        check(ctx, rc)?;
        Ok(Compiler {
            ctx,
            pending_system_config: None,
            pending_options: None,
        })
    }

    /// Apply a system configuration string.
    ///
    /// The document must carry an `[architecture]` section sizing the NPU and a
    /// `[vela]` section naming which of the Vela `.ini`'s `System_Config.*` and
    /// `Memory_Mode.*` sections to apply. Prefer
    /// [`set_system_config`](Self::set_system_config), which builds that for
    /// you.
    ///
    /// The configuration is recorded now and handed to the C library when
    /// [`compile`](Self::compile) runs, so any error in it surfaces there.
    pub fn system_config(&mut self, config: &str) -> crate::Result<&mut Self> {
        self.pending_system_config = Some(config.to_owned());
        Ok(self)
    }

    /// Apply a system configuration built with
    /// [`SystemConfig`](crate::options::SystemConfig).
    ///
    /// This is how the accelerator is selected; there is no compiler option for
    /// it.
    ///
    /// ```no_run
    /// # use regor::{Compiler, options::*};
    /// let system = SystemConfig::new(AcceleratorConfig::EthosU55_256)
    ///     .system_config_name("Ethos_U55_High_End_Embedded")
    ///     .memory_mode_name("Shared_Sram")
    ///     .vela_ini(std::fs::read_to_string("vela.ini")?)
    ///     .build();
    ///
    /// let mut c = Compiler::new(AcceleratorConfig::EthosU55_256.architecture())?;
    /// c.set_system_config(&system)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn set_system_config(&mut self, config: &str) -> crate::Result<&mut Self> {
        self.system_config(config)
    }

    /// Apply typed compiler options built with
    /// [`CompilerOptions`](crate::options::CompilerOptions).
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
    ///
    /// The options are recorded now and handed to the C library when
    /// [`compile`](Self::compile) runs, so any error in them surfaces there.
    pub fn compiler_options(&mut self, options: &str) -> crate::Result<&mut Self> {
        self.pending_options = Some(options.to_owned());
        Ok(self)
    }

    /// Push the buffered configuration into the C library.
    ///
    /// Must be called with `COMPILE_LOCK` held: it mutates the state that
    /// compilation then reads.
    fn apply_pending(&mut self) -> crate::Result<()> {
        if let Some(config) = self.pending_system_config.take() {
            let rc = unsafe {
                ffi::regor_set_system_config(self.ctx, config.as_ptr().cast(), config.len())
            };
            check(self.ctx, rc)?;
        }
        if let Some(options) = self.pending_options.take() {
            let rc = unsafe {
                ffi::regor_set_compiler_options(self.ctx, options.as_ptr().cast(), options.len())
            };
            check(self.ctx, rc)?;
        }
        Ok(())
    }

    /// Compile a model, collecting the output via an internal buffer.
    ///
    /// Returns the compiled bytes as an [`Output`].
    ///
    /// Serialised process-wide; see [`REGOR_LOCK`].
    pub fn compile(&mut self, format: InputFormat, model: &[u8]) -> crate::Result<Output> {
        let mut buf: Vec<u8> = Vec::new();
        let buf_ptr: *mut Vec<u8> = &mut buf;

        let _guard = lock();
        self.apply_pending()?;

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
        let _guard = lock();
        self.apply_pending()?;

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
        let _guard = lock();
        let mut ptr: *mut ffi::IRegorBlob = std::ptr::null_mut();
        let rc = unsafe { ffi::regor_get_output(self.ctx, &mut ptr) };
        check(self.ctx, rc)?;
        Ok(unsafe { Blob::from_raw(self.ctx, ptr) })
    }

    /// Retrieve the performance report from the most recent compilation.
    pub fn perf_report(&self) -> crate::Result<PerfReport> {
        let _guard = lock();
        let mut raw = MaybeUninit::<ffi::regor_perf_report_t>::zeroed();
        let rc = unsafe { ffi::regor_get_perf_report(self.ctx, raw.as_mut_ptr()) };
        check(self.ctx, rc)?;
        Ok(unsafe { PerfReport::from_ffi(&raw.assume_init()) })
    }

    /// Retrieve TFLite operator constraints.
    ///
    /// Reports which operators cannot be accelerated and why.
    pub fn tflite_constraints(&self) -> crate::Result<ConstraintsReport> {
        let _guard = lock();
        let mut raw = MaybeUninit::<ffi::regor_operator_constraints_report_t>::zeroed();
        let rc = unsafe { ffi::regor_get_tflite_constraints(self.ctx, raw.as_mut_ptr()) };
        check(self.ctx, rc)?;
        Ok(unsafe { ConstraintsReport::from_ffi(&raw.assume_init()) })
    }

    /// Obtain the raw reporting interface pointer.
    ///
    /// Returns `None` if no reporting data is available.
    pub fn reporting_interface(&self) -> Option<*mut ffi::IRegorReporting> {
        let _guard = lock();
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
        let _guard = lock();
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
