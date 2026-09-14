//! Raw FFI bindings to ARM's regor compiler.
//!
//! This crate provides unsafe, low-level bindings matching `regor.h`,
//! `regor_interface.hpp`, and `regor_database.hpp`. Prefer the safe
//! `regor` wrapper crate for application code.

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use std::os::raw::{c_char, c_int, c_uint, c_void};

// ---------------------------------------------------------------------------
// Typedefs
// ---------------------------------------------------------------------------

pub type regor_context_t = c_int;

/// Callback invoked by `regor_compile` to write output data.
///
/// Returns the number of bytes successfully written.
pub type regor_writer_t =
    Option<unsafe extern "C" fn(user_arg: *mut c_void, data: *const c_void, size: usize) -> usize>;

/// Callback invoked by the logging subsystem.
pub type regor_log_writer_t = Option<unsafe extern "C" fn(data: *const c_void, size: usize)>;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Input model format.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum regor_format_t {
    REGOR_INPUTFORMAT_GRAPHAPI = 0,
    REGOR_INPUTFORMAT_TFLITE = 1,
    REGOR_INPUTFORMAT_TOSA = 2,
}

/// Logging output format.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum regor_logging_format_t {
    REGOR_LOG_FORMAT_TEXT = 0,
    REGOR_LOG_FORMAT_TERMINAL = 1,
}

/// Tensor type tag inside a raw tensor header.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum raw_tensor_type_t {
    RAW_TENSOR_TYPE_COMMAND_STREAM = 0,
    RAW_TENSOR_TYPE_READ_ONLY = 1,
    RAW_TENSOR_TYPE_SCRATCH = 2,
    RAW_TENSOR_TYPE_SCRATCH_FAST = 3,
    RAW_TENSOR_TYPE_INPUT = 4,
    RAW_TENSOR_TYPE_OUTPUT = 5,
    RAW_TENSOR_TYPE_VARIABLE = 6,
}

/// Memory region a tensor resides in.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum raw_tensor_region_t {
    RAW_TENSOR_REGION_WEIGHTS = 0,
    RAW_TENSOR_REGION_SCRATCH = 1,
    RAW_TENSOR_REGION_SCRATCH_FAST = 2,
}

/// Bitflags on a raw tensor.
pub type regor_raw_tensor_flags_t = c_uint;
pub const REGOR_RAW_TENSOR_FLAG_HAS_QUANTIZATION: regor_raw_tensor_flags_t = 1 << 0;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const REGOR_PERF_NAME_MAX: usize = 32;
pub const REGOR_MAX_CONSTRAINTS: usize = 32;
pub const REGOR_CONSTRAINT_MAX_LENGTH: usize = 512;

pub const REGOR_ARCH_ETHOSU55: &[u8] = b"EthosU55\0";
pub const REGOR_ARCH_ETHOSU65: &[u8] = b"EthosU65\0";
pub const REGOR_ARCH_ETHOSU85: &[u8] = b"EthosU85\0";

// ---------------------------------------------------------------------------
// Performance / reporting structs
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone)]
pub struct regor_memory_access_perf_t {
    pub memory_name: [c_char; REGOR_PERF_NAME_MAX],
    pub access_type: [c_char; REGOR_PERF_NAME_MAX],
    pub bytes_read: i64,
    pub bytes_written: i64,
    pub access_cycles: i64,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct regor_peak_memory_usage_t {
    pub memory_name: [c_char; REGOR_PERF_NAME_MAX],
    pub peak_usage: i64,
    pub total_access_cycles: i64,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct regor_perf_report_t {
    pub npu_cycles: i64,
    pub cpu_cycles: i64,
    pub total_cycles: i64,
    pub mac_count: i64,
    pub cpu_ops: i64,
    pub npu_ops: i64,
    pub cascaded_ops: i64,
    pub cascades: i64,
    pub original_weights: i64,
    pub encoded_weights: i64,
    pub read_only_peak_usage: c_int,
    pub access_count: c_int,
    pub memory: c_int,
    pub num_memories: c_int,
    pub staging_memory: c_int,
    pub peak_usages: [regor_peak_memory_usage_t; 4],
    pub access: *mut regor_memory_access_perf_t,
}

// ---------------------------------------------------------------------------
// Operator constraints
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone)]
pub struct regor_operator_constraints_t {
    pub operator_name: [c_char; REGOR_PERF_NAME_MAX],
    pub constraint: [[c_char; REGOR_CONSTRAINT_MAX_LENGTH]; REGOR_MAX_CONSTRAINTS],
    pub constraints: c_int,
}

#[repr(C)]
#[derive(Debug)]
pub struct regor_operator_constraints_report_t {
    pub op_constraints: *mut regor_operator_constraints_t,
    pub operators: c_int,
}

// ---------------------------------------------------------------------------
// Raw tensor header (binary output parsing)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct regor_raw_tensor_header_t {
    pub tensor_type: raw_tensor_type_t,
    pub region: raw_tensor_region_t,
    pub flags: regor_raw_tensor_flags_t,
    pub offset: u64,
    pub size: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct regor_raw_quantization_t {
    pub count: u32,
    // Followed in memory by `count` f32 scales, then `count` i32 zero-points.
    // Access through pointer arithmetic from the struct address.
}

// ---------------------------------------------------------------------------
// Opaque C++ interface pointers
// ---------------------------------------------------------------------------

/// Opaque handle to `regor::IRegorBlob` (C++ COM-style interface).
#[repr(C)]
pub struct IRegorBlob {
    _opaque: [u8; 0],
}

/// Opaque handle to `regor::IRegorReporting` (C++ interface).
#[repr(C)]
pub struct IRegorReporting {
    _opaque: [u8; 0],
}

/// Opaque handle to the graph builder interface.
#[repr(C)]
pub struct IGraphBuilder {
    _opaque: [u8; 0],
}

// ---------------------------------------------------------------------------
// Extern "C" API
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Create a compiler context for the given architecture.
    ///
    /// `arch_name` must be one of `REGOR_ARCH_ETHOSU55`, `REGOR_ARCH_ETHOSU65`,
    /// or `REGOR_ARCH_ETHOSU85` as a NUL-terminated C string.
    ///
    /// Returns 0 on success, non-zero on failure.
    pub fn regor_create(ctx: *mut regor_context_t, arch_name: *const c_char) -> c_int;

    /// Destroy a previously created compiler context.
    pub fn regor_destroy(ctx: regor_context_t);

    /// Apply a TOML/INI system configuration string.
    pub fn regor_set_system_config(
        ctx: regor_context_t,
        config_text: *const c_char,
        length: usize,
    ) -> c_int;

    /// Apply compiler option overrides.
    pub fn regor_set_compiler_options(
        ctx: regor_context_t,
        config_text: *const c_char,
        length: usize,
    ) -> c_int;

    /// Attach an opaque user argument passed through to writer callbacks.
    pub fn regor_set_callback_arg(ctx: regor_context_t, user_arg: *mut c_void) -> c_int;

    /// Compile a model.
    ///
    /// `fmt` selects the input format, `input` / `in_size` point to the model
    /// bytes, and `write_func` receives the compiled output.
    pub fn regor_compile(
        ctx: regor_context_t,
        fmt: regor_format_t,
        input: *const c_void,
        in_size: usize,
        write_func: regor_writer_t,
    ) -> c_int;

    /// Retrieve the compiled output as a blob.
    pub fn regor_get_output(ctx: regor_context_t, blob: *mut *mut IRegorBlob) -> c_int;

    /// Retrieve the last error message.
    ///
    /// On entry `*length` is the buffer capacity. On return it holds the
    /// actual message length (excluding NUL).
    pub fn regor_get_error(ctx: regor_context_t, text: *mut c_char, length: *mut usize) -> c_int;

    /// Free data previously allocated by regor on behalf of this context.
    pub fn regor_free_data(ctx: regor_context_t, data: *const c_void) -> c_int;

    /// Install a global log callback.
    pub fn regor_set_logging(log_writer: regor_log_writer_t, filter_mask: c_uint) -> c_int;

    /// Install a global log callback with format control.
    pub fn regor_set_logging_ex(
        log_writer: regor_log_writer_t,
        filter_mask: c_uint,
        format: c_uint,
    ) -> c_int;

    /// Retrieve the performance report from the last compilation.
    pub fn regor_get_perf_report(ctx: regor_context_t, report: *mut regor_perf_report_t) -> c_int;

    /// Retrieve TFLite operator constraint information.
    pub fn regor_get_tflite_constraints(
        ctx: regor_context_t,
        report: *mut regor_operator_constraints_report_t,
    ) -> c_int;

    /// Obtain the reporting / database interface for the last compilation.
    pub fn regor_get_reporting_interface(ctx: regor_context_t) -> *mut IRegorReporting;

    /// Obtain a graph builder for constructing models programmatically.
    pub fn regor_get_graph_builder(
        ctx: regor_context_t,
        graph_name: *const c_char,
    ) -> *mut IGraphBuilder;
}
