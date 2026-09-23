//! Typed builders for the two configuration strings regor accepts.
//!
//! regor is configured through two separate INI documents, and they are not
//! interchangeable:
//!
//! - [`SystemConfig`] describes the *hardware*, and is passed to
//!   [`Compiler::system_config`](crate::Compiler::system_config). It carries the
//!   accelerator size and selects which sections of a Vela `.ini` to apply.
//! - [`CompilerOptions`] describes *how to compile*, and is passed to
//!   [`Compiler::set_options`](crate::Compiler::set_options).
//!
//! Both are sectioned INI, and several options are spelled differently from the
//! equivalent Vela command-line switch (`--optimise` is `optimize`,
//! `--arena-cache-size` is `arena_size_limit`, and the various `--disable-*`
//! switches collapse into a single `disable_feature` list). These builders own
//! that translation so callers can use the names they already know.
//!
//! # Example
//!
//! ```no_run
//! use regor::{Compiler, InputFormat};
//! use regor::options::*;
//!
//! let accelerator = AcceleratorConfig::EthosU55_256;
//!
//! let system = SystemConfig::new(accelerator)
//!     .system_config_name("Ethos_U55_High_End_Embedded")
//!     .memory_mode_name("Shared_Sram")
//!     .vela_ini(std::fs::read_to_string("vela.ini")?)
//!     .build();
//!
//! let options = CompilerOptions::new()
//!     .optimize(Optimize::Performance)
//!     .arena_cache_size(2 * 1024 * 1024)
//!     .build()?;
//!
//! let mut compiler = Compiler::new(accelerator.architecture())?;
//! compiler.system_config(&system)?;
//! compiler.set_options(&options)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::fmt::Write;

/// Optimization strategy (`[scheduler] optimize`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Optimize {
    /// Prioritise inference speed; uses `arena_cache_size` as a memory target.
    Performance,
    /// Prioritise lower memory usage.
    Size,
}

impl Optimize {
    fn as_str(self) -> &'static str {
        match self {
            Self::Performance => "Performance",
            Self::Size => "Size",
        }
    }
}

/// Tensor allocation algorithm (`[scheduler] tensor_allocator`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorAllocator {
    /// Iterative search; usually the tightest packing.
    HillClimb,
    /// Single linear pass; fastest to run.
    LinearAlloc,
}

impl TensorAllocator {
    fn as_str(self) -> &'static str {
        match self {
            Self::HillClimb => "HillClimb",
            Self::LinearAlloc => "LinearAlloc",
        }
    }
}

/// Custom operator payload format version (`[compiler] cop_format`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CopFormat {
    Cop1,
    /// Required for [`CompilerOptions::separate_io_regions`].
    Cop2,
}

impl CopFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cop1 => "COP1",
            Self::Cop2 => "COP2",
        }
    }
}

/// Output format for the compiled model (`[compiler] output_format`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    /// Standard `.tflite` with custom operator nodes.
    TfLite,
    /// Raw command stream with directly accessible arrays. No CPU fallback.
    Raw,
}

impl OutputFormat {
    /// The name regor matches, which it compares **case-sensitively** — a
    /// lowercase `tflite` is not recognised.
    fn as_str(self) -> &'static str {
        match self {
            Self::TfLite => "TFLite",
            Self::Raw => "Raw",
        }
    }
}

bitflags::bitflags! {
    /// Scheduler features that can be switched off (`[scheduler] disable_feature`).
    ///
    /// These are the names regor's own scheduler uses. Where Vela's CLI differs,
    /// the switch that maps to each one is noted.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct SchedulerFeature: u16 {
        /// Double-buffering of weights. Vela's `--disable-buffering`.
        const WEIGHT_BUFFERING = 1 << 0;
        /// Fusing operators to keep intermediates on-chip. Vela's
        /// `--disable-cascading`.
        const CASCADING = 1 << 1;
        /// Operator grouping. Vela's `--disable-chaining`.
        const GROUPING = 1 << 2;
        /// Fast weight decoder. Vela's `--disable-fwd`.
        const FWD = 1 << 3;
        /// Exploiting weight sparsity. Not exposed by Vela's CLI.
        const SPARSITY = 1 << 4;
        /// Staging feature maps in the faster memory region. Not exposed by
        /// Vela's CLI.
        const FM_STAGING = 1 << 5;
        /// Reusing an input feature map's buffer for the output. Vela's
        /// `--disable-ifm-reuse`.
        const REUSE_IFM = 1 << 6;
    }
}

impl SchedulerFeature {
    /// Serialise to the `|`-separated list regor parses.
    ///
    /// regor's flag lexer accepts `|` (and `^`) as separators — **not** commas,
    /// which it rejects outright, leaving the whole option silently unapplied.
    fn to_regor_value(self) -> String {
        const NAMES: &[(SchedulerFeature, &str)] = &[
            (SchedulerFeature::WEIGHT_BUFFERING, "WeightBuffering"),
            (SchedulerFeature::CASCADING, "Cascading"),
            (SchedulerFeature::GROUPING, "Grouping"),
            (SchedulerFeature::FWD, "FWD"),
            (SchedulerFeature::SPARSITY, "Sparsity"),
            (SchedulerFeature::FM_STAGING, "FMStaging"),
            (SchedulerFeature::REUSE_IFM, "ReuseIFM"),
        ];

        NAMES
            .iter()
            .filter(|(flag, _)| self.contains(*flag))
            .map(|(_, name)| *name)
            .collect::<Vec<_>>()
            .join("|")
    }
}

/// Accelerator variant with MAC count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum AcceleratorConfig {
    EthosU55_32,
    EthosU55_64,
    EthosU55_128,
    EthosU55_256,
    EthosU65_256,
    EthosU65_512,
    EthosU85_128,
    EthosU85_256,
    EthosU85_512,
    EthosU85_1024,
    EthosU85_2048,
}

impl AcceleratorConfig {
    /// Every accelerator this crate knows about.
    pub const ALL: &'static [AcceleratorConfig] = &[
        Self::EthosU55_32,
        Self::EthosU55_64,
        Self::EthosU55_128,
        Self::EthosU55_256,
        Self::EthosU65_256,
        Self::EthosU65_512,
        Self::EthosU85_128,
        Self::EthosU85_256,
        Self::EthosU85_512,
        Self::EthosU85_1024,
        Self::EthosU85_2048,
    ];

    /// The Vela `--accelerator-config` spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EthosU55_32 => "ethos-u55-32",
            Self::EthosU55_64 => "ethos-u55-64",
            Self::EthosU55_128 => "ethos-u55-128",
            Self::EthosU55_256 => "ethos-u55-256",
            Self::EthosU65_256 => "ethos-u65-256",
            Self::EthosU65_512 => "ethos-u65-512",
            Self::EthosU85_128 => "ethos-u85-128",
            Self::EthosU85_256 => "ethos-u85-256",
            Self::EthosU85_512 => "ethos-u85-512",
            Self::EthosU85_1024 => "ethos-u85-1024",
            Self::EthosU85_2048 => "ethos-u85-2048",
        }
    }

    /// The [`crate::Architecture`] this accelerator belongs to.
    pub fn architecture(self) -> crate::Architecture {
        match self {
            Self::EthosU55_32 | Self::EthosU55_64 | Self::EthosU55_128 | Self::EthosU55_256 => {
                crate::Architecture::EthosU55
            }
            Self::EthosU65_256 | Self::EthosU65_512 => crate::Architecture::EthosU65,
            Self::EthosU85_128
            | Self::EthosU85_256
            | Self::EthosU85_512
            | Self::EthosU85_1024
            | Self::EthosU85_2048 => crate::Architecture::EthosU85,
        }
    }

    /// MAC units per cycle.
    ///
    /// This is what actually selects the accelerator inside regor: it matches
    /// the architecture's configuration table on the MAC count alone.
    pub fn macs(self) -> u32 {
        match self {
            Self::EthosU55_32 => 32,
            Self::EthosU55_64 => 64,
            Self::EthosU55_128 | Self::EthosU85_128 => 128,
            Self::EthosU55_256 | Self::EthosU65_256 | Self::EthosU85_256 => 256,
            Self::EthosU65_512 | Self::EthosU85_512 => 512,
            Self::EthosU85_1024 => 1024,
            Self::EthosU85_2048 => 2048,
        }
    }

    /// NPU cores in this configuration.
    ///
    /// All but the largest U65 and U85 variants are single-core; those two are
    /// built as two cores driven together.
    pub fn cores(self) -> u32 {
        match self {
            Self::EthosU65_512 | Self::EthosU85_2048 => 2,
            _ => 1,
        }
    }
}

impl std::fmt::Display for AcceleratorConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for AcceleratorConfig {
    type Err = UnknownAccelerator;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|a| a.as_str() == s)
            .ok_or_else(|| UnknownAccelerator(s.to_owned()))
    }
}

/// [`AcceleratorConfig::from_str`] was given a name it does not know.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownAccelerator(pub String);

impl std::fmt::Display for UnknownAccelerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown accelerator configuration '{}'", self.0)
    }
}

impl std::error::Error for UnknownAccelerator {}

/// A section of regor's compiler options document.
///
/// Only needed for [`CompilerOptions::raw`]; the typed setters pick the right
/// section themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Section {
    /// Output and command-stream options.
    Compiler,
    /// Scheduling, allocation and memory options.
    Scheduler,
    /// Graph-optimiser options.
    Graph,
    /// Tracing.
    Debug,
}

/// Builder for the compiler options document.
///
/// Every option is left unset by default, which means "whatever regor's own
/// default is" rather than "off" — so a value is only ever written because a
/// caller asked for it.
///
/// ```
/// # use regor::options::*;
/// let opts = CompilerOptions::new()
///     .optimize(Optimize::Performance)
///     .arena_cache_size(2 * 1024 * 1024)
///     .build()
///     .unwrap();
///
/// assert!(opts.contains("[scheduler]"));
/// assert!(opts.contains("optimize=Performance"));
/// assert!(opts.contains("arena_size_limit=2097152"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct CompilerOptions {
    // [compiler]
    output_format: Option<OutputFormat>,
    cop_format: Option<CopFormat>,
    enable_debug_db: Option<bool>,
    perf_report: Option<bool>,
    verbose_high_level_command_stream: Option<bool>,
    verbose_register_command_stream: Option<bool>,
    // [scheduler]
    optimize: Option<Optimize>,
    arena_cache_size: Option<u64>,
    tensor_allocator: Option<TensorAllocator>,
    cpu_tensor_alignment: Option<u32>,
    separate_io_regions: Option<bool>,
    disable_features: SchedulerFeature,
    verbose_schedule: Option<bool>,
    verbose_allocation: Option<bool>,
    // [graph]
    ignore_ops: Vec<String>,
    verbose_graph: Option<bool>,
    verbose_quantization: Option<bool>,
    softmax_int16_neg_exp_range: Option<f32>,
    // [debug]
    trace: Option<bool>,
    // escape hatch
    raw: Vec<(Section, String, String)>,
}

impl CompilerOptions {
    pub fn new() -> Self {
        Self::default()
    }

    // ------------------------------------------------------------------
    // [compiler]
    // ------------------------------------------------------------------

    /// Format the compiled model is written in.
    pub fn output_format(mut self, fmt: OutputFormat) -> Self {
        self.output_format = Some(fmt);
        self
    }

    /// Custom operator payload format version.
    pub fn cop_format(mut self, fmt: CopFormat) -> Self {
        self.cop_format = Some(fmt);
        self
    }

    /// Emit the debug database mapping command-stream offsets to operators.
    ///
    /// This is also what makes regor report the operators it could not offload,
    /// which Vela exposes separately as `--show-cpu-operations`.
    pub fn enable_debug_db(mut self, enable: bool) -> Self {
        self.enable_debug_db = Some(enable);
        self
    }

    /// Emit the performance report.
    pub fn perf_report(mut self, enable: bool) -> Self {
        self.perf_report = Some(enable);
        self
    }

    /// Dump the high-level command stream.
    pub fn verbose_high_level_command_stream(mut self, enable: bool) -> Self {
        self.verbose_high_level_command_stream = Some(enable);
        self
    }

    /// Dump the register-level command stream.
    pub fn verbose_register_command_stream(mut self, enable: bool) -> Self {
        self.verbose_register_command_stream = Some(enable);
        self
    }

    // ------------------------------------------------------------------
    // [scheduler]
    // ------------------------------------------------------------------

    /// Whether to optimise for inference speed or memory footprint.
    ///
    /// Written as `optimize`, which is how regor spells it.
    pub fn optimize(mut self, strategy: Optimize) -> Self {
        self.optimize = Some(strategy);
        self
    }

    /// Arena/cache memory budget in bytes.
    ///
    /// Written as `arena_size_limit`. Under [`Optimize::Performance`] regor
    /// treats it as a target rather than a hard cap.
    pub fn arena_cache_size(mut self, bytes: u64) -> Self {
        self.arena_cache_size = Some(bytes);
        self
    }

    /// Which algorithm packs tensors into the arena.
    pub fn tensor_allocator(mut self, alloc: TensorAllocator) -> Self {
        self.tensor_allocator = Some(alloc);
        self
    }

    /// CPU tensor alignment in bytes.
    ///
    /// Validated at [`build`](Self::build) time: must be a power of two and at
    /// least 16.
    pub fn cpu_tensor_alignment(mut self, alignment: u32) -> Self {
        self.cpu_tensor_alignment = Some(alignment);
        self
    }

    /// Allocate inputs/outputs in separate regions instead of scratch.
    ///
    /// Requires [`CopFormat::Cop2`], checked at [`build`](Self::build) time.
    pub fn separate_io_regions(mut self, enable: bool) -> Self {
        self.separate_io_regions = Some(enable);
        self
    }

    /// Switch off scheduler features.
    ///
    /// Replaces any previously set features; combine them in one call:
    ///
    /// ```
    /// # use regor::options::*;
    /// let opts = CompilerOptions::new()
    ///     .disable_features(SchedulerFeature::CASCADING | SchedulerFeature::FWD)
    ///     .build()
    ///     .unwrap();
    /// assert!(opts.contains("disable_feature=Cascading|FWD"));
    /// ```
    pub fn disable_features(mut self, features: SchedulerFeature) -> Self {
        self.disable_features = features;
        self
    }

    /// Dump the schedule regor chose.
    pub fn verbose_schedule(mut self, enable: bool) -> Self {
        self.verbose_schedule = Some(enable);
        self
    }

    /// Dump the tensor arena allocation.
    pub fn verbose_allocation(mut self, enable: bool) -> Self {
        self.verbose_allocation = Some(enable);
        self
    }

    // ------------------------------------------------------------------
    // [graph]
    // ------------------------------------------------------------------

    /// TFLite operators to force onto the CPU.
    ///
    /// Written comma-separated, which is what regor's operator-list parser
    /// expects — unlike [`disable_features`](Self::disable_features).
    pub fn ignore_ops<I, S>(mut self, ops: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.ignore_ops = ops.into_iter().map(Into::into).collect();
        self
    }

    /// Dump the optimised graph.
    pub fn verbose_graph(mut self, enable: bool) -> Self {
        self.verbose_graph = Some(enable);
        self
    }

    /// Dump the quantisation decisions.
    pub fn verbose_quantization(mut self, enable: bool) -> Self {
        self.verbose_quantization = Some(enable);
        self
    }

    /// Negative exponent range for an int16 softmax.
    ///
    /// Validated at [`build`](Self::build) time: regor requires
    /// `0 < range < 65535`.
    pub fn softmax_int16_neg_exp_range(mut self, range: f32) -> Self {
        self.softmax_int16_neg_exp_range = Some(range);
        self
    }

    // ------------------------------------------------------------------
    // [debug]
    // ------------------------------------------------------------------

    /// Raise the log filter to include trace output.
    pub fn trace(mut self, enable: bool) -> Self {
        self.trace = Some(enable);
        self
    }

    // ------------------------------------------------------------------
    // Escape hatch
    // ------------------------------------------------------------------

    /// Set an arbitrary `key=value` in `section`, for options this API does not
    /// model.
    ///
    /// Unknown keys are ignored by regor rather than rejected, so a typo here is
    /// silent.
    pub fn raw(mut self, section: Section, key: &str, value: &str) -> Self {
        self.raw.push((section, key.to_owned(), value.to_owned()));
        self
    }

    // ------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------

    /// Serialise to the sectioned INI `regor_set_compiler_options` expects.
    ///
    /// # Errors
    ///
    /// Returns [`OptionsError`] if `cpu_tensor_alignment` is not a power of two
    /// of at least 16, if `softmax_int16_neg_exp_range` is out of range, or if
    /// `separate_io_regions` is set without [`CopFormat::Cop2`].
    pub fn build(&self) -> Result<String, OptionsError> {
        if let Some(a) = self.cpu_tensor_alignment {
            if a < 16 || !a.is_power_of_two() {
                return Err(OptionsError::InvalidAlignment(a));
            }
        }

        if let Some(r) = self.softmax_int16_neg_exp_range {
            if !(r > 0.0 && r < 65535.0) {
                return Err(OptionsError::SoftmaxRangeOutOfRange(r));
            }
        }

        if self.separate_io_regions == Some(true) && self.cop_format != Some(CopFormat::Cop2) {
            return Err(OptionsError::SeparateIoRequiresCop2);
        }

        let mut out = String::new();

        out.push_str("[compiler]\n");
        if let Some(v) = self.output_format {
            let _ = writeln!(out, "output_format={}", v.as_str());
        }
        if let Some(v) = self.cop_format {
            let _ = writeln!(out, "cop_format={}", v.as_str());
        }
        if let Some(v) = self.enable_debug_db {
            let _ = writeln!(out, "enable_db={v}");
        }
        if let Some(v) = self.perf_report {
            let _ = writeln!(out, "perf_report={v}");
        }
        if let Some(v) = self.verbose_high_level_command_stream {
            let _ = writeln!(out, "verbose_high_level_command_stream={v}");
        }
        if let Some(v) = self.verbose_register_command_stream {
            let _ = writeln!(out, "verbose_register_command_stream={v}");
        }
        self.write_raw(&mut out, Section::Compiler);

        out.push_str("\n[scheduler]\n");
        if let Some(v) = self.optimize {
            let _ = writeln!(out, "optimize={}", v.as_str());
        }
        if let Some(v) = self.arena_cache_size {
            let _ = writeln!(out, "arena_size_limit={v}");
        }
        if let Some(v) = self.tensor_allocator {
            let _ = writeln!(out, "tensor_allocator={}", v.as_str());
        }
        if let Some(v) = self.cpu_tensor_alignment {
            let _ = writeln!(out, "cpu_tensor_alignment={v}");
        }
        if let Some(v) = self.separate_io_regions {
            let _ = writeln!(out, "separate_io_regions={v}");
        }
        if !self.disable_features.is_empty() {
            let _ = writeln!(
                out,
                "disable_feature={}",
                self.disable_features.to_regor_value()
            );
        }
        if let Some(v) = self.verbose_schedule {
            let _ = writeln!(out, "verbose={v}");
        }
        if let Some(v) = self.verbose_allocation {
            let _ = writeln!(out, "verbose_allocation={v}");
        }
        self.write_raw(&mut out, Section::Scheduler);

        out.push_str("\n[graph]\n");
        if !self.ignore_ops.is_empty() {
            let _ = writeln!(out, "ignore_ops={}", self.ignore_ops.join(","));
        }
        if let Some(v) = self.verbose_graph {
            let _ = writeln!(out, "verbose={v}");
        }
        if let Some(v) = self.verbose_quantization {
            let _ = writeln!(out, "verbose_quantization={v}");
        }
        if let Some(v) = self.softmax_int16_neg_exp_range {
            let _ = writeln!(out, "softmax_int16_neg_exp_range={v}");
        }
        self.write_raw(&mut out, Section::Graph);

        let has_debug_raw = self.raw.iter().any(|(s, _, _)| *s == Section::Debug);
        if self.trace.is_some() || has_debug_raw {
            out.push_str("\n[debug]\n");
            if let Some(v) = self.trace {
                let _ = writeln!(out, "trace={v}");
            }
            self.write_raw(&mut out, Section::Debug);
        }

        Ok(out)
    }

    fn write_raw(&self, out: &mut String, section: Section) {
        for (_, k, v) in self.raw.iter().filter(|(s, _, _)| *s == section) {
            let _ = writeln!(out, "{k}={v}");
        }
    }
}

/// Builder for the system configuration document.
///
/// This is how the accelerator reaches regor. There is deliberately no
/// accelerator setter on [`CompilerOptions`]: regor sizes the NPU from the
/// `[architecture]` section written here, and would ignore an
/// `accelerator_config` compiler option entirely.
///
/// ```
/// # use regor::options::*;
/// let system = SystemConfig::new(AcceleratorConfig::EthosU55_256)
///     .system_config_name("Ethos_U55_High_End_Embedded")
///     .memory_mode_name("Shared_Sram")
///     .vela_ini("[System_Config.Ethos_U55_High_End_Embedded]\ncore_clock=500e6\n")
///     .build();
///
/// assert!(system.starts_with("[architecture]\nmacs=256\ncores=1\n"));
/// assert!(system.contains("system_config_name=Ethos_U55_High_End_Embedded"));
/// ```
#[derive(Debug, Clone)]
pub struct SystemConfig {
    accelerator: AcceleratorConfig,
    system_config_name: Option<String>,
    memory_mode_name: Option<String>,
    vela_ini: String,
}

impl SystemConfig {
    /// Start a system configuration for `accelerator`.
    pub fn new(accelerator: AcceleratorConfig) -> Self {
        Self {
            accelerator,
            system_config_name: None,
            memory_mode_name: None,
            vela_ini: String::new(),
        }
    }

    /// The `System_Config.*` section of the Vela `.ini` to apply.
    ///
    /// When unset, regor takes the first one it finds.
    pub fn system_config_name(mut self, name: impl Into<String>) -> Self {
        self.system_config_name = Some(name.into());
        self
    }

    /// The `Memory_Mode.*` section of the Vela `.ini` to apply.
    ///
    /// When unset, regor takes the first one it finds.
    pub fn memory_mode_name(mut self, name: impl Into<String>) -> Self {
        self.memory_mode_name = Some(name.into());
        self
    }

    /// The Vela configuration `.ini` supplying those sections, appended
    /// verbatim.
    pub fn vela_ini(mut self, ini: impl Into<String>) -> Self {
        self.vela_ini = ini.into();
        self
    }

    /// The accelerator this configuration describes.
    pub fn accelerator(&self) -> AcceleratorConfig {
        self.accelerator
    }

    /// Serialise to the document `regor_set_system_config` expects.
    pub fn build(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "[architecture]\nmacs={}\ncores={}",
            self.accelerator.macs(),
            self.accelerator.cores()
        );
        out.push_str("[vela]\n");
        if let Some(v) = &self.system_config_name {
            let _ = writeln!(out, "system_config_name={v}");
        }
        if let Some(v) = &self.memory_mode_name {
            let _ = writeln!(out, "memory_mode_name={v}");
        }
        out.push_str(&self.vela_ini);
        out
    }
}

/// Error from option validation.
#[derive(Debug, Clone, PartialEq)]
pub enum OptionsError {
    /// `cpu_tensor_alignment` must be a power of two >= 16.
    InvalidAlignment(u32),
    /// `softmax_int16_neg_exp_range` must be in `(0, 65535)`.
    SoftmaxRangeOutOfRange(f32),
    /// `separate_io_regions` requires `cop_format = COP2`.
    SeparateIoRequiresCop2,
}

impl std::fmt::Display for OptionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptionsError::InvalidAlignment(v) => {
                write!(
                    f,
                    "cpu_tensor_alignment must be a power of two >= 16, got {v}"
                )
            }
            OptionsError::SoftmaxRangeOutOfRange(v) => {
                write!(
                    f,
                    "softmax_int16_neg_exp_range must be in (0, 65535), got {v}"
                )
            }
            OptionsError::SeparateIoRequiresCop2 => {
                write!(f, "separate_io_regions requires cop_format = COP2")
            }
        }
    }
}

impl std::error::Error for OptionsError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The section a key was written under, for assertions.
    fn section_of<'a>(doc: &'a str, key: &str) -> Option<&'a str> {
        let mut current = None;
        for line in doc.lines() {
            let line = line.trim();
            if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                current = Some(name);
            } else if line.split('=').next() == Some(key) {
                return current;
            }
        }
        None
    }

    #[test]
    fn options_use_regors_own_key_names() {
        let s = CompilerOptions::new()
            .optimize(Optimize::Performance)
            .arena_cache_size(2_097_152)
            .build()
            .unwrap();
        // Vela spells these `--optimise` and `--arena-cache-size`; regor does not.
        assert_eq!(section_of(&s, "optimize"), Some("scheduler"));
        assert_eq!(section_of(&s, "arena_size_limit"), Some("scheduler"));
        assert!(s.contains("optimize=Performance"), "{s}");
        assert!(s.contains("arena_size_limit=2097152"), "{s}");
    }

    #[test]
    fn options_are_grouped_into_the_right_sections() {
        let s = CompilerOptions::new()
            .output_format(OutputFormat::Raw)
            .cop_format(CopFormat::Cop2)
            .tensor_allocator(TensorAllocator::LinearAlloc)
            .ignore_ops(["CONV_2D"])
            .trace(true)
            .build()
            .unwrap();

        assert_eq!(section_of(&s, "output_format"), Some("compiler"));
        assert_eq!(section_of(&s, "cop_format"), Some("compiler"));
        assert_eq!(section_of(&s, "tensor_allocator"), Some("scheduler"));
        assert_eq!(section_of(&s, "ignore_ops"), Some("graph"));
        assert_eq!(section_of(&s, "trace"), Some("debug"));
    }

    #[test]
    fn output_format_is_spelled_the_way_regor_matches_it() {
        // regor compares these case-sensitively, so `tflite` would be ignored.
        let s = CompilerOptions::new()
            .output_format(OutputFormat::TfLite)
            .build()
            .unwrap();
        assert!(s.contains("output_format=TFLite"), "{s}");

        let s = CompilerOptions::new()
            .output_format(OutputFormat::Raw)
            .build()
            .unwrap();
        assert!(s.contains("output_format=Raw"), "{s}");
    }

    #[test]
    fn disabled_features_are_pipe_separated() {
        // regor's flag lexer accepts `|`, not `,` — a comma makes it reject the
        // whole value.
        let s = CompilerOptions::new()
            .disable_features(SchedulerFeature::CASCADING | SchedulerFeature::FWD)
            .build()
            .unwrap();
        assert!(s.contains("disable_feature=Cascading|FWD"), "{s}");
        assert!(!s.contains(','), "{s}");
    }

    #[test]
    fn every_scheduler_feature_has_regors_name() {
        assert_eq!(
            SchedulerFeature::all().to_regor_value(),
            "WeightBuffering|Cascading|Grouping|FWD|Sparsity|FMStaging|ReuseIFM"
        );
    }

    #[test]
    fn no_disabled_features_writes_nothing() {
        let s = CompilerOptions::new().build().unwrap();
        assert!(!s.contains("disable_feature"), "{s}");
    }

    #[test]
    fn ignore_ops_is_comma_separated() {
        // Unlike `disable_feature`, the operator list really is comma separated.
        let s = CompilerOptions::new()
            .ignore_ops(["CONV_2D", "FULLY_CONNECTED"])
            .build()
            .unwrap();
        assert!(s.contains("ignore_ops=CONV_2D,FULLY_CONNECTED"), "{s}");
    }

    #[test]
    fn unset_options_are_not_written() {
        let s = CompilerOptions::new().build().unwrap();
        for key in [
            "optimize",
            "arena_size_limit",
            "tensor_allocator",
            "output_format",
            "cop_format",
            "cpu_tensor_alignment",
        ] {
            assert!(section_of(&s, key).is_none(), "{key} should be unset:\n{s}");
        }
    }

    #[test]
    fn alignment_validation() {
        for good in [16, 32, 64, 128] {
            assert!(
                CompilerOptions::new()
                    .cpu_tensor_alignment(good)
                    .build()
                    .is_ok(),
                "{good} should be accepted"
            );
        }
        for bad in [0, 8, 15, 24] {
            assert_eq!(
                CompilerOptions::new().cpu_tensor_alignment(bad).build(),
                Err(OptionsError::InvalidAlignment(bad))
            );
        }
    }

    #[test]
    fn softmax_range_validation() {
        assert!(CompilerOptions::new()
            .softmax_int16_neg_exp_range(8.0)
            .build()
            .is_ok());
        for bad in [0.0, -1.0, 65535.0, 70000.0] {
            assert!(
                CompilerOptions::new()
                    .softmax_int16_neg_exp_range(bad)
                    .build()
                    .is_err(),
                "{bad} should be rejected"
            );
        }
    }

    #[test]
    fn separate_io_requires_cop2() {
        assert_eq!(
            CompilerOptions::new().separate_io_regions(true).build(),
            Err(OptionsError::SeparateIoRequiresCop2)
        );

        assert!(CompilerOptions::new()
            .cop_format(CopFormat::Cop2)
            .separate_io_regions(true)
            .build()
            .is_ok());

        // Explicitly disabling it is fine without COP2.
        assert!(CompilerOptions::new()
            .separate_io_regions(false)
            .build()
            .is_ok());
    }

    #[test]
    fn raw_escape_hatch_targets_a_section() {
        let s = CompilerOptions::new()
            .raw(Section::Scheduler, "some_future_option", "42")
            .build()
            .unwrap();
        assert_eq!(section_of(&s, "some_future_option"), Some("scheduler"));
    }

    #[test]
    fn accelerator_architecture_consistency() {
        assert_eq!(
            AcceleratorConfig::EthosU55_256.architecture(),
            crate::Architecture::EthosU55,
        );
        assert_eq!(
            AcceleratorConfig::EthosU85_2048.architecture(),
            crate::Architecture::EthosU85,
        );
    }

    #[test]
    fn accelerator_macs_match_the_name() {
        for acc in AcceleratorConfig::ALL {
            let suffix = acc.as_str().rsplit('-').next().unwrap();
            assert_eq!(suffix.parse::<u32>().unwrap(), acc.macs(), "{acc} macs");
        }
    }

    #[test]
    fn accelerator_round_trips_through_its_name() {
        for acc in AcceleratorConfig::ALL {
            assert_eq!(acc.as_str().parse::<AcceleratorConfig>().unwrap(), *acc);
        }
        assert!("ethos-u99-1".parse::<AcceleratorConfig>().is_err());
    }

    #[test]
    fn dual_core_accelerators_report_two_cores() {
        assert_eq!(AcceleratorConfig::EthosU65_512.cores(), 2);
        assert_eq!(AcceleratorConfig::EthosU85_2048.cores(), 2);
        assert_eq!(AcceleratorConfig::EthosU55_256.cores(), 1);
        assert_eq!(AcceleratorConfig::EthosU85_1024.cores(), 1);
    }

    #[test]
    fn system_config_leads_with_the_architecture() {
        let s = SystemConfig::new(AcceleratorConfig::EthosU55_128)
            .system_config_name("Ethos_U55_Deep_Embedded")
            .memory_mode_name("Sram_Only")
            .vela_ini("[System_Config.Ethos_U55_Deep_Embedded]\ncore_clock=200e6\n")
            .build();

        assert!(s.starts_with("[architecture]\nmacs=128\ncores=1\n"), "{s}");
        assert!(s.contains("[vela]\n"), "{s}");
        assert!(
            s.contains("system_config_name=Ethos_U55_Deep_Embedded"),
            "{s}"
        );
        assert!(s.contains("memory_mode_name=Sram_Only"), "{s}");
        // The vela ini is appended unchanged.
        assert!(s.contains("core_clock=200e6"), "{s}");
    }

    #[test]
    fn system_config_may_leave_the_section_names_out() {
        let s = SystemConfig::new(AcceleratorConfig::EthosU85_512).build();
        assert!(s.contains("macs=512"), "{s}");
        assert!(!s.contains("system_config_name"), "{s}");
        assert!(!s.contains("memory_mode_name"), "{s}");
    }
}
