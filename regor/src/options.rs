use std::fmt::Write;

/// Optimization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Optimise {
    /// Prioritise inference speed; uses `arena_cache_size` as a memory target.
    Performance,
    /// Prioritise lower memory usage.
    Size,
}

/// Tensor allocation algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorAllocator {
    HillClimb,
    LinearAlloc,
    #[deprecated = "will be removed in a future vela release"]
    Greedy,
}

/// Custom operator payload format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CopFormat {
    Cop1,
    /// Required for `separate_io_regions`.
    Cop2,
}

/// Output format for the compiled model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    /// Standard `.tflite` with custom operator nodes.
    TfLite,
    /// Raw `.npz` with accessible arrays. Not compatible with CPU fallback.
    Raw,
}

/// Accelerator variant with MAC count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    fn as_str(self) -> &'static str {
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
}

/// Builder for regor compiler options.
///
/// Typed setters validate at build time; the result is serialised to the
/// key=value format that `regor_set_compiler_options` expects.
///
/// ```no_run
/// # use regor::options::*;
/// let opts = CompilerOptions::new()
///     .accelerator(AcceleratorConfig::EthosU55_256)
///     .optimise(Optimise::Performance)
///     .arena_cache_size(2 * 1024 * 1024)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CompilerOptions {
    entries: Vec<(String, String)>,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl CompilerOptions {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn set(mut self, key: &str, value: impl std::fmt::Display) -> Self {
        self.entries.push((key.to_owned(), value.to_string()));
        self
    }

    // ------------------------------------------------------------------
    // Hardware
    // ------------------------------------------------------------------

    /// Accelerator configuration (architecture + MAC count).
    pub fn accelerator(self, config: AcceleratorConfig) -> Self {
        self.set("accelerator_config", config.as_str())
    }

    // ------------------------------------------------------------------
    // Optimisation
    // ------------------------------------------------------------------

    pub fn optimise(self, strategy: Optimise) -> Self {
        let v = match strategy {
            Optimise::Performance => "Performance",
            Optimise::Size => "Size",
        };
        self.set("optimise", v)
    }

    pub fn tensor_allocator(self, alloc: TensorAllocator) -> Self {
        #[allow(deprecated)]
        let v = match alloc {
            TensorAllocator::HillClimb => "HillClimb",
            TensorAllocator::LinearAlloc => "LinearAlloc",
            TensorAllocator::Greedy => "Greedy",
        };
        self.set("tensor_allocator", v)
    }

    /// Arena/cache memory budget in bytes. Must be >= 0.
    pub fn arena_cache_size(self, bytes: u64) -> Self {
        self.set("arena_cache_size", bytes)
    }

    // ------------------------------------------------------------------
    // Memory & alignment
    // ------------------------------------------------------------------

    /// CPU tensor alignment in bytes. Must be a power of two and >= 16.
    pub fn cpu_tensor_alignment(self, alignment: u32) -> Result<Self, OptionsError> {
        if alignment < 16 || !alignment.is_power_of_two() {
            return Err(OptionsError::InvalidAlignment(alignment));
        }
        Ok(self.set("cpu_tensor_alignment", alignment))
    }

    pub fn cop_format(self, fmt: CopFormat) -> Self {
        let v = match fmt {
            CopFormat::Cop1 => "COP1",
            CopFormat::Cop2 => "COP2",
        };
        self.set("cop_format", v)
    }

    /// Allocate inputs/outputs in separate regions instead of scratch.
    /// Requires [`CopFormat::Cop2`].
    pub fn separate_io_regions(self, enable: bool) -> Self {
        self.set("separate_io_regions", enable)
    }

    // ------------------------------------------------------------------
    // Output
    // ------------------------------------------------------------------

    pub fn output_format(self, fmt: OutputFormat) -> Self {
        let v = match fmt {
            OutputFormat::TfLite => "tflite",
            OutputFormat::Raw => "raw",
        };
        self.set("output_format", v)
    }

    // ------------------------------------------------------------------
    // Quantisation
    // ------------------------------------------------------------------

    /// Force all signed-integer weight zero-points to 0.
    pub fn force_symmetric_int_weights(self, enable: bool) -> Self {
        self.set("force_symmetric_int_weights", enable)
    }

    // ------------------------------------------------------------------
    // Operator control
    // ------------------------------------------------------------------

    /// TFLite operators to force onto the CPU (comma-separated).
    pub fn ignore_ops(self, ops: &[&str]) -> Self {
        self.set("ignore_ops", ops.join(","))
    }

    // ------------------------------------------------------------------
    // Regor feature toggles
    // ------------------------------------------------------------------

    pub fn disable_chaining(self, disable: bool) -> Self {
        self.set("disable_chaining", disable)
    }

    pub fn disable_fast_weight_decoder(self, disable: bool) -> Self {
        self.set("disable_fwd", disable)
    }

    pub fn disable_cascading(self, disable: bool) -> Self {
        self.set("disable_cascading", disable)
    }

    pub fn disable_buffering(self, disable: bool) -> Self {
        self.set("disable_buffering", disable)
    }

    // ------------------------------------------------------------------
    // Debug / verbose
    // ------------------------------------------------------------------

    pub fn enable_debug_db(self, enable: bool) -> Self {
        self.set("enable_debug_db", enable)
    }

    pub fn show_cpu_operations(self, enable: bool) -> Self {
        self.set("show_cpu_operations", enable)
    }

    pub fn timing(self, enable: bool) -> Self {
        self.set("timing", enable)
    }

    /// Maximum block dependency (0..=3).
    #[deprecated = "will be removed in a future vela release"]
    pub fn max_block_dependency(self, dep: u8) -> Result<Self, OptionsError> {
        if dep > 3 {
            return Err(OptionsError::OutOfRange {
                option: "max_block_dependency",
                value: dep as u64,
                min: 0,
                max: 3,
            });
        }
        Ok(self.set("max_block_dependency", dep))
    }

    // ------------------------------------------------------------------
    // Escape hatch
    // ------------------------------------------------------------------

    /// Set an arbitrary key=value pair not covered by the typed API.
    pub fn raw(self, key: &str, value: &str) -> Self {
        self.set(key, value)
    }

    // ------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------

    /// Serialise to the newline-separated `key=value` string expected by
    /// `regor_set_compiler_options`.
    pub fn build(&self) -> Result<String, OptionsError> {
        // Validate cross-field constraints.
        let has_separate_io = self
            .entries
            .iter()
            .any(|(k, v)| k == "separate_io_regions" && v == "true");
        if has_separate_io {
            let has_cop2 = self
                .entries
                .iter()
                .any(|(k, v)| k == "cop_format" && v == "COP2");
            if !has_cop2 {
                return Err(OptionsError::SeparateIoRequiresCop2);
            }
        }

        let mut out = String::new();
        for (i, (k, v)) in self.entries.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            write!(out, "{k}={v}").unwrap();
        }
        Ok(out)
    }
}

/// Error from option validation.
#[derive(Debug)]
pub enum OptionsError {
    /// `cpu_tensor_alignment` must be a power of two >= 16.
    InvalidAlignment(u32),
    /// A numeric option was out of its valid range.
    OutOfRange {
        option: &'static str,
        value: u64,
        min: u64,
        max: u64,
    },
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
            OptionsError::OutOfRange {
                option,
                value,
                min,
                max,
            } => write!(f, "{option} must be in {min}..={max}, got {value}"),
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

    #[test]
    fn basic_build() {
        let s = CompilerOptions::new()
            .optimise(Optimise::Performance)
            .arena_cache_size(2_097_152)
            .build()
            .unwrap();
        assert_eq!(s, "optimise=Performance\narena_cache_size=2097152");
    }

    #[test]
    fn alignment_validation() {
        assert!(CompilerOptions::new().cpu_tensor_alignment(16).is_ok());
        assert!(CompilerOptions::new().cpu_tensor_alignment(64).is_ok());
        assert!(CompilerOptions::new().cpu_tensor_alignment(15).is_err());
        assert!(CompilerOptions::new().cpu_tensor_alignment(8).is_err());
    }

    #[test]
    fn separate_io_requires_cop2() {
        let err = CompilerOptions::new().separate_io_regions(true).build();
        assert!(err.is_err());

        let ok = CompilerOptions::new()
            .cop_format(CopFormat::Cop2)
            .separate_io_regions(true)
            .build();
        assert!(ok.is_ok());
    }

    #[test]
    fn raw_escape_hatch() {
        let s = CompilerOptions::new()
            .optimise(Optimise::Size)
            .raw("some_future_option", "42")
            .build()
            .unwrap();
        assert!(s.contains("some_future_option=42"));
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
}
