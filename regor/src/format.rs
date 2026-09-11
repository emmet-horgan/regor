use regor_sys as ffi;

/// Input model format accepted by the compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputFormat {
    /// Regor's internal graph-builder API format.
    GraphApi,
    /// TensorFlow Lite / LiteRT flatbuffer.
    TfLite,
    /// TOSA flatbuffer.
    Tosa,
}

impl InputFormat {
    pub(crate) fn to_ffi(self) -> ffi::regor_format_t {
        match self {
            InputFormat::GraphApi => ffi::regor_format_t::REGOR_INPUTFORMAT_GRAPHAPI,
            InputFormat::TfLite => ffi::regor_format_t::REGOR_INPUTFORMAT_TFLITE,
            InputFormat::Tosa => ffi::regor_format_t::REGOR_INPUTFORMAT_TOSA,
        }
    }
}
