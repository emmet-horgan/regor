use regor_sys as ffi;

/// Compiled output returned by [`crate::Compiler::compile`].
///
/// Holds the raw bytes produced by the compilation. Created via the
/// writer-callback path so all data is owned by Rust.
pub struct Output {
    data: Vec<u8>,
}

impl Output {
    pub(crate) fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl AsRef<[u8]> for Output {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

/// RAII wrapper around `IRegorBlob*`.
///
/// The blob is reference-counted on the C++ side. Dropping this handle
/// calls `Release()` through the vtable.
pub struct Blob {
    ctx: ffi::regor_context_t,
    ptr: *mut ffi::IRegorBlob,
}

// IRegorBlob is internally synchronized by the C++ runtime.
unsafe impl Send for Blob {}

impl Blob {
    pub(crate) unsafe fn from_raw(
        ctx: ffi::regor_context_t,
        ptr: *mut ffi::IRegorBlob,
    ) -> Self {
        Self { ctx, ptr }
    }

    /// The raw blob pointer, for advanced interop.
    pub fn as_ptr(&self) -> *mut ffi::IRegorBlob {
        self.ptr
    }

    /// The compiler context this blob belongs to.
    pub fn context(&self) -> ffi::regor_context_t {
        self.ctx
    }
}
