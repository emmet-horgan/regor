use std::marker::PhantomData;

use regor_sys as ffi;

/// Compiled output returned by [`crate::Compiler::compile`].
///
/// Holds the raw bytes produced by the compilation. Created via the
/// writer-callback path so all data is owned by Rust.
pub struct Output {
    data: Vec<u8>,
}

impl std::fmt::Debug for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Command streams run to hundreds of kilobytes, so show the size rather
        // than the bytes.
        f.debug_struct("Output")
            .field("len", &self.data.len())
            .finish_non_exhaustive()
    }
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

/// Handle to an `IRegorBlob*` owned by a [`Compiler`](crate::Compiler).
///
/// The blob belongs to the context that produced it and is released when that
/// context is destroyed. The borrow tying it to the compiler is what makes this
/// sound: the handle would dangle the moment the context went away, and there
/// is no `Release` entry point in the C API to hand ownership over instead.
///
/// Not `Send`. The pointer is only meaningful while the borrowed compiler is
/// alive, and that compiler is itself `!Sync`.
pub struct Blob<'a> {
    ctx: ffi::regor_context_t,
    ptr: *mut ffi::IRegorBlob,
    owner: PhantomData<&'a mut ()>,
}

impl std::fmt::Debug for Blob<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Blob")
            .field("ctx", &self.ctx)
            .field("ptr", &self.ptr)
            .finish()
    }
}

impl Blob<'_> {
    /// # Safety
    ///
    /// `ptr` must be a blob produced by `ctx`, and the returned lifetime must
    /// not outlive that context.
    pub(crate) unsafe fn from_raw(ctx: ffi::regor_context_t, ptr: *mut ffi::IRegorBlob) -> Self {
        Self {
            ctx,
            ptr,
            owner: PhantomData,
        }
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
