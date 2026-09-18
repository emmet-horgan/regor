//! Process-wide serialisation of the regor C library.
//!
//! # Why a single global lock
//!
//! regor is not safe to drive from more than one thread, even through separate
//! contexts. Its own mutex covers the context registry — creation, lookup and
//! destruction — but the compiler, scheduler and architecture code underneath
//! reach global state that nothing guards, and that state is shared between
//! contexts for as long as they exist. Holding a lock only around
//! `regor_compile` is therefore not enough: configuring one context while
//! another compiles corrupts it just the same.
//!
//! So every entry point into the C library takes [`lock`]. Concurrency is lost,
//! which is a real cost, but a safe wrapper cannot offer a faster contract than
//! the library underneath actually honours — and the failure mode being
//! prevented is a segfault, not a wrong answer.
//!
//! # Consequences for the public API
//!
//! * [`Compiler`](crate::Compiler) is `Send` but deliberately not `Sync`. A
//!   context may move between threads; it may not be used from two at once.
//! * Configuration is buffered in Rust and pushed into the C library inside
//!   `compile`, while the lock is held, so one thread cannot reconfigure the
//!   state another is compiling against.
//! * The logging writer is process-global, so installing it also takes this
//!   lock — see [`crate::logging`].
//!
//! # Lock ordering
//!
//! This is the innermost lock. Nothing invoked while it is held may take it
//! again, which matters most for the log writer: regor calls that writer from
//! inside compilation, with this lock already held by the calling thread. The
//! writer therefore uses its own, unrelated mutex.
//!
//! # Poisoning
//!
//! The lock is poisoned only if a call panics while holding it, which for this
//! FFI means the library is in an unknown state. The guard is recovered rather
//! than propagating a poison error, since the alternative is that every later
//! call fails for a reason unrelated to its own input.

use std::sync::{Mutex, MutexGuard, PoisonError};

static REGOR_LOCK: Mutex<()> = Mutex::new(());

/// Acquire exclusive access to the regor C library.
pub(crate) fn lock() -> MutexGuard<'static, ()> {
    REGOR_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}
