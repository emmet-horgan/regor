# Patches to the regor source

Fixes for upstream regor bugs, applied to the `ethos-u-vela` tree.

## Who applies these

Only CI. Both workflows apply every patch in this directory to their
`ethos-u-vela` checkout before building:

- `build-native.yml` — so the published pre-built artifacts carry the fixes.
- `ci.yml` — so tests run against the same source those artifacts are built
  from.

`regor-sys/build.rs` does **not** patch. If you build from source via
`ETHOS_U_VELA_PATH`, you get your checkout exactly as it is. To pick the fixes
up, apply them yourself:

```sh
cd ethos-u-vela
git apply /path/to/regor-sys/patches/*.patch
```

They are written against the repository root, so they apply with the default
`-p1` from an `ethos-u-vela` checkout.

## 0001-monotonic-context-ids.patch

Upstream derives each context id from the size of the context map:

```cpp
*ctx = regor_context_t(s_contextMap.size() + 1);
```

Ids therefore collide as soon as a context is destroyed while others are alive.
Create A (id 1) and B (id 2), destroy A, and the next create sees size 1 and
takes id 2 as well — the assignment that follows replaces B's `unique_ptr`,
destroying the `Compiler` that B's still-live handle points at. Every later call
through B is a use-after-free, which shows up as an empty error at best and a
segfault at worst.

A monotonic `std::atomic<int>` counter fixes it. The bug cannot be worked around
from the Rust side, because the id is chosen entirely inside `regor_create`.

Reported upstream; drop the patch once the fix lands in the pinned release.
