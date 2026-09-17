use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Pinned ethos-u-vela release tag for auto-download.
const PINNED_RELEASE: &str = "5.2.0";

const GITLAB_PROJECT: &str = "artificial-intelligence%2Fethos-u%2Fethos-u-vela";
const GITLAB_HOST: &str = "https://gitlab.arm.com";

fn main() {
    println!("cargo:rerun-if-env-changed=ETHOS_U_VELA_PATH");
    println!("cargo:rerun-if-env-changed=REGOR_LIB_DIR");
    println!("cargo:rerun-if-env-changed=REGOR_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=REGOR_CXX_LIB");
    println!("cargo:rerun-if-env-changed=REGOR_CACHE_DIR");
    println!("cargo:rerun-if-env-changed=REGOR_FORCE_REBUILD");

    if let Ok(lib_dir) = env::var("REGOR_LIB_DIR") {
        link_prebuilt(&PathBuf::from(lib_dir), env::var("REGOR_INCLUDE_DIR").ok());
        return;
    }

    let regor_source = if let Ok(vela_path) = env::var("ETHOS_U_VELA_PATH") {
        let p = PathBuf::from(&vela_path).join("ethosu").join("regor");
        if !p.join("CMakeLists.txt").exists() {
            panic!("ETHOS_U_VELA_PATH={vela_path} does not contain ethosu/regor/CMakeLists.txt");
        }
        p
    } else {
        download_source()
    };

    let install_dir = cmake_build(&regor_source);

    let lib_dir = find_lib_dir(&install_dir).unwrap_or_else(|| {
        panic!(
            "Could not find libregor.a or regor.lib under {}",
            install_dir.display()
        )
    });

    // Tell cargo to re-run if the library file disappears (e.g. cache eviction).
    let lib_file = if lib_dir.join("regor.lib").exists() {
        lib_dir.join("regor.lib")
    } else {
        lib_dir.join("libregor.a")
    };
    println!("cargo:rerun-if-changed={}", lib_file.display());

    let include_dir = install_dir.join("include").join("regor");
    let include_str = if include_dir.exists() {
        include_dir.display().to_string()
    } else {
        install_dir.join("include").display().to_string()
    };

    link_prebuilt(&lib_dir, Some(include_str));
}

/// Link a pre-built libregor static library.
fn link_prebuilt(lib_dir: &Path, include_dir: Option<String>) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=regor");

    // regor depends on mlw_codec; link it if present alongside.
    let has_mlw = lib_dir.join("libmlw_codec.a").exists() || lib_dir.join("mlw_codec.lib").exists();
    if has_mlw {
        println!("cargo:rustc-link-lib=static=mlw_codec");
    }

    link_cxx_stdlib();

    // pthreads — required on Unix
    let target = env::var("TARGET").unwrap_or_default();
    if !target.contains("windows") {
        println!("cargo:rustc-link-lib=pthread");
    }

    if let Some(inc) = include_dir {
        println!("cargo:include={inc}");
    }
}

/// Link the appropriate C++ standard library for the target platform.
fn link_cxx_stdlib() {
    if let Ok(lib) = env::var("REGOR_CXX_LIB") {
        println!("cargo:rustc-link-lib={lib}");
        return;
    }

    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("msvc") {
        // MSVC links the C++ runtime automatically.
    } else if target.contains("apple") || target.contains("darwin") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

/// Download the ethos-u-vela source tarball from GitLab and extract it.
fn download_source() -> PathBuf {
    let out_dir = cache_root();
    let cache_dir = out_dir.join("vela-source");
    let marker = cache_dir.join(".extracted");

    let patches = patch_files();
    for patch in &patches {
        println!("cargo:rerun-if-changed={}", patch.display());
    }
    let fingerprint = patches_fingerprint(&patches);

    // Reuse the extracted tree only if it was patched with this exact patch set.
    if fs::read_to_string(&marker).map(|s| s == fingerprint).unwrap_or(false) {
        if let Some(src) = find_regor_in(&cache_dir) {
            return src;
        }
    }

    let tarball = out_dir.join(format!("ethos-u-vela-{PINNED_RELEASE}.tar.gz"));
    let url = format!(
        "{GITLAB_HOST}/api/v4/projects/{GITLAB_PROJECT}/repository/archive.tar.gz?sha={PINNED_RELEASE}"
    );

    download_file(&url, &tarball);

    if cache_dir.exists() {
        fs::remove_dir_all(&cache_dir).expect("failed to clean cache dir");
    }
    fs::create_dir_all(&cache_dir).expect("failed to create cache dir");

    extract_tarball(&tarball, &cache_dir);

    let src = find_regor_in(&cache_dir).expect("extracted tarball does not contain ethosu/regor");
    apply_patches(&src, &patches);

    fs::write(&marker, &fingerprint).expect("failed to write marker");

    src
}

/// Give each regor context a unique id.
///
/// Upstream derives the id from the size of the context map:
///
/// ```cpp
/// *ctx = regor_context_t(s_contextMap.size() + 1);
/// ```
///
/// Ids therefore collide as soon as a context is destroyed while others are
/// alive. Create A (id 1) and B (id 2), destroy A, and the next create sees
/// size 1 and takes id 2 as well — the assignment that follows replaces B's
/// `unique_ptr`, destroying the `Compiler` that B's still-live handle points at.
/// Every later call through B is a use-after-free, which shows up as an empty
/// error at best and a segfault at worst.
///
/// A monotonic counter fixes it. Patching the vendored source is deliberate:
/// the bug is not reachable around from the Rust side, because the id is chosen
/// entirely inside `regor_create`.
///
/// Reported upstream; remove the patch when the fix lands in a pinned release.
fn apply_patches(src: &Path, patches: &[PathBuf]) {
    // Patches are generated against the repository root, so paths look like
    // `a/ethosu/regor/regor.cpp` while `src` is already `.../ethosu/regor`.
    // Strip those three leading components.
    const STRIP: &str = "-p3";

    // The extracted source usually sits under `target/`, i.e. inside the
    // consuming crate's own git repository and matched by its .gitignore.
    // In that situation `git apply` reports success while printing
    // "Skipped patch" and changing nothing. Stop repository discovery at the
    // source directory so git treats it as a plain directory tree.
    let ceiling = src.parent().unwrap_or(src);

    let git = |args: &[&str], patch: &PathBuf| {
        Command::new("git")
            .args(args)
            .arg(patch)
            .current_dir(src)
            .env("GIT_CEILING_DIRECTORIES", ceiling)
            .output()
    };

    for patch in patches {
        // Already applied (e.g. a partially reused cache)? Nothing to do.
        let applied = |p: &PathBuf| {
            git(&["apply", STRIP, "--reverse", "--check"], p)
                .map(|o| o.status.success())
                .unwrap_or(false)
        };

        if applied(patch) {
            continue;
        }

        let output = git(&["apply", STRIP, "--verbose"], patch).unwrap_or_else(|e| {
            panic!(
                "failed to run `git apply` for {}: {e}. git is required to patch \
                 the vendored regor source.",
                patch.display()
            )
        });

        if !output.status.success() {
            panic!(
                "failed to apply {}:\n{}\nThe pinned release ({PINNED_RELEASE}) may already \
                 contain this fix, or the patch needs rebasing.",
                patch.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // `git apply` can exit 0 without doing anything, so confirm the patch
        // really is present instead of trusting the exit code.
        assert!(
            applied(patch),
            "`git apply` reported success but {} is not present in {}",
            patch.display(),
            src.display()
        );

        eprintln!("regor-sys: applied {}", patch.display());
    }
}

/// Patches to apply to the vendored regor source, in sorted (apply) order.
fn patch_files() -> Vec<PathBuf> {
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("patches");
    let mut patches: Vec<PathBuf> = fs::read_dir(&dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "patch"))
                .collect()
        })
        .unwrap_or_default();
    patches.sort();
    patches
}

/// Fingerprint of the patch set, stored in the extraction marker so that
/// editing, adding or removing a patch forces a clean re-extract rather than
/// silently reusing a differently-patched source tree.
fn patches_fingerprint(patches: &[PathBuf]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for b in bytes {
            hash ^= u64::from(*b);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for patch in patches {
        feed(patch.file_name().unwrap_or_default().as_encoded_bytes());
        feed(&fs::read(patch).unwrap_or_default());
    }
    format!("v1 {hash:016x}")
}

/// Locate the ethosu/regor directory inside an extracted tarball.
/// GitLab archives have a top-level directory like `ethos-u-vela-5.0.0-<hash>/`.
fn find_regor_in(dir: &Path) -> Option<PathBuf> {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let candidate = entry.path().join("ethosu").join("regor");
            if candidate.join("CMakeLists.txt").exists() {
                return Some(candidate);
            }
        }
    }
    let direct = dir.join("ethosu").join("regor");
    if direct.join("CMakeLists.txt").exists() {
        return Some(direct);
    }
    None
}

/// Download a URL to a local file using curl or wget.
fn download_file(url: &str, dest: &Path) {
    let curl = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(dest)
        .arg(url)
        .status();

    match curl {
        Ok(status) if status.success() => return,
        _ => {}
    }

    let wget = Command::new("wget")
        .args(["-q", "-O"])
        .arg(dest)
        .arg(url)
        .status();

    match wget {
        Ok(status) if status.success() => return,
        _ => {}
    }

    panic!(
        "Failed to download {url}. Install curl or wget, or set ETHOS_U_VELA_PATH to a local checkout."
    );
}

/// Extract a .tar.gz archive into `dest`.
fn extract_tarball(tarball: &Path, dest: &Path) {
    let status = Command::new("tar")
        .args(["xzf"])
        .arg(tarball)
        .arg("-C")
        .arg(dest)
        .status()
        .expect("failed to run tar");

    if !status.success() {
        panic!("tar extraction failed");
    }
}

/// Build the regor static library via cmake.
fn cmake_build(regor_source: &Path) -> PathBuf {
    let cache_dir = cache_root();
    let build_dir = cache_dir.join("regor-build");
    let install_dir = cache_dir.join("regor-install");

    let target = env::var("TARGET").unwrap_or_default();

    let generator = if has_ninja() {
        "Ninja"
    } else if target.contains("msvc") {
        "NMake Makefiles"
    } else {
        "Unix Makefiles"
    };

    // Stamp describing the configuration this cache directory was built with.
    // If anything here changes we must start from scratch, because stale cmake
    // state (different generator, different source tree) causes hard-to-debug
    // linker failures.
    let stamp_file = cache_dir.join("regor-build.stamp");
    let stamp = format!(
        "v1\nsource={}\ntarget={}\ngenerator={}\npatches={}\n",
        regor_source.display(),
        target,
        generator,
        patches_fingerprint(&patch_files())
    );
    let stamp_matches = fs::read_to_string(&stamp_file).map(|s| s == stamp).unwrap_or(false);
    let force_rebuild = env::var_os("REGOR_FORCE_REBUILD").is_some();

    // Fast path: a previous run already produced an installed library with the
    // exact same configuration, so there is nothing to do.
    if stamp_matches && !force_rebuild {
        if let Some(lib_dir) = find_lib_dir(&install_dir) {
            eprintln!(
                "regor-sys: reusing cached build at {} (set REGOR_FORCE_REBUILD=1 to rebuild)",
                lib_dir.display()
            );
            return install_dir;
        }
    }

    if !stamp_matches || force_rebuild {
        fs::remove_dir_all(&build_dir).ok();
        fs::remove_dir_all(&install_dir).ok();
        fs::remove_file(&stamp_file).ok();
    }

    fs::create_dir_all(&build_dir).expect("failed to create build dir");
    fs::create_dir_all(&install_dir).expect("failed to create install dir");

    let mut configure = Command::new("cmake");
    configure
        .arg(regor_source)
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()))
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DREGOR_ENABLE_ASSERT=OFF")
        // Disable LTO — regor's cmake enables it by default via REGOR_ENABLE_LTO
        // when check_ipo_supported() succeeds. On MSVC this produces LTCG bitcode
        // in the .lib that requires /LTCG at final link time, which Rust
        // doesn't pass. Disabling it produces normal object code.
        .arg("-DREGOR_ENABLE_LTO=OFF")
        .args(["-G", generator])
        .current_dir(&build_dir);

    // On MSVC targets, explicitly set the compiler to cl.exe so cmake doesn't
    // pick up MinGW from PATH.
    if target.contains("msvc") {
        configure.args(["-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl"]);
    }

    let status = configure.status().expect("failed to run cmake configure");
    if !status.success() {
        panic!("cmake configure failed");
    }

    let mut build = Command::new("cmake");
    build
        .args(["--build", "."])
        .args(["--target", "regor-static"])
        .args(["--config", "Release"])
        .args(["--parallel"])
        .current_dir(&build_dir);

    let status = build.status().expect("failed to run cmake build");
    if !status.success() {
        panic!("cmake build failed");
    }

    let mut install = Command::new("cmake");
    install
        .args(["--install", "."])
        .args(["--config", "Release"])
        .current_dir(&build_dir);

    let status = install.status().expect("failed to run cmake install");
    if !status.success() {
        // Not all cmake configs have install rules for the static target.
        // Fall back to finding the library in the build tree.
        return find_lib_in_build_tree(&build_dir, regor_source);
    }

    fs::write(&stamp_file, &stamp).ok();

    install_dir
}

/// Stable, shared cache directory for the C++ build artifacts.
///
/// This deliberately avoids `OUT_DIR`: cargo derives `OUT_DIR` from a hash of
/// the build script's inputs, so it changes whenever the build script is
/// recompiled and the (very expensive) C++ build would be discarded. Keying off
/// the target directory instead keeps artifacts across such changes.
fn cache_root() -> PathBuf {
    let root = if let Ok(dir) = env::var("REGOR_CACHE_DIR") {
        PathBuf::from(dir)
    } else {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        // OUT_DIR is <target>/<profile>/build/<pkg>-<hash>/out
        let base = out_dir
            .ancestors()
            .nth(3)
            .map(Path::to_path_buf)
            .unwrap_or(out_dir);
        base.join("regor-sys-cache")
    };

    let root = root.join(env::var("TARGET").unwrap_or_default());
    fs::create_dir_all(&root).expect("failed to create cache dir");
    root
}

/// When cmake install doesn't work, search the build tree for the static lib.
fn find_lib_in_build_tree(build_dir: &Path, source_dir: &Path) -> PathBuf {
    let fallback_dir = cache_root().join("regor-fallback");
    let lib_dir = fallback_dir.join("lib");
    let include_dir = fallback_dir.join("include");

    fs::create_dir_all(&lib_dir).unwrap();
    fs::create_dir_all(&include_dir).unwrap();

    // Search for the static library
    let lib_names = ["libregor.a", "regor.lib"];
    let mut found_lib = None;

    for entry in walkdir(build_dir) {
        if let Some(fname) = entry.file_name() {
            let name = fname.to_string_lossy();
            if lib_names.iter().any(|&n| name == n) {
                found_lib = Some(entry.clone());
                break;
            }
        }
    }

    let lib_path = found_lib.unwrap_or_else(|| {
        panic!(
            "Could not find libregor.a or regor.lib in {}",
            build_dir.display()
        )
    });

    fs::copy(&lib_path, lib_dir.join(lib_path.file_name().unwrap())).unwrap();

    // Copy headers from source
    let src_include = source_dir.join("include");
    if src_include.exists() {
        for entry in walkdir(&src_include) {
            if entry.is_file() {
                let rel = entry.strip_prefix(&src_include).unwrap();
                let dest = include_dir.join(rel);
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent).unwrap();
                }
                fs::copy(&entry, &dest).unwrap();
            }
        }
    }

    fallback_dir
}

/// Search for the directory containing libregor.a or regor.lib.
/// cmake may install to lib/, lib64/, or even lib/Release/ depending on
/// the platform and generator.
fn find_lib_dir(install_dir: &Path) -> Option<PathBuf> {
    let lib_names = ["libregor.a", "regor.lib"];
    let candidates = [
        install_dir.join("lib"),
        install_dir.join("lib64"),
        install_dir.join("lib").join("Release"),
        install_dir.join("lib64").join("Release"),
    ];
    for dir in &candidates {
        for name in &lib_names {
            if dir.join(name).exists() {
                return Some(dir.clone());
            }
        }
    }
    None
}

/// Check if Ninja is available on the system.
fn has_ninja() -> bool {
    Command::new("ninja")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Simple recursive directory walker.
fn walkdir(dir: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                results.extend(walkdir(&path));
            } else {
                results.push(path);
            }
        }
    }
    results
}
