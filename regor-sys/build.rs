//! Resolves the regor native library, in one of three ways:
//!
//! 1. `REGOR_LIB_DIR` — link a library the caller already has.
//! 2. `ETHOS_U_VELA_PATH` — build from a local ethos-u-vela checkout.
//! 3. Otherwise — download the pre-built artifact for this target.
//!
//! There is deliberately no fetch-the-source path. Downloading a source
//! tarball and driving cmake is the slow, fragile half of the problem, and it
//! only ever existed to produce what option 3 now hands over ready-made. A
//! caller who wants to build regor themselves already has a checkout, so they
//! can say where it is.
//!
//! Downloading needs `curl` or `wget` on PATH; unpacking is done in-process.
//! Only option 2 needs cmake and a C++ compiler.
//!
//! # Patches
//!
//! `patches/` holds fixes for upstream regor bugs. They are applied by the
//! workflow that publishes the pre-built artifacts, and nowhere else. This
//! script never patches: it builds whatever source it is pointed at, out of
//! tree, leaving the caller's checkout exactly as it found it.
//!
//! So options 1 and 2 link precisely the source the caller supplied — patched
//! only if they patched it. That is the intended contract: if you build regor
//! from source, you get your source. `patches/` is there to apply yourself if
//! you want the fixes.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};

/// ethos-u-vela release this crate's bindings and patches are written against.
///
/// Only used for diagnostics here — nothing is downloaded from Arm. The
/// publishing workflow reads this constant to decide what to check out.
const PINNED_RELEASE: &str = "5.2.0";

/// Repository publishing the pre-built native libraries, and the release tag
/// whose assets this version of the crate expects.
const ARTIFACT_REPO: &str = "emmet-horgan/regor";
const ARTIFACT_TAG: &str = "native-v5.2.0";

/// Pinned digests of those assets.
///
/// This is the whole point of the check. GitHub also reports a digest for each
/// release asset, but it arrives from the same server over the same connection
/// as the artifact, so it proves only that the download was not corrupted —
/// something TLS already covers. A digest recorded here is different in kind:
/// it is reviewed in a commit, and GitHub cannot change it. Release assets are
/// mutable (delete the file, upload another with the same name), so without
/// this the tag could quietly start meaning a different binary.
///
/// See the file itself for how to regenerate it.
const ARTIFACT_DIGESTS: &str = include_str!("artifacts.sha256");

fn main() {
    for var in [
        "ETHOS_U_VELA_PATH",
        "REGOR_LIB_DIR",
        "REGOR_INCLUDE_DIR",
        "REGOR_CXX_LIB",
        "REGOR_CACHE_DIR",
        "REGOR_FORCE_REBUILD",
        "REGOR_BUILD_TYPE",
        "REGOR_ARTIFACT_BASE_URL",
        "REGOR_ARTIFACT_TAG",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=artifacts.sha256");

    // docs.rs builds have no network access. The native library isn't
    // needed to generate the Rust API documentation.
    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    if let Ok(lib_dir) = env::var("REGOR_LIB_DIR") {
        link(&PathBuf::from(lib_dir), env::var("REGOR_INCLUDE_DIR").ok());
        return;
    }

    let install_dir = match env::var("ETHOS_U_VELA_PATH") {
        Ok(vela_path) => cmake_build(&regor_source(&vela_path)),
        Err(_) => download_prebuilt(),
    };

    let lib_dir = find_lib_dir(&install_dir)
        .unwrap_or_else(|| panic!("no libregor.a or regor.lib under {}", install_dir.display()));

    // Re-run if the library disappears (e.g. cache eviction).
    for name in ["regor.lib", "libregor.a"] {
        let path = lib_dir.join(name);
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    link(&lib_dir, find_include_dir(&install_dir));
}

/// The regor subdirectory of an ethos-u-vela checkout.
fn regor_source(vela_path: &str) -> PathBuf {
    let src = PathBuf::from(vela_path).join("ethosu").join("regor");
    assert!(
        src.join("CMakeLists.txt").exists(),
        "ETHOS_U_VELA_PATH={vela_path} does not contain ethosu/regor/CMakeLists.txt"
    );
    src
}

/// Headers as installed by cmake, or as laid out in a published artifact.
fn find_include_dir(install_dir: &Path) -> Option<String> {
    let nested = install_dir.join("include").join("regor");
    if nested.exists() {
        return Some(nested.display().to_string());
    }
    let flat = install_dir.join("include");
    flat.exists().then(|| flat.display().to_string())
}

/// Emit the link directives for a regor static library.
fn link(lib_dir: &Path, include_dir: Option<String>) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=regor");

    // regor depends on mlw_codec; link it if present alongside.
    let has_mlw = lib_dir.join("libmlw_codec.a").exists() || lib_dir.join("mlw_codec.lib").exists();
    if has_mlw {
        println!("cargo:rustc-link-lib=static=mlw_codec");
    }

    link_cxx_stdlib();

    // pthreads — required on Unix
    if !target_triple().contains("windows") {
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

    let target = target_triple();
    if target.contains("msvc") {
        // MSVC links the C++ runtime automatically.
    } else if target.contains("apple") || target.contains("darwin") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn target_triple() -> String {
    env::var("TARGET").unwrap_or_default()
}

/// CMake build type, and the lowercase form used in artifact names.
///
/// Only `Release` and `Debug` are accepted: the value is interpolated into a
/// download URL and a cmake command line, so it is validated against a fixed
/// set rather than passed through.
fn build_type() -> (&'static str, &'static str) {
    let requested = env::var("REGOR_BUILD_TYPE").unwrap_or_default();
    match requested.to_ascii_lowercase().as_str() {
        "" | "release" => ("Release", "release"),
        "debug" => ("Debug", "debug"),
        other => panic!("REGOR_BUILD_TYPE must be 'Release' or 'Debug', got '{other}'"),
    }
}

// ---------------------------------------------------------------------------
// Pre-built artifacts
// ---------------------------------------------------------------------------

/// Download, verify and unpack the pre-built library for this target.
///
/// Failure is fatal rather than falling back to a source build: without a
/// checkout there is no source to fall back to, and silently doing something
/// other than what was asked for is worse than a clear error.
fn download_prebuilt() -> PathBuf {
    let (_, config) = build_type();
    let tag = env::var("REGOR_ARTIFACT_TAG").unwrap_or_else(|_| ARTIFACT_TAG.to_string());
    let name = format!("regor-native-{}-{config}.tar.gz", target_triple());

    let Some(expected) = pinned_digest(&name) else {
        panic!(
            "no pre-built regor library is published for {} ({config}).\n\
             Build it yourself instead: set ETHOS_U_VELA_PATH to an ethos-u-vela \
             {PINNED_RELEASE} checkout, or REGOR_LIB_DIR to a directory containing \
             an already-built library.",
            target_triple()
        );
    };

    // The directory is named after the digest of what is in it, so its mere
    // existence proves it holds the verified contents this build wants. That
    // removes the need for any marker file, and means changing the pinned
    // digest lands in a new directory rather than invalidating this one.
    let dir = cache_root().join(format!("prebuilt-{}", &expected[..16]));
    if find_lib_dir(&dir).is_some() {
        return dir;
    }

    let base = env::var("REGOR_ARTIFACT_BASE_URL")
        .unwrap_or_else(|_| format!("https://github.com/{ARTIFACT_REPO}/releases/download/{tag}"));
    let url = format!("{base}/{name}");

    // Downloaded to memory rather than to a file: the bytes have to be hashed
    // before anything is done with them, and keeping them off disk means there
    // is no half-written or unverified tarball to clean up on failure.
    let tarball = download(&url).unwrap_or_else(|e| {
        panic!(
            "failed to download {url}\n  {e}\n\
             Needs curl or wget on PATH and network access. Set ETHOS_U_VELA_PATH \
             to an ethos-u-vela {PINNED_RELEASE} checkout to build from source instead."
        )
    });

    let actual = format!("{:x}", Sha256::digest(&tarball));
    assert!(
        actual == expected,
        "checksum mismatch for {name}:\n  expected {expected}\n  actual   {actual}\n\
         Refusing to link an unverified library."
    );

    // Unpacked into a temporary directory and renamed on success, so an
    // interrupted extraction cannot leave a partial tree sitting at a path
    // whose name asserts it holds verified contents.
    let staging = dir.with_extension("partial");
    fs::remove_dir_all(&staging).ok();
    fs::create_dir_all(&staging).expect("failed to create artifact dir");
    extract_tarball(&tarball, &staging);

    assert!(
        find_lib_dir(&staging).is_some(),
        "{name} unpacked without a regor static library in it"
    );

    fs::remove_dir_all(&dir).ok();
    fs::rename(&staging, &dir).expect("failed to publish artifact dir");
    dir
}

/// The pinned digest for an artifact, if one is published for it.
fn pinned_digest(name: &str) -> Option<String> {
    ARTIFACT_DIGESTS
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .find_map(|line| {
            let (digest, file) = line.split_once(char::is_whitespace)?;
            // `sha256sum` writes `<digest>  <name>`, with the extra space
            // landing in the filename field and a '*' prefix in binary mode.
            let file = file.trim().trim_start_matches('*');
            (file == name && digest.len() == 64 && digest.chars().all(|c| c.is_ascii_hexdigit()))
                .then(|| digest.to_ascii_lowercase())
        })
}

/// Fetch a URL into memory using curl or wget, whichever is available.
///
/// Both are told to write to stdout so the body never touches disk — it has to
/// be hashed before anything is done with it, and keeping it in memory means
/// there is no unverified file to clean up on failure.
///
/// curl's `-f` matters: without it a 404 error page is written to stdout with a
/// success status, and the caller would go on to treat it as a tarball.
fn download(url: &str) -> Result<Vec<u8>, String> {
    let attempts = [
        ("curl", vec!["-fsSL", "--retry", "3", "-o", "-", url]),
        ("wget", vec!["-q", "-O", "-", url]),
    ];

    let mut errors = Vec::new();

    for (program, args) in attempts {
        match Command::new(program).args(&args).output() {
            Ok(out) if out.status.success() && !out.stdout.is_empty() => return Ok(out.stdout),
            Ok(out) if out.status.success() => {
                errors.push(format!("{program}: succeeded but returned no data"));
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                // curl already prefixes its messages with "curl:".
                let detail = stderr
                    .trim()
                    .trim_start_matches(&format!("{program}: ")[..]);
                let detail = if detail.is_empty() {
                    format!("exit {}", out.status)
                } else {
                    detail.to_string()
                };
                errors.push(format!("{program}: {detail}"));
            }
            Err(e) => errors.push(format!("{program}: {e}")),
        }
    }

    Err(errors.join("; "))
}

/// Extract a .tar.gz archive into `dest`.
///
/// `Archive::unpack` refuses entries that would escape `dest` via absolute
/// paths or `..`, so a malicious archive cannot write outside the cache.
fn extract_tarball(tarball: &[u8], dest: &Path) {
    tar::Archive::new(GzDecoder::new(tarball))
        .unpack(dest)
        .unwrap_or_else(|e| panic!("failed to extract archive into {}: {e}", dest.display()));
}

// ---------------------------------------------------------------------------
// Source builds
// ---------------------------------------------------------------------------

/// Build the regor static library via cmake.
///
/// The caller's checkout is used directly, out of tree: cmake writes only into
/// the build directory, so nothing here modifies their working copy. The source
/// is built exactly as supplied — unpatched unless the caller patched it. See
/// the module docs.
fn cmake_build(src: &Path) -> PathBuf {
    let target = env::var("TARGET").unwrap_or_default();
    let (cmake_config, config) = build_type();

    let generator = if has_ninja() {
        "Ninja"
    } else if target.contains("msvc") {
        "NMake Makefiles"
    } else {
        "Unix Makefiles"
    };

    // Cargo walks this recursively, so it re-runs the build script when the
    // caller edits their checkout — and, just as importantly, does not
    // re-run it when they have not.
    println!("cargo:rerun-if-changed={}", src.display());

    // Everything that would invalidate cmake's own state — a different
    // generator, configuration or source tree — is folded into the directory
    // name. A change therefore lands in a *different* directory rather than
    // requiring this script to detect staleness and wipe. Within a directory,
    // cmake is left to work out what needs rebuilding.
    let root = cache_root().join(build_key(&target, generator, config, src));
    let build_dir = root.join("build");
    let install_dir = root.join("install");

    if env::var_os("REGOR_FORCE_REBUILD").is_some() {
        fs::remove_dir_all(&root).ok();
    }

    fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let mut configure = Command::new("cmake");
    configure
        .arg(src)
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()))
        .arg(format!("-DCMAKE_BUILD_TYPE={cmake_config}"))
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

    run(configure, "cmake configure");

    let mut build = Command::new("cmake");
    build
        .args(["--build", "."])
        .args(["--target", "regor-static"])
        .args(["--config", cmake_config])
        .args(["--parallel"])
        .current_dir(&build_dir);
    run(build, "cmake build");

    let mut install = Command::new("cmake");
    install
        .args(["--install", "."])
        .args(["--config", cmake_config])
        .current_dir(&build_dir);

    let installed = install
        .status()
        .expect("failed to run cmake install")
        .success();

    if installed {
        install_dir
    } else {
        // Not all cmake configs have install rules for the static target.
        // Fall back to finding the library in the build tree.
        find_lib_in_build_tree(&build_dir, src)
    }
}

fn run(mut command: Command, what: &str) {
    let status = command
        .status()
        .unwrap_or_else(|e| panic!("failed to run {what}: {e}"));
    assert!(status.success(), "{what} failed");
}

/// Directory name identifying one native build configuration.
fn build_key(target: &str, generator: &str, config: &str, src: &Path) -> String {
    let mut hasher = Sha256::new();
    for part in [
        generator.as_bytes(),
        config.as_bytes(),
        src.as_os_str().as_encoded_bytes(),
    ] {
        hasher.update(part);
        hasher.update([0]);
    }
    // Truncated to keep the path short. This only has to distinguish
    // configurations on one machine, not resist collisions.
    let digest = format!("{:x}", hasher.finalize());
    format!("{target}-{config}-{}", &digest[..16])
}

/// Stable, shared cache directory for the C++ build artifacts.
///
/// Root of the native build cache, inside cargo's target directory.
///
/// Living under `target/` means `cargo clean` disposes of it, it is covered by
/// existing `.gitignore` rules, and CI caches that already save `target/` pick
/// it up. It deliberately avoids `OUT_DIR` itself: cargo derives that from a
/// hash of the build script's inputs, so it changes whenever this script is
/// edited, and the very expensive C++ build would be discarded with it.
///
/// Nothing here is keyed by target or configuration — the per-build directory
/// names carry that, so distinct configurations coexist instead of evicting
/// each other.
fn cache_root() -> PathBuf {
    let root = if let Ok(dir) = env::var("REGOR_CACHE_DIR") {
        PathBuf::from(dir)
    } else {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        // OUT_DIR is <target>/<profile>/build/<pkg>-<hash>/out
        out_dir
            .ancestors()
            .nth(3)
            .map(Path::to_path_buf)
            .unwrap_or(out_dir)
            .join("regor-sys-cache")
    };

    fs::create_dir_all(&root).expect("failed to create cache dir");
    root
}

/// When cmake install doesn't work, search the build tree for the static lib.
fn find_lib_in_build_tree(build_dir: &Path, source_dir: &Path) -> PathBuf {
    let fallback_dir = build_dir
        .parent()
        .unwrap_or(build_dir)
        .join("install-fallback");
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
