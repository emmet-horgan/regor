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
        // Dump directory contents for debugging
        eprintln!("regor-sys: install_dir contents:");
        for entry in walkdir(&install_dir) {
            eprintln!("  {}", entry.display());
        }
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
    eprintln!(
        "regor-sys: linking {} ({})",
        lib_file.display(),
        if lib_file.exists() {
            format!(
                "{} bytes",
                fs::metadata(&lib_file).map(|m| m.len()).unwrap_or(0)
            )
        } else {
            "MISSING".to_string()
        }
    );

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
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cache_dir = out_dir.join("vela-source");
    let marker = cache_dir.join(".extracted");

    if marker.exists() {
        if let Some(src) = find_regor_in(&cache_dir) {
            return src;
        }
    }

    let tarball = out_dir.join(format!("ethos-u-vela-{PINNED_RELEASE}.tar.gz"));
    let url = format!(
        "{GITLAB_HOST}/api/v4/projects/{GITLAB_PROJECT}/repository/archive.tar.gz?sha={PINNED_RELEASE}"
    );

    eprintln!("regor-sys: downloading ethos-u-vela {PINNED_RELEASE} from {url}");

    download_file(&url, &tarball);

    if cache_dir.exists() {
        fs::remove_dir_all(&cache_dir).expect("failed to clean cache dir");
    }
    fs::create_dir_all(&cache_dir).expect("failed to create cache dir");

    extract_tarball(&tarball, &cache_dir);
    fs::write(&marker, "").expect("failed to write marker");

    find_regor_in(&cache_dir).expect("extracted tarball does not contain ethosu/regor")
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
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_dir = out_dir.join("regor-build");
    let install_dir = out_dir.join("regor-install");

    // Always start fresh — stale cmake state or install artifacts from a
    // prior failed build (e.g. different generator, partial install) cause
    // hard-to-debug linker failures.
    if build_dir.exists() {
        fs::remove_dir_all(&build_dir).ok();
    }
    if install_dir.exists() {
        fs::remove_dir_all(&install_dir).ok();
    }
    fs::create_dir_all(&build_dir).expect("failed to create build dir");
    fs::create_dir_all(&install_dir).expect("failed to create install dir");

    let target = env::var("TARGET").unwrap_or_default();

    let mut configure = Command::new("cmake");
    configure
        .arg(regor_source)
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()))
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DREGOR_ENABLE_ASSERT=OFF")
        .arg("-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF")
        .current_dir(&build_dir);

    // On MSVC, cmake's Release build type adds /GL (Whole Program Optimization)
    // which produces LTCG bitcode in the .lib instead of machine code. Rust's
    // linker invocation doesn't pass /LTCG, so all symbols appear unresolved.
    // Override the Release flags to exclude /GL.
    if target.contains("msvc") {
        configure.arg("-DCMAKE_C_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG");
        configure.arg("-DCMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG");
    }

    // Use a single-config generator. Prefer Ninja (fast, handles long paths).
    // Fall back to Unix Makefiles on non-Windows. On MSVC targets, explicitly
    // set the compiler to cl.exe so cmake doesn't pick up MinGW from PATH.
    if has_ninja() {
        configure.args(["-G", "Ninja"]);
    } else if target.contains("msvc") {
        configure.args(["-G", "NMake Makefiles"]);
    } else {
        configure.args(["-G", "Unix Makefiles"]);
    }
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
        eprintln!("regor-sys: cmake install failed, searching build tree for libregor");
        return find_lib_in_build_tree(&build_dir, regor_source);
    }

    install_dir
}

/// When cmake install doesn't work, search the build tree for the static lib.
fn find_lib_in_build_tree(build_dir: &Path, source_dir: &Path) -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let fallback_dir = out_dir.join("regor-fallback");
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
