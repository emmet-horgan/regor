use std::env;
use std::path::PathBuf;

fn main() {
    let regor_dir = env::var("REGOR_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Fall back to a co-located build directory.
            let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
            manifest.join("regor")
        });

    let lib_dir = env::var("REGOR_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| regor_dir.join("lib"));

    let include_dir = env::var("REGOR_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| regor_dir.join("include"));

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=regor-static");

    // C++ standard library — required because regor is compiled as C++.
    let cxx_lib = env::var("REGOR_CXX_LIB").unwrap_or_else(|_| "stdc++".to_string());
    println!("cargo:rustc-link-lib={cxx_lib}");

    println!("cargo:include={}", include_dir.display());

    println!("cargo:rerun-if-env-changed=REGOR_DIR");
    println!("cargo:rerun-if-env-changed=REGOR_LIB_DIR");
    println!("cargo:rerun-if-env-changed=REGOR_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=REGOR_CXX_LIB");
}
