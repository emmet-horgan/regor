//! Integration tests comparing Rust regor bindings against Python reference.
//!
//! These tests compile .tflite models using both the Rust bindings and the
//! Python regor module / vela CLI, then compare the outputs.
//!
//! Requirements:
//! - Python 3.8+ with `ethos-u-vela` installed (`pip install ethos-u-vela`)
//! - `tensorflow` for model generation
//! - The regor native library (built by regor-sys)
//!
//! Set `SKIP_PYTHON_TESTS=1` to skip these tests when Python/vela is unavailable.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

mod test_common;
use test_common::*;

/// Run the Python reference script and return (output_bytes, result_json).
fn python_regor_compile(
    model_path: &Path,
    accelerator: &str,
    system_config_name: &str,
    memory_mode: &str,
    optimise: &str,
) -> (Vec<u8>, serde_json::Value) {
    let output_dir = tempfile::tempdir().unwrap();
    let script = tests_root().join("python_reference.py");

    let output = Command::new(python())
        .arg(&script)
        .arg("regor")
        .arg(model_path)
        .arg(output_dir.path())
        .arg(format!("accelerator={accelerator}"))
        .arg(format!("system_config={system_config_name}"))
        .arg(format!("memory_mode={memory_mode}"))
        .arg(format!("optimise={optimise}"))
        .output()
        .expect("failed to run python_reference.py");

    if !output.status.success() {
        panic!(
            "python_reference.py regor failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let compiled = fs::read(output_dir.path().join("regor_output.tflite")).unwrap();
    let sys_config = fs::read_to_string(output_dir.path().join("system_config.ini")).unwrap();
    let compiler_opts = fs::read_to_string(output_dir.path().join("compiler_options.ini")).unwrap();

    // The result JSON is also saved as a file — more reliable than parsing stdout
    // which may contain TF warnings.
    let result_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(output_dir.path().join("regor_result.json")).unwrap(),
    )
    .unwrap();

    let mut result = result_json.as_object().unwrap().clone();
    result.insert("_system_config".into(), sys_config.into());
    result.insert("_compiler_options".into(), compiler_opts.into());

    (compiled, serde_json::Value::Object(result))
}

/// Compile using the Rust regor bindings with config strings from Python.
fn rust_regor_compile(
    model_bytes: &[u8],
    arch: regor::Architecture,
    system_config: &str,
    compiler_options: &str,
) -> regor::Result<(regor::Output, regor::PerfReport)> {
    let mut compiler = regor::Compiler::new(arch)?;
    compiler.system_config(system_config)?;
    compiler.compiler_options(compiler_options)?;
    let out =compiler.compile(regor::InputFormat::TfLite, model_bytes)
        .unwrap();
    let report = compiler.perf_report().unwrap();
    Ok((out, report))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TestConfig {
    name: &'static str,
    arch: regor::Architecture,
    accelerator: &'static str,
    system_config_name: &'static str,
    memory_mode: &'static str,
    optimise: &'static str,
}


#[rstest::rstest]
#[case(
    TestConfig {
        name: "u55_256_perf",
        arch: regor::Architecture::EthosU55,
        accelerator: "ethos-u55-256",
        system_config_name: "Ethos_U55_High_End_Embedded",
        memory_mode: "Shared_Sram",
        optimise: "Performance",
    })
]
#[case(
    TestConfig {
        name: "u55_128_size",
        arch: regor::Architecture::EthosU55,
        accelerator: "ethos-u55-128",
        system_config_name: "Ethos_U55_High_End_Embedded",
        memory_mode: "Shared_Sram",
        optimise: "Size",
    }
)]
#[case(
    TestConfig {
        name: "u65_256_perf",
        arch: regor::Architecture::EthosU65,
        accelerator: "ethos-u65-256",
        system_config_name: "Ethos_U65_High_End",
        memory_mode: "Shared_Sram",
        optimise: "Performance",
    }
)]
#[case(
    TestConfig {
        name: "u85_256_perf",
        arch: regor::Architecture::EthosU85,
        accelerator: "ethos-u85-256",
        system_config_name: "Ethos_U85_SYS_DRAM_Mid",
        memory_mode: "Shared_Sram",
        optimise: "Performance",
    }
)]
fn regor_matches_python(
    #[files("../tests/fixtures/*.tflite")] model_path: PathBuf,
    #[case] config: TestConfig,
)
{
    if should_skip() {
        eprintln!("Skipping Python comparison tests (SKIP_PYTHON_TESTS=1)");
        return;
    }

    eprintln!("--- Config: {} Model: {:?} ---", config.name, model_path.file_name());

    let model_bytes = fs::read(&model_path).unwrap();

    let (python_output, result) = python_regor_compile(
        &model_path,
        config.accelerator,
        config.system_config_name,
        config.memory_mode,
        config.optimise,
    );

    let sys_config = result["_system_config"].as_str().unwrap();
    let compiler_opts = result["_compiler_options"].as_str().unwrap();

    let (rust_output, rust_perf) = match rust_regor_compile(&model_bytes, config.arch, sys_config, compiler_opts) {
        Ok(o) => o,
        Err(e) => panic!("Rust compilation failed for {} {:?}: {e}", config.name, model_path.file_name()),
    };

    let rust_bytes = rust_output.as_bytes();

    assert_eq!(
        rust_bytes.len(),
        python_output.len(),
        "Output size mismatch for {} {:?}: rust={} python={}",
        config.name,
        model_path.file_name(),
        rust_bytes.len(),
        python_output.len(),
    );

    assert_eq!(
        rust_bytes,
        &python_output[..],
        "Output bytes differ for {} {:?}",
        config.name,
        model_path.file_name(),
    );

    eprintln!("  PASS: {} {:?} ({} bytes)", config.name, model_path.file_name(), rust_bytes.len());

    let report = rust_perf;
    let py_perf = &result["perf"];

    assert_eq!(
        report.npu_cycles,
        py_perf["npu_cycles"].as_i64().unwrap(),
        "NPU cycles mismatch"
    );
    assert_eq!(
        report.cpu_cycles,
        py_perf["cpu_cycles"].as_i64().unwrap(),
        "CPU cycles mismatch"
    );
    assert_eq!(
        report.total_cycles,
        py_perf["total_cycles"].as_i64().unwrap(),
        "Total cycles mismatch"
    );
    assert_eq!(
        report.npu_ops,
        py_perf["npu_ops"].as_i64().unwrap(),
        "NPU ops mismatch"
    );
    assert_eq!(
        report.cpu_ops,
        py_perf["cpu_ops"].as_i64().unwrap(),
        "CPU ops mismatch"
    );
    assert_eq!(
        report.original_weights,
        py_perf["original_weights"].as_i64().unwrap(),
        "Original weights mismatch"
    );
    assert_eq!(
        report.encoded_weights,
        py_perf["encoded_weights"].as_i64().unwrap(),
        "Encoded weights mismatch"
    );
}
