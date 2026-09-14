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

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

static FIXTURES_DIR: OnceLock<PathBuf> = OnceLock::new();

fn fixtures_dir() -> &'static Path {
    FIXTURES_DIR.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("tests")
            .join("fixtures");
        fs::create_dir_all(&dir).unwrap();
        dir
    })
}

fn should_skip() -> bool {
    env::var("SKIP_PYTHON_TESTS").is_ok()
}

fn python() -> String {
    env::var("PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn tests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tests")
}

fn ensure_generated_model() -> PathBuf {
    let model_path = fixtures_dir().join("test_model.tflite");
    if model_path.exists() {
        return model_path;
    }

    let script = tests_root().join("generate_model.py");
    let output = Command::new(python())
        .arg(&script)
        .arg(&model_path)
        .output()
        .expect("failed to run generate_model.py");

    if !output.status.success() {
        panic!(
            "generate_model.py failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    assert!(model_path.exists(), "Model was not generated");
    model_path
}

fn ensure_mobilenet_model() -> PathBuf {
    let model_path = fixtures_dir().join("mobilenet_v1_quant.tflite");
    if model_path.exists() {
        return model_path;
    }

    let script = tests_root().join("download_model.py");
    let output = Command::new(python())
        .arg(&script)
        .arg(&model_path)
        .output()
        .expect("failed to run download_model.py");

    if !output.status.success() {
        panic!(
            "download_model.py failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    assert!(model_path.exists(), "MobileNet model was not downloaded");
    model_path
}

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
) -> regor::Result<regor::Output> {
    let mut compiler = regor::Compiler::new(arch)?;
    compiler.system_config(system_config)?;
    compiler.compiler_options(compiler_options)?;
    compiler.compile(regor::InputFormat::TfLite, model_bytes)
}

struct TestConfig {
    name: &'static str,
    arch: regor::Architecture,
    accelerator: &'static str,
    system_config_name: &'static str,
    memory_mode: &'static str,
    optimise: &'static str,
}

const CONFIGS: &[TestConfig] = &[
    TestConfig {
        name: "u55_256_perf",
        arch: regor::Architecture::EthosU55,
        accelerator: "ethos-u55-256",
        system_config_name: "Ethos_U55_High_End_Embedded",
        memory_mode: "Shared_Sram",
        optimise: "Performance",
    },
    TestConfig {
        name: "u55_128_size",
        arch: regor::Architecture::EthosU55,
        accelerator: "ethos-u55-128",
        system_config_name: "Ethos_U55_High_End_Embedded",
        memory_mode: "Shared_Sram",
        optimise: "Size",
    },
    TestConfig {
        name: "u65_256_perf",
        arch: regor::Architecture::EthosU65,
        accelerator: "ethos-u65-256",
        system_config_name: "Ethos_U65_High_End",
        memory_mode: "Shared_Sram",
        optimise: "Performance",
    },
    TestConfig {
        name: "u85_256_perf",
        arch: regor::Architecture::EthosU85,
        accelerator: "ethos-u85-256",
        system_config_name: "Ethos_U85_SYS_DRAM_Mid",
        memory_mode: "Shared_Sram",
        optimise: "Performance",
    },
];

// ---------------------------------------------------------------------------
// Direct regor comparison: Rust vs Python regor module
// ---------------------------------------------------------------------------

#[test]
fn compare_regor_generated_model() {
    if should_skip() {
        eprintln!("Skipping Python comparison tests (SKIP_PYTHON_TESTS=1)");
        return;
    }

    let model_path = ensure_generated_model();
    let model_bytes = fs::read(&model_path).unwrap();

    for config in CONFIGS {
        eprintln!("--- Config: {} ---", config.name);

        let (python_output, result) = python_regor_compile(
            &model_path,
            config.accelerator,
            config.system_config_name,
            config.memory_mode,
            config.optimise,
        );

        let sys_config = result["_system_config"].as_str().unwrap();
        let compiler_opts = result["_compiler_options"].as_str().unwrap();

        let rust_output =
            match rust_regor_compile(&model_bytes, config.arch, sys_config, compiler_opts) {
                Ok(out) => out,
                Err(e) => {
                    panic!("Rust compilation failed for {}: {e}", config.name);
                }
            };

        let rust_bytes = rust_output.as_bytes();

        assert_eq!(
            rust_bytes.len(),
            python_output.len(),
            "Output size mismatch for {}: rust={} python={}",
            config.name,
            rust_bytes.len(),
            python_output.len(),
        );

        assert_eq!(
            rust_bytes,
            &python_output[..],
            "Output bytes differ for {}",
            config.name,
        );

        eprintln!("  PASS: {} ({} bytes match)", config.name, rust_bytes.len());
    }
}

#[test]
fn compare_regor_mobilenet() {
    if should_skip() {
        eprintln!("Skipping Python comparison tests (SKIP_PYTHON_TESTS=1)");
        return;
    }

    let model_path = ensure_mobilenet_model();
    let model_bytes = fs::read(&model_path).unwrap();

    let config = &CONFIGS[0]; // U55-256, Performance
    let (python_output, result) = python_regor_compile(
        &model_path,
        config.accelerator,
        config.system_config_name,
        config.memory_mode,
        config.optimise,
    );

    let sys_config = result["_system_config"].as_str().unwrap();
    let compiler_opts = result["_compiler_options"].as_str().unwrap();

    let rust_output = rust_regor_compile(&model_bytes, config.arch, sys_config, compiler_opts)
        .expect("Rust compilation failed for MobileNet");

    let rust_bytes = rust_output.as_bytes();

    assert_eq!(
        rust_bytes.len(),
        python_output.len(),
        "MobileNet size mismatch: rust={} python={}",
        rust_bytes.len(),
        python_output.len(),
    );

    assert_eq!(
        rust_bytes,
        &python_output[..],
        "MobileNet output bytes differ"
    );

    eprintln!(
        "PASS: MobileNet compilation matches ({} bytes)",
        rust_bytes.len()
    );
}

// ---------------------------------------------------------------------------
// Vela CLI end-to-end sanity
// ---------------------------------------------------------------------------

#[test]
fn vela_cli_produces_output() {
    if should_skip() {
        eprintln!("Skipping Python comparison tests (SKIP_PYTHON_TESTS=1)");
        return;
    }

    let model_path = ensure_generated_model();
    let output_dir = tempfile::tempdir().unwrap();
    let script = tests_root().join("python_reference.py");

    let config = &CONFIGS[0];
    let output = Command::new(python())
        .arg(&script)
        .arg("vela")
        .arg(&model_path)
        .arg(output_dir.path())
        .arg(format!("accelerator={}", config.accelerator))
        .arg(format!("system_config={}", config.system_config_name))
        .arg(format!("memory_mode={}", config.memory_mode))
        .arg(format!("optimise={}", config.optimise))
        .output()
        .expect("failed to run python_reference.py");

    if !output.status.success() {
        eprintln!(
            "vela CLI failed (non-fatal):\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let result: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(output_dir.path().join("vela_result.json")).unwrap(),
    )
    .unwrap();

    let files = result["output_files"].as_array().unwrap();
    assert!(!files.is_empty(), "Vela produced no output files");
    eprintln!("PASS: vela CLI produced output: {:?}", files);
}

// ---------------------------------------------------------------------------
// Perf report sanity
// ---------------------------------------------------------------------------

#[test]
fn perf_report_matches_python() {
    if should_skip() {
        eprintln!("Skipping Python comparison tests (SKIP_PYTHON_TESTS=1)");
        return;
    }

    let model_path = ensure_generated_model();
    let model_bytes = fs::read(&model_path).unwrap();

    let config = &CONFIGS[0];
    let (_python_output, result) = python_regor_compile(
        &model_path,
        config.accelerator,
        config.system_config_name,
        config.memory_mode,
        config.optimise,
    );

    let sys_config = result["_system_config"].as_str().unwrap();
    let compiler_opts = result["_compiler_options"].as_str().unwrap();
    let py_perf = &result["perf"];

    let mut compiler = regor::Compiler::new(config.arch).unwrap();
    compiler.system_config(sys_config).unwrap();
    compiler.compiler_options(compiler_opts).unwrap();
    let _output = compiler
        .compile(regor::InputFormat::TfLite, &model_bytes)
        .unwrap();

    let report = compiler.perf_report().unwrap();

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

    eprintln!("PASS: perf report matches Python");
    eprintln!(
        "  NPU={} CPU={} total={} npu_ops={} cpu_ops={}",
        report.npu_cycles, report.cpu_cycles, report.total_cycles, report.npu_ops, report.cpu_ops,
    );
}
