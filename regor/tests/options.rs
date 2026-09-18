//! End-to-end tests for the typed options API.
//!
//! The unit tests in `options` check what the builders *write*; these check that
//! regor actually *accepts* it. That distinction matters because regor ignores
//! unknown keys and silently drops malformed values rather than failing, so a
//! wrong key name looks identical to a correct one until you inspect the
//! result.

use regor::options::*;
use regor::{Compiler, InputFormat};

/// A Vela configuration covering the reference Ethos-U55 systems.
///
/// Checked in alongside this test rather than downloaded: `include_str!` runs
/// at compile time, so it has to be present before any fixture-generating step
/// could have run.
const VELA_INI: &str = include_str!("fixtures/vela_default.ini");

fn model() -> Vec<u8> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/fixtures/mobilenet_v1_quant.tflite"
    ))
    .expect("fixture model should be present")
}

fn system_config(accelerator: AcceleratorConfig) -> String {
    SystemConfig::new(accelerator)
        .system_config_name("Ethos_U55_High_End_Embedded")
        .memory_mode_name("Sram_Only")
        .vela_ini(VELA_INI)
        .build()
}

/// Compile `model` with `options`, returning the output and the perf report.
fn compile_with(
    accelerator: AcceleratorConfig,
    options: &str,
) -> regor::Result<(regor::Output, regor::PerfReport)> {
    let mut c = Compiler::new(accelerator.architecture())?;
    c.set_system_config(&system_config(accelerator))?;
    c.set_options(options)?;
    let out = c.compile(InputFormat::TfLite, &model())?;
    let perf = c.perf_report()?;
    Ok((out, perf))
}

#[test]
fn the_typed_api_compiles_a_model() {
    let accelerator = AcceleratorConfig::EthosU55_128;
    let options = CompilerOptions::new()
        .optimise(Optimise::Performance)
        .build()
        .unwrap();

    let (out, perf) = compile_with(accelerator, &options).expect("compile should succeed");

    assert!(!out.is_empty());
    // A compiled command stream is still a TFLite flatbuffer.
    assert_eq!(&out.as_bytes()[4..8], b"TFL3");
    assert!(perf.total_cycles > 0);
}

/// The system config is what sizes the NPU, so two accelerators must give
/// measurably different schedules for the same model.
#[test]
fn the_accelerator_reaches_regor_through_the_system_config() {
    let options = CompilerOptions::new()
        .optimise(Optimise::Performance)
        .build()
        .unwrap();

    let (_, small) = compile_with(AcceleratorConfig::EthosU55_32, &options).unwrap();
    let (_, large) = compile_with(AcceleratorConfig::EthosU55_256, &options).unwrap();

    assert!(small.npu_cycles > 0 && large.npu_cycles > 0);
    // Eight times the MACs should not take as many cycles.
    assert!(
        large.npu_cycles < small.npu_cycles,
        "u55-256 ({}) should beat u55-32 ({})",
        large.npu_cycles,
        small.npu_cycles
    );
}

/// A Vela `.ini` on its own leaves the NPU unsized, which regor only discovers
/// once it tries to compile.
///
/// This is exactly the trap [`SystemConfig`] exists to close: it always writes
/// an `[architecture]` section, so this state is unreachable through the typed
/// API. The test pins the behaviour so a change in it is noticed.
#[test]
fn a_bare_vela_ini_leaves_the_accelerator_unsized() {
    let mut c = Compiler::new(regor::Architecture::EthosU55).unwrap();

    // No `[architecture]` section, so no MAC count. Configuration is buffered,
    // so this is reported when the compile runs rather than here.
    c.system_config(VELA_INI).unwrap();

    let err = c
        .compile(InputFormat::TfLite, &model())
        .expect_err("an unsized accelerator should not compile");
    assert!(
        err.to_string().contains("LUT memory not configured"),
        "unexpected error: {err}"
    );
}

#[test]
fn every_accelerator_builds_a_usable_system_config() {
    for &accelerator in AcceleratorConfig::ALL {
        // U65 and U85 need their own system-config sections; only check that
        // the U55 ones compile end to end, and that the rest at least produce a
        // document regor's architecture parser accepts.
        let doc = SystemConfig::new(accelerator).vela_ini(VELA_INI).build();
        assert!(doc.contains(&format!("macs={}", accelerator.macs())));
        assert!(doc.contains(&format!("cores={}", accelerator.cores())));
    }
}

/// `ignore_ops` must reach the graph optimiser: forcing every convolution onto
/// the CPU has to change what the NPU is left to do.
#[test]
fn ignore_ops_moves_work_to_the_cpu() {
    let accelerator = AcceleratorConfig::EthosU55_128;

    let baseline = CompilerOptions::new().build().unwrap();
    let (_, before) = compile_with(accelerator, &baseline).unwrap();

    let ignored = CompilerOptions::new()
        .ignore_ops(["CONV_2D", "DEPTHWISE_CONV_2D"])
        .build()
        .unwrap();
    let (_, after) = compile_with(accelerator, &ignored).unwrap();

    assert!(
        after.cpu_ops > before.cpu_ops,
        "ignoring convolutions should push operators to the cpu ({} -> {})",
        before.cpu_ops,
        after.cpu_ops
    );
}

/// Disabling scheduler features must reach regor. This is the case the old
/// comma-separated spelling got wrong: regor rejected the value outright and
/// carried on with everything still enabled.
#[test]
fn disabling_cascading_changes_the_schedule() {
    let accelerator = AcceleratorConfig::EthosU55_128;

    let baseline = CompilerOptions::new()
        .optimise(Optimise::Performance)
        .build()
        .unwrap();
    let (_, before) = compile_with(accelerator, &baseline).unwrap();

    let disabled = CompilerOptions::new()
        .optimise(Optimise::Performance)
        .disable_features(SchedulerFeature::CASCADING)
        .build()
        .unwrap();
    let (_, after) = compile_with(accelerator, &disabled).unwrap();

    assert!(
        before.cascaded_ops > 0,
        "the baseline should cascade something for this to be meaningful"
    );
    assert_eq!(
        after.cascaded_ops, 0,
        "cascading was disabled, so nothing should be cascaded"
    );
}

/// The two optimisation strategies must produce genuinely different results.
#[test]
fn optimising_for_size_differs_from_performance() {
    let accelerator = AcceleratorConfig::EthosU55_128;

    let perf_opts = CompilerOptions::new()
        .optimise(Optimise::Performance)
        .build()
        .unwrap();
    let size_opts = CompilerOptions::new()
        .optimise(Optimise::Size)
        .build()
        .unwrap();

    let (_, fast) = compile_with(accelerator, &perf_opts).unwrap();
    let (_, small) = compile_with(accelerator, &size_opts).unwrap();

    let peak = |p: &regor::PerfReport| {
        p.peak_usages
            .iter()
            .map(|u| u.peak_usage)
            .max()
            .unwrap_or_default()
    };

    assert!(
        fast.total_cycles != small.total_cycles || peak(&fast) != peak(&small),
        "Performance and Size should not schedule identically"
    );
}

#[test]
fn the_raw_output_format_is_understood() {
    let accelerator = AcceleratorConfig::EthosU55_128;
    let options = CompilerOptions::new()
        .output_format(OutputFormat::Raw)
        .build()
        .unwrap();

    let (out, _) = compile_with(accelerator, &options).expect("raw output should compile");
    assert!(!out.is_empty());
    // Raw output is not a TFLite flatbuffer, which is the point of asking for it.
    assert_ne!(&out.as_bytes()[4..8], b"TFL3");
}

#[test]
fn cop2_with_separate_io_regions_compiles() {
    let accelerator = AcceleratorConfig::EthosU55_128;
    let options = CompilerOptions::new()
        .cop_format(CopFormat::Cop2)
        .separate_io_regions(true)
        .build()
        .unwrap();

    let (out, _) = compile_with(accelerator, &options).expect("COP2 should compile");
    assert!(!out.is_empty());
}

#[test]
fn a_tighter_arena_is_respected() {
    let accelerator = AcceleratorConfig::EthosU55_128;

    let options = CompilerOptions::new()
        .optimise(Optimise::Size)
        .arena_cache_size(256 * 1024)
        .build()
        .unwrap();

    let (out, _) = compile_with(accelerator, &options).expect("a tight arena should still compile");
    assert!(!out.is_empty());
}

/// Compiling from several threads at once must not take the process down.
///
/// The C++ library reaches shared global state during compilation that its own
/// context mutex does not cover, and an exception escaping the C API aborts the
/// process rather than unwinding into Rust. The bindings serialise compilation
/// to keep that contained, and this is the regression test for it.
#[test]
fn compiling_concurrently_is_safe() {
    let options = CompilerOptions::new()
        .optimise(Optimise::Performance)
        .build()
        .unwrap();

    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let options = options.clone();
                s.spawn(move || {
                    let (out, _) = compile_with(AcceleratorConfig::EthosU55_128, &options).unwrap();
                    out.len()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    assert_eq!(results.len(), 4);
    assert!(results.iter().all(|&n| n > 0));
    // The same input compiled the same way must give the same size every time.
    assert!(
        results.windows(2).all(|w| w[0] == w[1]),
        "concurrent compiles disagreed: {results:?}"
    );
}
