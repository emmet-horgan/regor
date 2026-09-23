const VELA_INI: &str = include_str!("../tests/fixtures/vela_default.ini");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber when the `tracing` feature is enabled.
    #[cfg(feature = "tracing")]
    {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .init();
        let _ = regor::logging::init(regor::logging::LogFilter::all());
    }

    let model_path = std::env::args()
        .nth(1)
        .expect("usage: compile_tflite <model.tflite> [output.tflite]");
    let output_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "output.tflite".into());

    let model = std::fs::read(&model_path).expect("failed to read model file");

    let accelerator = regor::AcceleratorConfig::EthosU55_256;

    let system = regor::SystemConfig::new(accelerator)
        .system_config_name("Ethos_U55_High_End_Embedded")
        .memory_mode_name("Shared_Sram")
        .vela_ini(VELA_INI)
        .build();

    let options = regor::CompilerOptions::new()
        .optimize(regor::Optimize::Performance)
        .build().expect("failed to build compiler options");

    let mut compiler = regor::Compiler::new(accelerator.architecture())?;
    compiler.set_system_config(&system)?;
    compiler.set_options(&options)?;
    let output = compiler.compile(regor::InputFormat::TfLite, &model).expect("failed to compile");

    let report = compiler.perf_report().expect("failed to generate performance report");
    eprintln!(
        "NPU cycles: {}, CPU cycles: {}, total: {}",
        report.npu_cycles, report.cpu_cycles, report.total_cycles,
    );
    eprintln!(
        "NPU ops: {}, CPU ops: {}, cascaded: {}",
        report.npu_ops, report.cpu_ops, report.cascaded_ops,
    );
    eprintln!(
        "Weights: {} -> {} bytes",
        report.original_weights, report.encoded_weights,
    );

    std::fs::write(&output_path, output.as_bytes()).expect("failed to write output");
    eprintln!("Wrote {} bytes to {output_path}", output.len());

    Ok(())
}
