const VELA_INI: &str = include_str!("../tests/fixtures/vela_default.ini");

fn main() -> regor::Result<()> {
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
        .expect("usage: check_constraints <model.tflite>");

    let model = std::fs::read(&model_path).expect("failed to read model file");

    let accelerator = regor::AcceleratorConfig::EthosU55_256;

    let system = regor::SystemConfig::new(accelerator)
        .system_config_name("Ethos_U55_High_End_Embedded")
        .memory_mode_name("Shared_Sram")
        .vela_ini(VELA_INI)
        .build();

    let options = regor::CompilerOptions::new()
        .optimize(regor::Optimize::Performance)
        .build()
        .expect("failed to build compiler options");

    let mut compiler = regor::Compiler::new(accelerator.architecture())?;
    compiler.set_system_config(&system)?;
    compiler.set_options(&options)?;

    let _output = compiler.compile(regor::InputFormat::TfLite, &model)?;

    let report = compiler.tflite_constraints()?;
    for op in &report.operators {
        if op.constraints.is_empty() {
            continue;
        }
        println!("{}:", op.operator_name);
        for c in &op.constraints {
            println!("  - {c}");
        }
    }

    if report.operators.iter().all(|op| op.constraints.is_empty()) {
        println!("All operators are fully supported on the NPU.");
    }

    Ok(())
}
