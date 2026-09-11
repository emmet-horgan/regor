use regor::{Architecture, Compiler, InputFormat};

fn main() -> regor::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .expect("usage: compile_tflite <model.tflite> [output.tflite]");
    let output_path = std::env::args().nth(2).unwrap_or_else(|| "output.tflite".into());

    let model = std::fs::read(&model_path).expect("failed to read model file");

    let mut compiler = Compiler::new(Architecture::EthosU55)?;
    let output = compiler.compile(InputFormat::TfLite, &model)?;

    let report = compiler.perf_report()?;
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
