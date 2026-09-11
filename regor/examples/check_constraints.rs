use regor::{Architecture, Compiler, InputFormat};

fn main() -> regor::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .expect("usage: check_constraints <model.tflite>");

    let model = std::fs::read(&model_path).expect("failed to read model file");

    let mut compiler = Compiler::new(Architecture::EthosU55)?;
    let _output = compiler.compile(InputFormat::TfLite, &model)?;

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
