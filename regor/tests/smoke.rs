use regor::{Architecture, Compiler};

#[test]
fn create_context() {
    let compiler = Compiler::new(Architecture::EthosU55);
    assert!(
        compiler.is_ok(),
        "Failed to create compiler: {:?}",
        compiler.err()
    );
    eprintln!("Created EthosU55 compiler context successfully");
}

#[test]
fn create_all_architectures() {
    for arch in [
        Architecture::EthosU55,
        Architecture::EthosU65,
        Architecture::EthosU85,
    ] {
        let compiler = Compiler::new(arch);
        assert!(
            compiler.is_ok(),
            "Failed for {:?}: {:?}",
            arch,
            compiler.err()
        );
    }
    eprintln!("All architectures created successfully");
}
