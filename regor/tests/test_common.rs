use std::env;
use std::path::PathBuf;


pub fn should_skip() -> bool {
    std::env::var("SKIP_PYTHON_TESTS").is_ok()
}

pub fn python() -> String {
    std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string())
}

pub fn tests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("tests")
}
