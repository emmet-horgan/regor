use std::ffi::CStr;

use regor_sys as ffi;

/// A single operator's constraints (reasons it may not be accelerated).
#[derive(Debug, Clone)]
pub struct OperatorConstraints {
    pub operator_name: String,
    pub constraints: Vec<String>,
}

/// Report of all operator constraints from TFLite analysis.
#[derive(Debug, Clone)]
pub struct ConstraintsReport {
    pub operators: Vec<OperatorConstraints>,
}

impl ConstraintsReport {
    pub(crate) unsafe fn from_ffi(raw: &ffi::regor_operator_constraints_report_t) -> Self {
        if raw.op_constraints.is_null() || raw.operators <= 0 {
            return ConstraintsReport {
                operators: Vec::new(),
            };
        }

        let slice = std::slice::from_raw_parts(raw.op_constraints, raw.operators as usize);

        let operators = slice
            .iter()
            .map(|op| {
                let name = CStr::from_ptr(op.operator_name.as_ptr())
                    .to_string_lossy()
                    .into_owned();

                let count = op.constraints.max(0) as usize;
                let constraints = (0..count)
                    .map(|i| {
                        CStr::from_ptr(op.constraint[i].as_ptr())
                            .to_string_lossy()
                            .into_owned()
                    })
                    .collect();

                OperatorConstraints {
                    operator_name: name,
                    constraints,
                }
            })
            .collect();

        ConstraintsReport { operators }
    }
}
