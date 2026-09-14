use std::ffi::CStr;

use regor_sys as ffi;

/// Memory access performance for a single region.
#[derive(Debug, Clone)]
pub struct MemoryAccessPerf {
    pub memory_name: String,
    pub access_type: String,
    pub bytes_read: i64,
    pub bytes_written: i64,
    pub access_cycles: i64,
}

/// Peak memory usage for a single region.
#[derive(Debug, Clone)]
pub struct PeakMemoryUsage {
    pub memory_name: String,
    pub peak_usage: i64,
    pub total_access_cycles: i64,
}

/// Performance report from a compilation.
#[derive(Debug, Clone)]
pub struct PerfReport {
    pub npu_cycles: i64,
    pub cpu_cycles: i64,
    pub total_cycles: i64,
    pub mac_count: i64,
    pub cpu_ops: i64,
    pub npu_ops: i64,
    pub cascaded_ops: i64,
    pub cascades: i64,
    pub original_weights: i64,
    pub encoded_weights: i64,
    pub read_only_peak_usage: i32,
    pub access_count: i32,
    pub memory: i32,
    pub num_memories: i32,
    pub staging_memory: i32,
    pub peak_usages: Vec<PeakMemoryUsage>,
    pub accesses: Vec<MemoryAccessPerf>,
}

fn char_array_to_string(buf: &[std::os::raw::c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

impl PerfReport {
    pub(crate) unsafe fn from_ffi(raw: &ffi::regor_perf_report_t) -> Self {
        let peak_usages = raw
            .peak_usages
            .iter()
            .map(|p| PeakMemoryUsage {
                memory_name: char_array_to_string(&p.memory_name),
                peak_usage: p.peak_usage,
                total_access_cycles: p.total_access_cycles,
            })
            .collect();

        let accesses = if raw.access.is_null() || raw.access_count <= 0 {
            Vec::new()
        } else {
            let slice = std::slice::from_raw_parts(raw.access, raw.access_count as usize);
            slice
                .iter()
                .map(|a| MemoryAccessPerf {
                    memory_name: char_array_to_string(&a.memory_name),
                    access_type: char_array_to_string(&a.access_type),
                    bytes_read: a.bytes_read,
                    bytes_written: a.bytes_written,
                    access_cycles: a.access_cycles,
                })
                .collect()
        };

        PerfReport {
            npu_cycles: raw.npu_cycles,
            cpu_cycles: raw.cpu_cycles,
            total_cycles: raw.total_cycles,
            mac_count: raw.mac_count,
            cpu_ops: raw.cpu_ops,
            npu_ops: raw.npu_ops,
            cascaded_ops: raw.cascaded_ops,
            cascades: raw.cascades,
            original_weights: raw.original_weights,
            encoded_weights: raw.encoded_weights,
            read_only_peak_usage: raw.read_only_peak_usage,
            access_count: raw.access_count,
            memory: raw.memory,
            num_memories: raw.num_memories,
            staging_memory: raw.staging_memory,
            peak_usages,
            accesses,
        }
    }
}
