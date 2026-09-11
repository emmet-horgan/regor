use std::ffi::CStr;

/// Target Ethos-U architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    EthosU55,
    EthosU65,
    EthosU85,
}

impl Architecture {
    pub(crate) fn as_cstr(self) -> &'static CStr {
        match self {
            Architecture::EthosU55 => {
                unsafe { CStr::from_bytes_with_nul_unchecked(regor_sys::REGOR_ARCH_ETHOSU55) }
            }
            Architecture::EthosU65 => {
                unsafe { CStr::from_bytes_with_nul_unchecked(regor_sys::REGOR_ARCH_ETHOSU65) }
            }
            Architecture::EthosU85 => {
                unsafe { CStr::from_bytes_with_nul_unchecked(regor_sys::REGOR_ARCH_ETHOSU85) }
            }
        }
    }
}
