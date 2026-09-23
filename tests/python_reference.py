#!/usr/bin/env python3
"""Reference compilation using Python regor bindings and vela CLI.

Provides functions to compile a .tflite model using:
1. The regor Python module directly (same C library the Rust bindings wrap)
2. The vela CLI tool (full pipeline)

Results are written as files that the Rust integration tests can compare against.

Usage:
    python python_reference.py <mode> <input_model> <output_dir> [options...]

Modes:
    regor   -- Direct regor module compilation
    vela    -- Full vela CLI compilation

Options (key=value):
    accelerator=ethos-u55-256       (regor mode: sets arch from this)
    system_config=Ethos_U55_High_End_Embedded
    memory_mode=Shared_Sram
    optimise=Performance
    arena_cache_size=0
    tensor_allocator=HillClimb
    config_path=/path/to/vela.ini   (optional, uses installed default)
"""

import json
import pathlib
import subprocess
import sys


def _find_vela_ini() -> str:
    """Locate the installed vela.ini config file."""
    try:
        import ethosu.vela
        vela_dir = pathlib.Path(ethosu.vela.__file__).parent
        candidates = list(vela_dir.parent.rglob("vela.ini"))
        if candidates:
            return str(candidates[0])
    except ImportError:
        pass
    raise RuntimeError("Could not find vela.ini. Install ethos-u-vela.")


def _arch_from_accelerator(accelerator: str) -> tuple[str, int, int]:
    """Return (arch_name, macs, cores) from an accelerator config string."""
    mapping = {
        "ethos-u55-32": ("EthosU55", 32, 1),
        "ethos-u55-64": ("EthosU55", 64, 1),
        "ethos-u55-128": ("EthosU55", 128, 1),
        "ethos-u55-256": ("EthosU55", 256, 1),
        "ethos-u65-256": ("EthosU65", 256, 1),
        "ethos-u65-512": ("EthosU65", 512, 2),
        "ethos-u85-128": ("EthosU85", 128, 1),
        "ethos-u85-256": ("EthosU85", 256, 1),
        "ethos-u85-512": ("EthosU85", 512, 1),
        "ethos-u85-1024": ("EthosU85", 1024, 1),
        "ethos-u85-2048": ("EthosU85", 2048, 2),
    }
    if accelerator not in mapping:
        raise ValueError(f"Unknown accelerator: {accelerator}")
    return mapping[accelerator]


def _build_system_config(
    accelerator: str,
    system_config: str,
    memory_mode: str,
    config_path: str | None = None,
) -> str:
    """Build the system config string the way vela does for regor."""
    arch_name, macs, cores = _arch_from_accelerator(accelerator)

    if config_path is None:
        config_path = _find_vela_ini()

    ini_content = pathlib.Path(config_path).read_text()

    sysconfig = f"[architecture]\nmacs={macs}\ncores={cores}\n"
    sysconfig += f"[vela]\nsystem_config_name={system_config}\n"
    sysconfig += f"memory_mode_name={memory_mode}\n"
    sysconfig += ini_content
    return sysconfig


def _build_compiler_options(
    optimise: str = "Performance",
    arena_cache_size: int = 0,
    tensor_allocator: str = "HillClimb",
    cop_format: str = "COP1",
    cpu_tensor_alignment: int = 16,
    output_format: str = "TFLite",
) -> str:
    """Build the compiler options string the way vela does for regor."""
    config = "\n[compiler]\n"
    config += f"output_format={output_format}\n"
    config += f"cop_format={cop_format}\n"
    config += "\n[scheduler]\n"
    config += f"optimize={optimise}\n"
    config += f"arena_size_limit={arena_cache_size}\n"
    config += "disable_feature=\n"
    config += f"cpu_tensor_alignment={cpu_tensor_alignment}\n"
    config += f"tensor_allocator={tensor_allocator}\n"
    config += "\n[graph]\n"
    return config


def compile_with_regor(
    input_path: str,
    output_dir: str,
    accelerator: str = "ethos-u55-256",
    system_config: str = "Ethos_U55_High_End_Embedded",
    memory_mode: str = "Shared_Sram",
    optimise: str = "Performance",
    arena_cache_size: str = "0",
    tensor_allocator: str = "HillClimb",
    config_path: str | None = None,
) -> dict:
    """Compile using the regor Python module directly."""
    import ethosu.regor as regor

    arch_name, _, _ = _arch_from_accelerator(accelerator)
    input_data = pathlib.Path(input_path).read_bytes()

    sysconfig = _build_system_config(accelerator, system_config, memory_mode, config_path)
    options = _build_compiler_options(
        optimise=optimise,
        arena_cache_size=int(arena_cache_size),
        tensor_allocator=tensor_allocator,
    )

    compiled = regor.compile(
        arch_name,
        input_data,
        "TFLITE",
        sysconfig,
        options=options,
        verbose=False,
    )

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    output_bytes = bytes(compiled.model)
    output_path = out / "regor_output.tflite"
    output_path.write_bytes(output_bytes)

    # Also save the exact config strings so Rust can use them
    (out / "system_config.ini").write_text(sysconfig)
    (out / "compiler_options.ini").write_text(options)

    # Get perf report via class API
    r = regor.Regor(arch_name, False)
    r.SetSystemConfig(sysconfig)
    r.SetCompilerOptions(options)
    r.Compile(input_data, "TFLITE")
    report = r.GetPerfReport()

    result = {
        "mode": "regor",
        "accelerator": accelerator,
        "arch": arch_name,
        "system_config_name": system_config,
        "memory_mode": memory_mode,
        "optimise": optimise,
        "output_path": str(output_path),
        "output_size": len(output_bytes),
        "perf": {
            "npu_cycles": report.npuCycles,
            "cpu_cycles": report.cpuCycles,
            "total_cycles": report.totalCycles,
            "npu_ops": report.npuOps,
            "cpu_ops": report.cpuOps,
            "original_weights": report.originalWeights,
            "encoded_weights": report.encodedWeights,
        },
    }

    result_path = out / "regor_result.json"
    result_path.write_text(json.dumps(result, indent=2))
    return result


def compile_with_vela(
    input_path: str,
    output_dir: str,
    accelerator: str = "ethos-u55-256",
    system_config: str = "Ethos_U55_High_End_Embedded",
    memory_mode: str = "Shared_Sram",
    optimise: str = "Performance",
) -> dict:
    """Compile using the vela CLI."""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "ethosu.vela",
        str(input_path),
        "--accelerator-config", accelerator,
        "--system-config", system_config,
        "--memory-mode", memory_mode,
        "--optimise", optimise,
        "--output-dir", str(out),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    input_stem = pathlib.Path(input_path).stem
    output_files = list(out.glob(f"{input_stem}_vela.*"))

    info = {
        "mode": "vela",
        "accelerator": accelerator,
        "system_config": system_config,
        "memory_mode": memory_mode,
        "optimise": optimise,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_files": [str(f) for f in output_files],
    }

    if output_files:
        info["output_size"] = output_files[0].stat().st_size

    result_path = out / "vela_result.json"
    result_path.write_text(json.dumps(info, indent=2))
    return info


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <regor|vela> <input_model> <output_dir> [key=value...]")
        sys.exit(1)

    mode = sys.argv[1]
    input_path = sys.argv[2]
    output_dir = sys.argv[3]

    kwargs = {}
    for arg in sys.argv[4:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            kwargs[k] = v

    if mode == "regor":
        result = compile_with_regor(input_path, output_dir, **kwargs)
    elif mode == "vela":
        result = compile_with_vela(input_path, output_dir, **kwargs)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
