#!/usr/bin/env python3
"""Download and cache a small public quantized TFLite model for integration testing.

Usage:
    python download_model.py [output_path]

Downloads a quantized MobileNet v1 (0.25, 128) model from TensorFlow's hosted models.
This model is small (~500KB) and fully int8-quantized, suitable for Ethos-U compilation.
"""

import hashlib
import pathlib
import sys
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/"
    "mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz"
)
EXPECTED_SHA256 = None  # Set after first download if pinning is desired
MODEL_FILENAME = "mobilenet_v1_0.25_128_quant.tflite"


def download_model(output_path: str) -> None:
    out = pathlib.Path(output_path)
    if out.exists():
        print(f"Model already cached: {out}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)

    import tempfile
    import tarfile

    print(f"Downloading {MODEL_URL} ...")
    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
        urllib.request.urlretrieve(MODEL_URL, tmp.name)
        tmp_path = pathlib.Path(tmp.name)

    try:
        with tarfile.open(tmp_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".tflite"):
                    f = tar.extractfile(member)
                    if f is not None:
                        data = f.read()
                        out.write_bytes(data)
                        sha = hashlib.sha256(data).hexdigest()
                        print(f"Saved {out} ({len(data)} bytes, sha256={sha[:16]}...)")
                        return

        raise RuntimeError("No .tflite file found in archive")
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/mobilenet_v1_quant.tflite"
    download_model(output)


if __name__ == "__main__":
    main()
