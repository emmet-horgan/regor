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

def download_mlcommons_tiny_models(output_dir: str) -> None:
    """Download and cache MLCommons Tiny models for integration testing."""

    output_dir = pathlib.Path(output_dir)

    root = 'https://github.com/mlcommons/tiny/raw/refs/tags/v1.1/benchmark/training'

    kws_ref_name = 'kws_ref_model.tflite'
    kws_ref_float_name = 'kws_ref_model_float32.tflite'
    vww_96_float_name = 'vww_96_float.tflite'
    vww_96_int8_name = 'vww_96_int8.tflite'
    resnet_name = 'pretrainedResnet.tflite'
    resnet_quant_name = 'pretrainedResnet_quant.tflite'

    kws_ref = f'{root}/keyword_spotting/trained_models/{kws_ref_name}'
    kws_ref_float = f'{root}/keyword_spotting/trained_models/{kws_ref_float_name}'
    vww_96_float = f'{root}/visual_wake_words/trained_models/{vww_96_float_name}'
    vww_96_int8 = f'{root}/visual_wake_words/trained_models/{vww_96_int8_name}'
    resnet = f'{root}/image_classification/trained_models/{resnet_name}'
    resnet_quant = f'{root}/image_classification/trained_models/{resnet_quant_name}'

    if (output_dir / kws_ref_name).exists():
        print(f"Model already cached: {output_dir / kws_ref_name}")
    else:
        urllib.request.urlretrieve(kws_ref, output_dir / kws_ref_name)
    if (output_dir / kws_ref_float_name).exists():
        print(f"Model already cached: {output_dir / kws_ref_float_name}")
    else:
        urllib.request.urlretrieve(kws_ref_float, output_dir / kws_ref_float_name)
    if (output_dir / vww_96_float_name).exists():
        print(f"Model already cached: {output_dir / vww_96_float_name}")
    else:
        urllib.request.urlretrieve(vww_96_float, output_dir / vww_96_float_name)
    if (output_dir / vww_96_int8_name).exists():
        print(f"Model already cached: {output_dir / vww_96_int8_name}")
    else:
        urllib.request.urlretrieve(vww_96_int8, output_dir / vww_96_int8_name)
    if (output_dir / resnet_name).exists():
        print(f"Model already cached: {output_dir / resnet_name}")
    else:
        urllib.request.urlretrieve(resnet, output_dir / resnet_name)
    if (output_dir / resnet_quant_name).exists():
        print(f"Model already cached: {output_dir / resnet_quant_name}")
    else:
        urllib.request.urlretrieve(resnet_quant, output_dir / resnet_quant_name)


def main():
    output = "tests/fixtures/mobilenet_v1_quant.tflite"
    download_model(output)
    download_mlcommons_tiny_models("tests/fixtures/")


if __name__ == "__main__":
    main()
