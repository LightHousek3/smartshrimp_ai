import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch

from training.artifacts import load_artifact, seal, sha256, write_json


def export_onnx(checkpoint):
    model, config, meta, directory = load_artifact(checkpoint)
    torch.set_num_threads(config["num_threads"])
    generator = torch.Generator().manual_seed(config["seed"])
    sample = torch.randn(2, 3, config["input_size"], config["input_size"], generator=generator)
    output = directory / "model.onnx"
    torch.onnx.export(
        model,
        sample,
        str(output),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(output)))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    errors = []
    for size in (1, 2, 5):
        batch = torch.randn(size, 3, 224, 224, generator=generator)
        with torch.inference_mode():
            expected = model(batch).numpy()
        actual = session.run(None, {"images": batch.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        errors.append(float(np.max(np.abs(actual - expected))))
    write_json(
        directory / "export.json",
        {
            "format": "onnx",
            "opset": 17,
            "output": "logits (apply softmax before aggregation)",
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "onnx_sha256": sha256(output),
            "parity_max_absolute_error": max(errors),
            "tested_batch_sizes": [1, 2, 5],
        },
    )
    seal(directory)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", choices=["onnx"], default="onnx")
    print(export_onnx(parser.parse_args().checkpoint))
