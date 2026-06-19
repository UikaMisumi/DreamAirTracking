# Model Packages

Do not commit large model weights directly to the repository.

The current public Dream Air model package is hosted at:

https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current

A model package should contain:

```text
model.onnx
metadata.json
runtime_defaults.json
acceptance.json
model_card.md
sha256.txt
```

The app should discover packages through `model_registry.json`, whose public schema is documented in `model_registry.schema.json`.
