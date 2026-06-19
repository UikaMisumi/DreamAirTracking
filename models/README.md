# Model Packages

Do not commit large model weights directly to the repository.

The current public Dream Air model package is hosted at:

https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current

Download that Hugging Face repository and copy its contents into:

```text
%LOCALAPPDATA%\DreamAirTracking\models\
```

The installed layout should be:

```text
%LOCALAPPDATA%\DreamAirTracking\models\model_registry.json
%LOCALAPPDATA%\DreamAirTracking\models\sha256.txt

%LOCALAPPDATA%\DreamAirTracking\models\dreamair-main-current\model.onnx
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-main-current\metadata.json
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-main-current\runtime_defaults.json
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-main-current\acceptance.json
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-main-current\model_card.md

%LOCALAPPDATA%\DreamAirTracking\models\dreamair-expression-current\model.onnx
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-expression-current\metadata.json
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-expression-current\runtime_defaults.json
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-expression-current\acceptance.json
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-expression-current\model_card.md
```

The app should discover packages through `model_registry.json`, whose public schema is documented in `model_registry.schema.json`.
