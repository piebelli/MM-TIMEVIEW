## MM-TIMEVIEW

**MM-TIMEVIEW** is a multimodal extension of TIMEVIEW that supports multiple input modalities (e.g., clinical data, imaging features, genomics). Key features:

- **Modality-specific encoders**: Each modality is processed by a dedicated encoder
- **Fusion network**: Combines modality embeddings to predict basis coefficients
- **Modality attribution**: Quantifies each modality's contribution to predictions

### Quick Start
```python
from mm_timeview import create_mm_timeview, train_mm_timeview, compute_modality_attributions

# Define modality structure
modality_config = {
    'clinical': {'indices': [0, 1, 2], 'embedding_dim': 8},
    'imaging': {'indices': [3, 4, 5, 6], 'embedding_dim': 8}
}

# Create and train model
model = create_mm_timeview(modality_config, n_basis=10)
train_mm_timeview(model, X_train, Phi, Y_train)

# Compute modality attributions
attributions = compute_modality_attributions(model, X_test, Phi, Y_test)
```

See `mm_timeview/mm_timeview.ipynb` for a complete example.

### Dependencies

Install required dependencies with conda:

```bash
conda env create -n timeview --file environment.yml
```

This will also install `timeview` (the main module) in editable mode.