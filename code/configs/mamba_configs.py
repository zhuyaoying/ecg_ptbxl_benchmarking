conf_mamba_small = {
    'modelname': 'mamba_small',
    'modeltype': 'mamba_model',
    'parameters': dict(
        d_model=128,
        n_layers=6,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        epochs=30,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-2,
    ),
}

conf_mamba_base = {
    'modelname': 'mamba_base',
    'modeltype': 'mamba_model',
    'parameters': dict(
        d_model=192,
        n_layers=8,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        epochs=40,
        batch_size=24,
        lr=8e-4,
        weight_decay=1e-2,
    ),
}

conf_mamba_large = {
    'modelname': 'mamba_large',
    'modeltype': 'mamba_model',
    'parameters': dict(
        d_model=256,
        n_layers=12,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        epochs=50,
        batch_size=16,
        lr=6e-4,
        weight_decay=1e-2,
    ),
}

MAMBA_MODEL_CONFIGS = [
    conf_mamba_small,
    conf_mamba_base,
    conf_mamba_large,
]
