from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.base_model import ClassificationModel

try:
    from mamba_ssm import Mamba
except ImportError:
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
    except ImportError:
        Mamba = None


class ECGDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = [np.asarray(sample, dtype=np.float32) for sample in X]
        self.y = None if y is None else [np.asarray(target, dtype=np.float32) for target in y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        if self.y is None:
            return x
        return x, torch.from_numpy(self.y[idx])


class MambaSequenceClassifier(nn.Module):
    def __init__(self, input_channels, num_classes, d_model=128, n_layers=6, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_channels, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "mamba": Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                "dropout": nn.Dropout(dropout),
            })
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        for block in self.blocks:
            residual = x
            x = block["norm"](x)
            x = block["mamba"](x)
            x = block["dropout"](x)
            x = x + residual
        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.head(x)


class MambaModel(ClassificationModel):
    def __init__(
        self,
        name,
        n_classes,
        freq,
        outputfolder,
        input_shape,
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
        num_workers=0,
        max_grad_norm=1.0,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "Official Mamba package not found. Install `mamba-ssm` in the training environment before running this model. "
                "Note: the legacy `ecg_env.yml` in this repository pins an older PyTorch stack and will likely need an updated environment for Mamba."
            )

        self.name = name
        self.n_classes = n_classes
        self.freq = freq
        self.input_shape = input_shape
        self.outputfolder = Path(outputfolder)
        self.outputfolder.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.outputfolder / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.models_dir / f"{self.name}.pt"

        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.max_grad_norm = max_grad_norm

        self.input_channels = input_shape[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _build_model(self):
        return MambaSequenceClassifier(
            input_channels=self.input_channels,
            num_classes=self.n_classes,
            d_model=self.d_model,
            n_layers=self.n_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dropout=self.dropout,
        )

    def _make_loader(self, X, y=None, shuffle=False):
        dataset = ECGDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def fit(self, X_train, y_train, X_val, y_val):
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.epochs))

        best_state = None
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_examples = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()

                batch_size = xb.size(0)
                train_loss += loss.item() * batch_size
                train_examples += batch_size

            self.model.eval()
            val_loss = 0.0
            val_examples = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    loss = self.loss_fn(logits, yb)

                    batch_size = xb.size(0)
                    val_loss += loss.item() * batch_size
                    val_examples += batch_size

            scheduler.step()

            train_loss = train_loss / max(1, train_examples)
            val_loss = val_loss / max(1, val_examples)
            print(f"[{self.name}] epoch {epoch + 1}/{self.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "model_state_dict": self.model.state_dict(),
                    "model_config": {
                        "input_channels": self.input_channels,
                        "num_classes": self.n_classes,
                        "d_model": self.d_model,
                        "n_layers": self.n_layers,
                        "d_state": self.d_state,
                        "d_conv": self.d_conv,
                        "expand": self.expand,
                        "dropout": self.dropout,
                    },
                }
                torch.save(best_state, self.checkpoint_path)

        if best_state is None:
            raise RuntimeError("Mamba training did not produce a checkpoint.")

        self.model.load_state_dict(best_state["model_state_dict"])

    def predict(self, X):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        loader = self._make_loader(X, y=None, shuffle=False)
        outputs = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                outputs.append(torch.sigmoid(logits).cpu().numpy())

        return np.concatenate(outputs, axis=0)
