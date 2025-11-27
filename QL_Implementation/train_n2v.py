from caremics.config import load_config
from caremics.models import build_model
from caremics.data import build_dataset
from caremics.engine import Trainer
from torch.utils.data import DataLoader


def main():
    # 1. Load yaml config
    cfg = load_config("configs/n2v_unet.yaml")

    # 2. Build model
    model = build_model(cfg.model)

    # 3. Build datasets
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)

    # 4. Build loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # 5. Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg.trainer
    )

    # 6. Start training
    trainer.fit()


if __name__ == "__main__":
    main()
