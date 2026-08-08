import torch
from pathlib import Path

ckpt_dir = Path('model/checkpoints')
files_to_check = [
    'day32_finetuned_converged.pth',
    'day32_finetuned_latest.pth',
    'day32_epoch_06.pth',
    'day8_finetuned_best.pth',
]
for f in files_to_check:
    p = ckpt_dir / f
    if p.exists():
        c = torch.load(p, map_location='cpu')
        keys = list(c.keys())
        epoch = c.get('epoch', '?')
        val_acc = c.get('val_acc', '?')
        val_loss = c.get('val_loss', '?')
        early_stop = c.get('early_stopped', '?')
        print(f"{f}:")
        print(f"  epoch={epoch}, val_acc={val_acc}, val_loss={val_loss}, early_stopped={early_stop}")
        print(f"  keys={keys}")
        print()
