# train_lprnet_nomeroff.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lprnet_pruned import LPRNetPrunable
import os
import cv2
import numpy as np

RU_CHARS = "0123456789АВЕКМНОРСТУХ"

def collate_fn(batch):
    # Для обработки последовательностей разной длины
    batch = [b for b in batch if b[1].numel() > 0]

    if not batch:
        return None

    imgs, targets, target_lens = zip(*batch)

    imgs = torch.stack(imgs, 0)

    # Для CTC Loss targets должны быть сконкатенированы в один длинный вектор
    targets = torch.cat(targets, 0)
    target_lens = torch.tensor(target_lens, dtype=torch.long)

    return imgs, targets, target_lens


class NomeroffDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=(94, 24)):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.img_size = img_size
        self.files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

        self.char_to_idx = {c: i for i, c in enumerate(RU_CHARS)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        lbl_path = os.path.join(self.lbl_dir, fname.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros((3, *self.img_size)), torch.zeros(0, dtype=torch.int32), 0

        # LPRNet ожидает фиксированный размер входа
        img = cv2.resize(img, self.img_size)

        # Нормализация [0, 1]
        img = img.astype(np.float32) / 255.0

        # Из [H, W, C] в [C, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1)

        if not os.path.exists(lbl_path):
            return torch.zeros((3, *self.img_size)), torch.zeros(0, dtype=torch.int32), 0

        with open(lbl_path, 'r', encoding='utf-8') as f:
            text = f.read().strip().upper()

        # Фильтрация
        valid_text = "".join([c for c in text if c in self.char_to_idx])

        if not valid_text:
            return torch.zeros((3, *self.img_size)), torch.zeros(0, dtype=torch.int32), 0

        targets = torch.IntTensor([self.char_to_idx[c] for c in valid_text])
        return img, targets, len(valid_text)


def train_nomeroff_lpr():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = 'nomeroff_lpr_ready'

    if not os.path.exists(data_path):
        print("Run prepare_nomeroff_dataset.py first!")
        return

    train_dataset = NomeroffDataset(
        os.path.join(data_path, 'images', 'train'),
        os.path.join(data_path, 'labels', 'train')
    )

    loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_fn)

    num_classes = len(RU_CHARS) + 1
    model = LPRNetPrunable(num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ctc_loss = nn.CTCLoss(blank=len(RU_CHARS), reduction='mean')

    print("Training")
    epochs_pre = 15
    for epoch in range(epochs_pre):
        model.train()
        total_loss = 0
        batch_count = 0
        for batch in loader:
            if batch is None: continue
            imgs, targets, target_lens = batch

            if targets.numel() == 0: continue

            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)

            input_lengths = torch.full(size=(imgs.size(0),), fill_value=preds.size(1), dtype=torch.long)

            log_probs = preds.log_softmax(2).permute(1, 0, 2)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            print(f"Epoch [{epoch + 1}/{epochs_pre}], Loss: {total_loss / batch_count:.4f}")

    # Pruning (30%)
    print("Pruning")
    model.apply_pruning(amount=0.3)

    # Fine-tuning
    print("Fine-tuning pruned model")
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs_ft = 5
    for epoch in range(epochs_ft):
        model.train()
        total_loss = 0
        batch_count = 0
        for batch in loader:
            if batch is None: continue
            imgs, targets, target_lens = batch

            if targets.numel() == 0: continue

            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)
            input_lengths = torch.full(size=(imgs.size(0),), fill_value=preds.size(1), dtype=torch.long)

            log_probs = preds.log_softmax(2).permute(1, 0, 2)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lens)

            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()
            total_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            print(f"Fine-tune Epoch [{epoch + 1}/{epochs_ft}], Loss: {total_loss / batch_count:.4f}")

    model.remove_pruning_reparametrization()

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/lprnet_nomeroff_pruned.pth')
    print("Model saved")


if __name__ == "__main__":
    train_nomeroff_lpr()
