# test_recognizer_only.py
import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from lprnet_pruned import LPRNetPrunable
import glob

RU_CHARS = "0123456789АВЕКМНОРСТУХ"


class RecognizerTester:
    def __init__(self, recognizer_weights='models\\lprnet_nomeroff_pruned.pth',
                 base_dir='archive\\autoriaNumberplateOcrRu-2021-09-01\\test'):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.test_img_dir = os.path.join(base_dir, 'img')
        self.test_ann_dir = os.path.join(base_dir, 'ann')

        print(f"Testing Recognizer ONLY on cropped plates from: {self.test_img_dir}")

        print("Loading Pruned LPRNet")
        self.num_classes = len(RU_CHARS) + 1
        self.recognizer = LPRNetPrunable(num_classes=self.num_classes)
        self.recognizer.load_state_dict(torch.load(recognizer_weights, map_location=self.device))
        self.recognizer.to(self.device)
        self.recognizer.eval()

        self.idx_to_char = {i: c for i, c in enumerate(RU_CHARS)}

        print("Loading Ground Truth")
        self.gt_data = self._load_ground_truth()

    def _load_ground_truth(self):
        gt_map = {}
        json_files = glob.glob(os.path.join(self.test_ann_dir, '**', '*.json'), recursive=True)

        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                items = []
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = [data]

                for item in items:
                    filename = item.get('name') or item.get('filename')
                    if not filename: continue

                    text = item.get('description')
                    if not text: text = item.get('predicted')

                    if text:
                        clean_text = str(text).replace(" ", "").upper()
                        valid_text = "".join([c for c in clean_text if c in RU_CHARS])
                        if valid_text:
                            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                                filename_key = filename + '.jpg'
                            else:
                                filename_key = filename

                            gt_map[filename_key] = valid_text
            except Exception as e:
                pass

        print(f"Loaded {len(gt_map)} ground truth labels.")
        return gt_map

    def preprocess_plate(self, plate_img):
        #Предобработка для LPRNet
        if plate_img is None or plate_img.size == 0:
            return None
        resized = cv2.resize(plate_img, (94, 24))
        resized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def decode_ctc(self, preds):
        if preds is None: return ""
        preds = preds.cpu().detach().numpy()
        pred_indices = np.argmax(preds, axis=2)

        prev_char = -1
        result_chars = []
        blank_idx = len(RU_CHARS)

        for idx in pred_indices[0]:
            if idx != blank_idx and idx != prev_char:
                if idx in self.idx_to_char:
                    result_chars.append(self.idx_to_char[idx])
            prev_char = idx

        return "".join(result_chars)

    def run_test(self):
        print("Starting Recognizer Test")

        all_files = os.listdir(self.test_img_dir)
        test_images = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Found {len(test_images)} images.")

        total_samples = 0
        correct_full_match = 0
        correct_char_count = 0
        total_char_count = 0

        errors = []

        for img_name in test_images:
            # Поиск GT
            gt_text = self.gt_data.get(img_name)
            if gt_text is None:
                name_no_ext = os.path.splitext(img_name)[0]
                gt_text = self.gt_data.get(name_no_ext + '.jpg')

            if not gt_text:
                continue

            total_samples += 1
            img_path = os.path.join(self.test_img_dir, img_name)
            frame = cv2.imread(img_path)

            if frame is None: continue

            input_tensor = self.preprocess_plate(frame)

            predicted_text = ""
            if input_tensor is not None:
                with torch.no_grad():
                    preds = self.recognizer(input_tensor)
                predicted_text = self.decode_ctc(preds)

            # Метрики
            min_len = min(len(gt_text), len(predicted_text))
            matches = sum(1 for i in range(min_len) if gt_text[i] == predicted_text[i])

            total_char_count += len(gt_text)
            correct_char_count += matches

            if gt_text == predicted_text:
                correct_full_match += 1
            else:
                errors.append({
                    'file': img_name,
                    'gt': gt_text,
                    'pred': predicted_text
                })

        if total_samples == 0:
            print("No samples processed.")
            return

        full_acc = correct_full_match / total_samples
        char_acc = correct_char_count / total_char_count if total_char_count > 0 else 0

        print("\n" + "=" * 40)
        print("RECOGNIZER TEST RESULTS (Pruned LPRNet)")
        print("=" * 40)
        print(f"Total Samples:      {total_samples}")
        print(f"Full Accuracy:      {full_acc:.4f} ({correct_full_match}/{total_samples})")
        print(f"Character Accuracy: {char_acc:.4f}")
        print("-" * 40)

        if errors:
            print("Sample Errors (First 10):")
            for err in errors[:10]:
                print(f"File: {err['file']}")
                print(f"  GT:   {err['gt']}")
                print(f"  Pred: {err['pred']}")
                print("-" * 10)

        # Сохранение ошибок
        with open('recognizer_errors.json', 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print("Errors saved to recognizer_errors.json")


if __name__ == "__main__":
    tester = RecognizerTester()
    tester.run_test()
