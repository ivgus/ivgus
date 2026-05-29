# prepare_nomeroff_dataset.py
import os
import json
import shutil
from pathlib import Path
import random
import glob

def load_all_json_annotations(annotations_dir):
    all_data = []
    json_files = glob.glob(os.path.join(annotations_dir, '**', '*.json'), recursive=True)

    print(f"Found {len(json_files)} JSON annotation files.")

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Если JSON содержит список, добавляем элементы списка
            if isinstance(data, list):
                all_data.extend(data)
            elif isinstance(data, dict):
                # Если одиночный объект, добавляем его как элемент списка
                all_data.append(data)
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

    return all_data


def prepare_nomeroff_data(annotations_dir, images_source_dir, output_dir, max_images=2000):
    out_path = Path(output_dir)
    img_dir = out_path / "images"
    lbl_dir = out_path / "labels"

    train_img_dir = img_dir / "train"
    train_lbl_dir = lbl_dir / "train"
    val_img_dir = img_dir / "val"
    val_lbl_dir = lbl_dir / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    data = load_all_json_annotations(annotations_dir)

    if not data:
        print("No annotations")
        return

    print(f"Total loaded annotations: {len(data)}")

    #Фильтрация и перемешивание
    random.shuffle(data)
    if max_images and len(data) > max_images:
        data = data[:max_images]

    print(f"Processing {len(data)} images...")

    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    def process_subset(subset, img_dest, lbl_dest, subset_name):
        count = 0
        skipped_no_img = 0
        skipped_no_text = 0

        for item in subset:
            filename = item.get('name') or item.get('filename') or item.get('img_name')

            if not filename:
                continue

            text = item.get('description')
            if not text:
                text = item.get('predicted')

            if not text and 'objects' in item and item['objects']:
                text = item['objects'][0].get('text') or item['objects'][0].get('number')

            if not text:
                skipped_no_text += 1
                continue

            #Очистка текста
            text = str(text).replace(" ", "").upper()

            #Оставляем только символы кириллица/латиница/цифры
            if len(text) < 4:
                continue

            src_img = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                path = Path(images_source_dir) / f"{filename}{ext}"
                if path.exists():
                    src_img = path
                    break

            if src_img is None:
                path = Path(images_source_dir) / filename
                if path.exists() and path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    src_img = path

            if src_img is None:
                skipped_no_img += 1
                continue

            safe_filename = filename.replace('/', '_').replace('\\', '_')
            dest_img_name = f"{safe_filename}.jpg"

            try:
                shutil.copy(src_img, img_dest / dest_img_name)
            except Exception as e:
                print(f"Copy error for {filename}: {e}")
                continue

            with open(lbl_dest / f"{safe_filename}.txt", 'w', encoding='utf-8') as f:
                f.write(text)

            count += 1

        print(
            f"[{subset_name}] Processed: {count}, Skipped (no img): {skipped_no_img}, Skipped (no text): {skipped_no_text}")

    process_subset(train_data, train_img_dir, train_lbl_dir, "Train")
    process_subset(val_data, val_img_dir, val_lbl_dir, "Val")

    print(f"Dataset prepared at {output_dir}")


if __name__ == "__main__":

    annotations_dir = 'archive\\autoriaNumberplateOcrRu-2021-09-01\\train\\ann'
    imgs_dir = 'archive\\autoriaNumberplateOcrRu-2021-09-01\\train\\img'
    out_dir = 'nomeroff_lpr_ready'

    if os.path.exists(annotations_dir) and os.path.exists(imgs_dir):
        prepare_nomeroff_data(annotations_dir, imgs_dir, out_dir, max_images=2000)
    else:
        print(f"Paths check failed:\nAnn: {os.path.exists(annotations_dir)}\nImg: {os.path.exists(imgs_dir)}")
        print("Check relative paths to your extracted archive.")
