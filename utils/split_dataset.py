import os
import shutil
import random
from tqdm import tqdm

def split_dataset(
    raw_data_dir='../data/raw/flowers',
    output_dir='../data/processed',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    random.seed(seed)

    # створюємо цільові папки
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_dir, split)
        os.makedirs(split_path, exist_ok=True)

    # беремо всі класи
    classes = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]

    print("Знайдені класи:", classes)

    for cls in classes:
        print(f"\nОбробляється клас: {cls}")

        class_dir = os.path.join(raw_data_dir, cls)
        images = os.listdir(class_dir)
        images = [img for img in images if img.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # перемішуємо
        random.shuffle(images)

        total = len(images)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        # розподіляємо
        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        # створюємо папки для класу
        for split_name, files in splits.items():
            split_class_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_class_dir, exist_ok=True)

            # копіюємо файли
            for img in tqdm(files, desc=f"{split_name}"):
                src = os.path.join(class_dir, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy(src, dst)

    print("\nДатасет підготовлено")
