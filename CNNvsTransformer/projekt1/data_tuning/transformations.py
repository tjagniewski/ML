import os 
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

def organize_pneumonia_data(source_dir: str):
    """
    Rozdziela zdjęcia z folderu PNEUMONIA na podkategorie: BACTERIA i VIRUS.
    """
    base_path = Path(source_dir)
    bacteria_path = base_path / "BACTERIA"
    virus_path = base_path / "VIRUS"

    # Tworzenie folderów docelowych (jeśli nie istnieją)
    bacteria_path.mkdir(exist_ok=True)
    virus_path.mkdir(exist_ok=True)

    # Liczniki dla podsumowania
    stats = {"bacteria": 0, "virus": 0, "unknown": 0}

    # Iteracja po plikach w folderze źródłowym
    # Używamy rglob dla rekurencyjnego wyszukiwania lub glob dla bieżącego folderu
    for file_path in Path(str(base_path / "PNEUMONIA")).glob("*"):
        if file_path.is_dir():
            continue
            
        filename = file_path.name.lower()

        # Logika segregacji oparta na nazewnictwie
        if "bacteria" in filename:
            shutil.move(str(file_path), str(bacteria_path / file_path.name))
            stats["bacteria"] += 1
        elif "virus" in filename:
            shutil.move(str(file_path), str(virus_path / file_path.name))
            stats["virus"] += 1
        else:
            stats["unknown"] += 1

    shutil.rmtree(str(base_path / "PNEUMONIA"))

    return stats

def split_dataset(root_dir: str, train_size: float = 0.9):
    root = Path(root_dir)
    classes = ['NORMAL','BACTERIA', 'VIRUS']
    
    # Tworzenie folderów train i val
    for split in ['train', 'val']:
        for cls in classes:
            (root / "transformed" / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        cls_path = root / "train"/ cls
        images = list(cls_path.glob('*'))
        
        # Split plików
        train_imgs, val_imgs = train_test_split(
            images, 
            train_size=train_size, 
            random_state=42, # Dla powtarzalności
            shuffle=True
        )

        # Przenoszenie plików do docelowych lokalizacji
        for img in train_imgs:
            shutil.move(str(img), str(root / "transformed" / "train" / cls / img.name))
        for img in val_imgs:
            shutil.move(str(img), str(root / "transformed" / "val" / cls / img.name))

def augment_and_save(root_dir: str, target_dir: str):
    """
    Dla każdego zdjęcia w root_dir generuje n_variants nowych obrazów i zapisuje w target_dir.
    """
    root = Path(root_dir)
    target = Path(target_dir)
    
    # Definicja zestawu transformacji (bez ToTensor i Normalize, bo zapisujemy do PIL)
    augmentation_pipeline = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomRotation(degrees=10),
    ])

    classes = ['NORMAL','BACTERIA', 'VIRUS']
    
    for cls in classes:
        source_path = root / cls
        dest_path = target / cls
        dest_path.mkdir(parents=True, exist_ok=True)
        
        images = list(source_path.glob('*.jpg')) + list(source_path.glob('*.jpeg'))

        random.seed(42)
        if cls == 'BACTERIA':
            n_variants = 750
        else:
            n_variants = 500
        random_idxs = random.sample(range(0, len(images)), n_variants)
        
        print(f"🔄 Generowanie danych dla klasy: {cls}")
        for img_id in tqdm(range(0, len(images))):
            img_path = images[img_id]
            # 1. Skopiuj oryginał do nowego folderu
            shutil.copy(img_path, dest_path / img_path.name)
            
            # 2. Wygeneruj warianty
            if img_id in random_idxs:
                with Image.open(img_path).convert('RGB') as img:
                    aug_img = augmentation_pipeline(img)

                    # Nowa nazwa pliku: nazwa_oryginalu_aug_1.jpg
                    new_name = f"{img_path.stem}_aug_{img_path.suffix}"
                    aug_img.save(dest_path / new_name, "JPEG")


BASE_PATH = "c:/Users/tjagn/Desktop/chestxray"

if __name__ == "__main__":
    try:
        train_results = organize_pneumonia_data(BASE_PATH + "/train")
        print(f"train: {train_results}")
        test_results = organize_pneumonia_data(BASE_PATH + "/test")
        print(f"test: {test_results}")

        split_dataset(BASE_PATH)
        print(f"train i val utworzone")

        augment_and_save(BASE_PATH + "/transformed/train", BASE_PATH + "/transformed/train/aug")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")