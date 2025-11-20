from pathlib import Path
import random
import shutil
from collections import defaultdict

# ====== CONFIGURA AQUÍ ======
DATA_ROOT = Path(r"C:\Users\danie\Documents\Codigos\Capstone\ReciclAI\data")  # contiene carpetas: papel, carton, etc.
OUTPUT_ROOT = DATA_ROOT                   # si quieres crear /train y /val dentro del mismo root
VAL_RATIO = 0.2                           # 20% para validación
SEED = 42
MOVE_FILES = False                        # True = mover; False = copiar
DRY_RUN = False                           # True = no toca nada, solo muestra
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Si tus archivos quedaron como "material_subcarpeta_nombre.png",
# activa esto para estratificar por el prefijo (material_subcarpeta)
STRATIFY_BY_PREFIX = True
PREFIX_SEP = "_"   # separador para detectar prefijo
# ===========================

random.seed(SEED)

def list_categories(root: Path):
    return [p for p in root.iterdir() if p.is_dir() and p.name not in {"train", "val"}]

def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def ensure_dir(path: Path):
    if not DRY_RUN:
        path.mkdir(parents=True, exist_ok=True)

def move_or_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if DRY_RUN:
        print(f"[DRY] {'MOVE' if MOVE_FILES else 'COPY'} {src} -> {dst}")
        return
    if MOVE_FILES:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def split_indices(n, val_ratio):
    k_val = max(1, int(round(n * val_ratio))) if n > 1 else int(n > 0)
    idxs = list(range(n))
    random.shuffle(idxs)
    return set(idxs[:k_val])  # índices para val

def get_prefix(fname: str):
    """
    Devuelve el prefijo para estratificar.
    Ej.: 'magazines_real_world_Image_1.png' -> 'magazines_real_world'
    """
    if not STRATIFY_BY_PREFIX:
        return "__all__"
    parts = fname.split(PREFIX_SEP)
    # usa los dos primeros trozos para mayor estabilidad (material + subcarpeta)
    if len(parts) >= 2:
        return PREFIX_SEP.join(parts[:2])
    return parts[0] if parts else "__all__"

def split_category(cat_dir: Path, out_root: Path):
    cat_name = cat_dir.name
    imgs = list_images(cat_dir)
    if not imgs:
        print(f"- {cat_name}: sin imágenes, se omite.")
        return

    # Agrupamos por prefijo (estratificado por material_subcarpeta)
    groups = defaultdict(list)
    for img in imgs:
        groups[get_prefix(img.name)].append(img)

    # Directorios destino
    train_dir = out_root / "train" / cat_name
    val_dir   = out_root / "val"   / cat_name
    ensure_dir(train_dir)
    ensure_dir(val_dir)

    moved, copied = 0, 0
    for gname, files in groups.items():
        files_sorted = sorted(files, key=lambda p: p.name)  # orden estable
        val_idxs = split_indices(len(files_sorted), VAL_RATIO)

        for i, src in enumerate(files_sorted):
            dst_base = val_dir if i in val_idxs else train_dir
            dst = dst_base / src.name
            move_or_copy(src, dst)
            if not DRY_RUN:
                if MOVE_FILES: moved += 1
                else: copied += 1

    total = len(imgs)
    nval = sum(1 for _ in (val_dir.iterdir() if val_dir.exists() else []))
    print(f"- {cat_name}: {total} imgs -> train/val con ratio ~{VAL_RATIO:.2f}.")
    if not DRY_RUN:
        print(f"  Acción: {'MOVE' if MOVE_FILES else 'COPY'}; procesados: {moved or copied}")

def main():
    categories = list_categories(DATA_ROOT)
    if not categories:
        print("No se encontraron categorías.")
        return

    print(f"Categorías detectadas: {[c.name for c in categories]}")
    for cat in categories:
        split_category(cat, OUTPUT_ROOT)

    print("\nListo ✅ Estructura final (ejemplo):")
    print(str(OUTPUT_ROOT / "train" / "<categoria>" / "<archivo>"))
    print(str(OUTPUT_ROOT / "val"   / "<categoria>" / "<archivo>"))

if __name__ == "__main__":
    main()
