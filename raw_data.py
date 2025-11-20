from pathlib import Path
import shutil

# Configura esto:
ROOT = Path(r"C:\Users\danie\Documents\Codigos\Capstone\ReciclAI\raw_data")  # carpeta que contiene 'papel', 'carton', etc.
SUBFOLDERS_TO_COLLECT = {"default", "real_world"}  # carpetas fuentes
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MOVE_FILES = True     # True -> mover; False -> copiar
DRY_RUN = False       # True para ver qué haría sin tocar nada

def safe_move_or_copy(src: Path, dst: Path, move: bool):
    """Mueve o copia asegurando que no se pise un nombre existente."""
    final = dst
    if final.exists():
        stem, suf = final.stem, final.suffix
        i = 1
        while final.exists():
            final = final.with_name(f"{stem}__{i}{suf}")
            i += 1
    if DRY_RUN:
        action = "MOVE" if move else "COPY"
        print(f"[DRY] {action}: {src}  ->  {final}")
    else:
        final.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), str(final))
        else:
            shutil.copy2(str(src), str(final))

def consolidate_category(cat_dir: Path):
    """
    cat_dir: carpeta de categoría (p. ej. .../papel)
    Busca materiales (subcarpetas) y dentro junta imágenes de default/real_world
    hacia cat_dir.
    """
    if not cat_dir.is_dir():
        return

    # Recorremos materiales (subdirectorios inmediatos)
    for material_dir in [p for p in cat_dir.iterdir() if p.is_dir()]:
        # Dentro de cada material, buscamos las carpetas fuente (default/real_world)
        for src_sub in SUBFOLDERS_TO_COLLECT:
            src_dir = material_dir / src_sub
            if not src_dir.is_dir():
                continue

            # Tomamos todas las imágenes de forma no recursiva (o cambia a rglob si las anidas más)
            for img in src_dir.rglob("*"):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                    # Nombre destino con prefijo material + subcarpeta
                    new_name = f"{material_dir.name}_{src_sub}_{img.name}"
                    dst = cat_dir / new_name
                    safe_move_or_copy(img, dst, MOVE_FILES)

def consolidate_all(root: Path):
    # Recorre todas las categorías (subcarpetas a 1 nivel en ROOT)
    for cat_dir in [p for p in root.iterdir() if p.is_dir()]:
        print(f"\n=== Procesando categoría: {cat_dir.name} ===")
        consolidate_category(cat_dir)

if __name__ == "__main__":
    consolidate_all(ROOT)
    print("\nListo ✅")
