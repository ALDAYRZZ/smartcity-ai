
from pathlib import Path
from collections import defaultdict
import random
import shutil

# Carpetas del proyecto
PROYECTO = Path(__file__).resolve().parent
DATASET = PROYECTO / "dataset"
ORIGINALES = DATASET / "imagenes_originales"
EXPORTADAS = DATASET / "etiquetas_exportadas"

FORMATOS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CATEGORIAS = ["motokar", "motocarguero"]

# 1. Buscar todas las fotografias
fotos = defaultdict(list)

for categoria in CATEGORIAS:
    carpeta = ORIGINALES / categoria

    for archivo in carpeta.rglob("*"):
        if archivo.is_file() and archivo.suffix.lower() in FORMATOS:
            fotos[archivo.stem.lower()].append((archivo, categoria))

# 2. Buscar todas las etiquetas exportadas
etiquetas = defaultdict(list)

for archivo in EXPORTADAS.rglob("*.txt"):
    etiquetas[archivo.stem.lower()].append(archivo)

# 3. Comprobar que los nombres no sean ambiguos
errores = []

for nombre, archivos in fotos.items():
    if nombre not in etiquetas:
        continue

    if len(archivos) > 1:
        errores.append(
            f"Varias fotografias tienen el nombre: {nombre}"
        )

    if len(etiquetas[nombre]) > 1:
        errores.append(
            f"Hay varias etiquetas para: {nombre}"
        )

if errores:
    print("\nATENCION: se encontraron nombres repetidos.")
    for error in errores:
        print("-", error)

    print("\nNo se copio nada. Hay que revisar estos nombres.")
    raise SystemExit(1)

# 4. Revisar las fotografias que ya colocamos
ubicaciones = {}

for grupo in ["train", "val"]:
    carpeta_fotos = DATASET / "images" / grupo
    carpeta_labels = DATASET / "labels" / grupo

    carpeta_fotos.mkdir(parents=True, exist_ok=True)
    carpeta_labels.mkdir(parents=True, exist_ok=True)

    for foto in carpeta_fotos.iterdir():
        if foto.suffix.lower() not in FORMATOS:
            continue

        nombre = foto.stem.lower()
        etiqueta = carpeta_labels / (foto.stem + ".txt")

        if not etiqueta.exists():
            errores.append(f"Falta etiqueta para {foto.name}")

        if nombre in ubicaciones:
            errores.append(f"Imagen repetida en train y val: {nombre}")

        ubicaciones[nombre] = grupo

if errores:
    for error in errores:
        print("ERROR:", error)
    raise SystemExit(1)

# 5. Agrupar fotografias con sus etiquetas
pares = defaultdict(list)
sin_etiqueta = []

for nombre, lista in fotos.items():
    if nombre not in etiquetas:
        sin_etiqueta.append(nombre)
        continue

    foto, categoria = lista[0]
    etiqueta = etiquetas[nombre][0]

    contenido = etiqueta.read_text(encoding="utf-8-sig").strip()

    if not contenido:
        errores.append(f"Etiqueta vacia: {etiqueta.name}")
        continue

    for linea in contenido.splitlines():
        datos = linea.split()

        try:
            if len(datos) != 5:
                raise ValueError()

            clase = int(datos[0])
            valores = [float(x) for x in datos[1:]]

            if clase not in [0, 1]:
                raise ValueError()

            if not all(0 <= x <= 1 for x in valores):
                raise ValueError()

            if valores[2] <= 0 or valores[3] <= 0:
                raise ValueError()

        except ValueError:
            errores.append(f"Etiqueta incorrecta: {etiqueta.name}")
            break

    pares[categoria].append((foto, etiqueta))

if errores:
    for error in errores:
        print("ERROR:", error)
    raise SystemExit(1)

# 6. Distribuir el 80% en train y el 20% en val
azar = random.Random(42)

for categoria in CATEGORIAS:
    elementos = pares[categoria]
    azar.shuffle(elementos)

    total = len(elementos)
    objetivo_val = max(1, round(total * 0.20)) if total >= 2 else 0

    actuales_val = sum(
        1 for foto, _ in elementos
        if ubicaciones.get(foto.stem.lower()) == "val"
    )

    for foto, etiqueta in elementos:
        nombre = foto.stem.lower()

        # Respetar las fotografias que ya copiaste
        if nombre in ubicaciones:
            continue

        grupo = "val" if actuales_val < objetivo_val else "train"

        destino_foto = DATASET / "images" / grupo / foto.name
        destino_txt = DATASET / "labels" / grupo / (foto.stem + ".txt")

        shutil.copy2(foto, destino_foto)
        shutil.copy2(etiqueta, destino_txt)

        ubicaciones[nombre] = grupo

        if grupo == "val":
            actuales_val += 1

# 7. Mostrar resultados
print("\nDATASET ORGANIZADO CORRECTAMENTE")

for grupo in ["train", "val"]:
    carpeta = DATASET / "images" / grupo
    cantidad = sum(
        1 for f in carpeta.iterdir()
        if f.suffix.lower() in FORMATOS
    )
    print(f"Imagenes en {grupo}: {cantidad}")

print(f"Fotografias sin etiqueta, no utilizadas: {len(sin_etiqueta)}")
print("\nLas fotografias originales no fueron modificadas.")
