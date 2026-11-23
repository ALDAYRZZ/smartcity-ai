from pathlib import Path
import cv2
from ultralytics import YOLO

# =====================================
# SMARTCITY AI :DETECCION VEHICULAR
#seguridad para todos
# =====================================

# 1. Ubicación de los archivos
carpeta = Path(__file__).resolve().parent
imagen = carpeta / "prueba.jpg"
salida = carpeta / "resultado.jpg"

# 2. Comprobar que existe la imagen
if not imagen.exists():
    raise SystemExit("ERROR: No se encontro prueba.jpg")

foto = cv2.imread(str(imagen))

if foto is None:
    raise SystemExit("ERROR: No se pudo abrir la imagen")

# 3. Cargar modelo de inteligencia artificial
print("Iniciando SmartCity AI...")

modelo = YOLO("yolo11n.pt")

# 4. Detectar vehiculos
resultados = modelo.predict(
    source=str(imagen),
    classes=[2, 3, 5, 7],
    conf=0.20,
    imgsz=960,
    device="cpu",
    verbose=False
)

# 5. Categorias en español 
nombres = {
    2: "Automovil",
    3: "Motocicleta",
    5: "Bus",
    7: "Camion"
}

conteos = {nombre: 0 for nombre in nombres.values()}

# 6. Dibujar los vehiculos detectados
for objeto in resultados[0].boxes:

    clase = int(objeto.cls[0])
    confianza = float(objeto.conf[0])

    if clase not in nombres:
        continue

    x1, y1, x2, y2 = map(
        int, objeto.xyxy[0].tolist()
    )

    nombre = nombres[clase]
    conteos[nombre] += 1

    # Dibujar recuadro verde
    cv2.rectangle(
        foto,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Etiqueta con porcentaje de confianza
    etiqueta = f"{nombre} {confianza * 100:.0f}%"

    cv2.putText(
        foto,
        etiqueta,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

# 7. Guardar el resultado
if not cv2.imwrite(str(salida), foto):
    raise SystemExit("ERROR: No se pudo guardar el resultado")

# 8. Mostrar estadisticas
total = sum(conteos.values())

print("\n===== SMARTCITY AI =====")
print("Analisis terminado")

for nombre, cantidad in conteos.items():
    print(f"{nombre}: {cantidad}")

print("------------------------")
print(f"TOTAL DETECTADO: {total}")
print(f"Imagen guardada: {salida}")
