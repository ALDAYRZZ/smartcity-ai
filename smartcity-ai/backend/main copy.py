import subprocess
import imageio_ffmpeg
from pathlib import Path
import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from uuid import uuid4

from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from ai.detector_video import procesar_video
from ai.detector_camara import detectar_camara  # NUEVO: detección en vivo


# Rutas del proyecto
RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = RAIZ / "yolo11n.pt"

app = FastAPI(title="SmartCity AI")


# Mostrar las páginas web desde FastAPI
app.mount(
    "/frontend",
    StaticFiles(directory=str(RAIZ / "frontend"), html=True),
    name="frontend",
)


# Permitir que la página local se conecte al backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Cargar YOLO una sola vez, reutilizado por imagen y cámara
modelo = YOLO(str(RUTA_MODELO))

# Categorías reconocidas por el modelo base
CLASES = {
    2: ("automoviles", "Automovil"),
    3: ("motocicletas", "Motocicleta"),
    5: ("buses", "Bus"),
    7: ("camiones", "Camion"),
}

MAX_TAMANO = 10 * 1024 * 1024


@app.get("/")
def inicio():
    return {"mensaje": "SmartCity AI funcionando"}


@app.post("/analizar-imagen")
async def analizar_imagen(archivo: UploadFile = File(...)):

    # Validar formato
    formatos = ["image/jpeg", "image/png", "image/webp"]

    if archivo.content_type not in formatos:
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten imágenes JPG, PNG o WebP."
        )

    # Leer la imagen con límite de tamaño
    contenido = await archivo.read(MAX_TAMANO + 1)

    if len(contenido) > MAX_TAMANO:
        raise HTTPException(
            status_code=400,
            detail="La imagen supera los 10 MB."
        )

    datos = np.frombuffer(contenido, dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)

    if imagen is None:
        raise HTTPException(
            status_code=400,
            detail="No se pudo abrir la imagen."
        )

    # Analizar con YOLO
    resultados = modelo.predict(
        source=imagen,
        classes=[2, 3, 5, 7],
        conf=0.20,
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    conteos = {
        "automoviles": 0,
        "motocicletas": 0,
        "buses": 0,
        "camiones": 0,
    }

    # Dibujar detecciones y contar vehículos
    resultado = resultados[0]

    for caja in resultado.boxes:

        clase = int(caja.cls[0].item())
        confianza = float(caja.conf[0].item())

        if clase not in CLASES:
            continue

        clave, nombre = CLASES[clase]
        conteos[clave] += 1

        x1, y1, x2, y2 = map(
            int, caja.xyxy[0].tolist()
        )

        # Recuadro celeste
        cv2.rectangle(
            imagen,
            (x1, y1),
            (x2, y2),
            (255, 215, 0),
            2,
        )

        etiqueta = f"{nombre} {confianza:.0%}"
        posicion = (x1, max(y1 - 10, 20))

        # Contorno oscuro para facilitar lectura
        cv2.putText(
            imagen, etiqueta, posicion,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 0, 0), 3,
        )

        cv2.putText(
            imagen, etiqueta, posicion,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1,
        )

    total = sum(conteos.values())

    # Convertir el resultado a imagen JPG
    correcto, imagen_jpg = cv2.imencode(
        ".jpg",
        imagen,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )

    if not correcto:
        raise HTTPException(
            status_code=500,
            detail="No se pudo generar el resultado."
        )

    imagen_base64 = base64.b64encode(
        imagen_jpg.tobytes()
    ).decode("utf-8")

    return {
        "conteos": conteos,
        "total": total,
        "imagen_resultado": (
            "data:image/jpeg;base64," + imagen_base64
        ),
        "nota": (
            "Motokares y motocargueros requieren "
            "un modelo especializado."
        ),
    }


# =====================================
# SMARTCITY AI - ANALIZAR VIDEO
# =====================================

CARPETA_RESULTADOS = RAIZ / "backend" / "resultados"
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

app.mount(
    "/resultados",
    StaticFiles(directory=str(CARPETA_RESULTADOS)),
    name="resultados",
)


@app.post("/analizar-video")
async def analizar_video(archivo: UploadFile = File(...)):

    extension = Path(archivo.filename or "").suffix.lower()

    if extension not in [".mp4", ".webm"]:
        raise HTTPException(
            status_code=400,
            detail="Selecciona un video MP4 o WebM.",
        )

    # Límite de 30 MB
    contenido = await archivo.read(30 * 1024 * 1024 + 1)

    if len(contenido) > 30 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="El video supera los 30 MB.",
        )

    identificador = uuid4().hex

    entrada = (
        CARPETA_RESULTADOS /
        f"entrada_{identificador}{extension}"
    )

    salida = (
        CARPETA_RESULTADOS /
        f"smartcity_{identificador}.mp4"
    )

    entrada.write_bytes(contenido)

    # Convertir temporalmente a un formato compatible con OpenCV
    normalizado = (
        CARPETA_RESULTADOS /
        f"normalizado_{identificador}.avi"
    )

    try:
        print("Convirtiendo video para OpenCV...")

        await run_in_threadpool(
            subprocess.run,
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel", "error",
                "-i", str(entrada),
                "-t", "20",
                "-vf",
                "scale=960:960:force_original_aspect_ratio=decrease,fps=12",
                "-an",
                "-c:v", "mjpeg",
                "-q:v", "5",
                str(normalizado),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        print("Video convertido. Ejecutando YOLO...")

        resumen = await run_in_threadpool(
            procesar_video,
            normalizado,
            salida,
        )

        print("Analisis de video completado.")

    except subprocess.CalledProcessError as error:
        print("ERROR FFMPEG:", error.stderr)
        salida.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="No se pudo convertir el video.",
        ) from error

    except Exception as error:
        import traceback
        traceback.print_exc()

        salida.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=f"Error al analizar video: {error}",
        ) from error

    finally:
        entrada.unlink(missing_ok=True)
        normalizado.unlink(missing_ok=True)

    return {
        "mensaje": "Analisis de video terminado",
        "video_url": (
            f"http://127.0.0.1:8000/resultados/{salida.name}"
        ),
        **resumen,
    }


# =====================================
# SMARTCITY AI - CÁMARA EN VIVO CON YOLO
# =====================================

MAX_FOTOGRAMA = 3 * 1024 * 1024  # 3 MB por fotograma


@app.post("/analizar-camara")
async def analizar_camara(
    archivo: UploadFile = File(...),
    modo: str = Form("vehiculos"),
):
    """Devuelve detecciones en el fotograma actual de la cámara."""
    if modo not in ("personas", "vehiculos"):
        raise HTTPException(
            status_code=400,
            detail="Selecciona el modo personas o vehiculos.",
        )

    if archivo.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail="La cámara debe enviar una imagen JPG o PNG.",
        )

    contenido = await archivo.read(MAX_FOTOGRAMA + 1)
    if len(contenido) > MAX_FOTOGRAMA:
        raise HTTPException(
            status_code=413,
            detail="El fotograma supera los 3 MB.",
        )

    datos = np.frombuffer(contenido, dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if imagen is None:
        raise HTTPException(
            status_code=400,
            detail="No se pudo leer el fotograma de la cámara.",
        )

    # Procesar fuera del bucle principal de FastAPI.
    # Reutiliza el modelo que ya cargamos al arrancar el servidor.
    try:
        return await run_in_threadpool(
            detectar_camara,
            imagen,
            modo,
            modelo,
        )
    except Exception as error:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Error al procesar el fotograma con YOLO.",
        ) from error
