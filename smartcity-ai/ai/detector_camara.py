"""Detección de objetos visibles para Cámara en vivo de SmartCity AI.

Este módulo no modifica el detector de videos grabados.
Recibe una imagen BGR y el modelo YOLO cargado por FastAPI.
"""

import cv2

CLASES_CAMARA = {
    0: ("personas", "Persona"),
    2: ("automoviles", "Automóvil"),
    3: ("motocicletas", "Motocicleta"),
    5: ("buses", "Bus"),
    7: ("camiones", "Camión"),
}


def detectar_camara(imagen, modo, modelo):
    """Devuelve detecciones y conteos del fotograma actual.

    Las cajas devueltas son [x1, y1, x2, y2], normalizadas entre 0 y 1.
    """
    if imagen is None or imagen.size == 0:
        raise ValueError("No se recibió un fotograma válido.")

    if modo == "personas":
        clases = [0]
        confianza_minima = 0.30
    elif modo == "vehiculos":
        clases = [2, 3, 5, 7]
        confianza_minima = 0.25
    else:
        raise ValueError("El modo debe ser personas o vehiculos.")

    alto, ancho = imagen.shape[:2]

    # Detección por fotograma, sin acumular detecciones previas.
    resultados = modelo.predict(
        source=imagen,
        classes=clases,
        conf=confianza_minima,
        imgsz=416,
        device="cpu",
        verbose=False,
    )

    conteos = {
        "personas": 0,
        "automoviles": 0,
        "motocicletas": 0,
        "buses": 0,
        "camiones": 0,
    }
    detecciones = []

    for caja in resultados[0].boxes:
        clase = int(caja.cls[0].item())
        if clase not in CLASES_CAMARA:
            continue

        clave, nombre = CLASES_CAMARA[clase]
        confianza = float(caja.conf[0].item())
        x1, y1, x2, y2 = map(float, caja.xyxy[0].tolist())

        # El navegador necesita coordenadas proporcionales al fotograma.
        coordenadas = [
            max(0.0, min(1.0, x1 / ancho)),
            max(0.0, min(1.0, y1 / alto)),
            max(0.0, min(1.0, x2 / ancho)),
            max(0.0, min(1.0, y2 / alto)),
        ]

        conteos[clave] += 1
        detecciones.append({
            "etiqueta": nombre,
            "confianza": round(confianza, 3),
            "caja": coordenadas,
        })

    return {
        "modo": modo,
        "conteos": conteos,
        "total": len(detecciones),
        "detecciones": detecciones,
    }
