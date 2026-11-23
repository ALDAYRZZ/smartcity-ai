
"""SmartCity AI - Detector de infracciones de tránsito.

Modos:
- semaforo: luz roja simulada o reconocimiento experimental en video.
- peatonal: ocupación de una zona manual de cuatro puntos.

Genera videos procesados, imágenes de evidencia y archivos JSON.
Los eventos son experimentales, no infracciones confirmadas.
"""

import argparse
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

from ai.detector_video import filtrar_duplicados, relacionar


# =====================================================
# CONFIGURACIÓN
# =====================================================

RAIZ = Path(__file__).resolve().parent.parent

MODELO_BASE = RAIZ / "yolo11n.pt"

MODELO_LOCAL = (
    RAIZ
    / "runs"
    / "detect"
    / "entrenamientos"
    / "modelo_motokar"
    / "weights"
    / "best.pt"
)

ZONA_GUARDADA = RAIZ / "zona_peatonal.json"

SEGUNDO_ROJO = 5.5

NOMBRES = {
    2: "Automovil",
    3: "Motocicleta",
    5: "Bus",
    7: "Camion",
    80: "Motokar",
    81: "Motocarguero",
}


# =====================================================
# RECONOCIMIENTO EXPERIMENTAL DEL SEMÁFORO
# =====================================================

def reconocer_color_semaforo(imagen_bgr, caja):
    """Estima el color activo dentro de una caja de semáforo."""

    x1, y1, x2, y2 = map(int, caja)

    alto, ancho = imagen_bgr.shape[:2]

    x1 = max(0, min(ancho, x1))
    x2 = max(0, min(ancho, x2))
    y1 = max(0, min(alto, y1))
    y2 = max(0, min(alto, y2))

    if x2 - x1 < 7 or y2 - y1 < 15:
        return "desconocido"

    recorte = imagen_bgr[y1:y2, x1:x2]

    hsv = cv2.cvtColor(
        recorte,
        cv2.COLOR_BGR2HSV
    )

    h, sat, val = cv2.split(hsv)

    brillante = (sat >= 95) & (val >= 115)

    mascara_roja = brillante & (
        (h <= 11) | (h >= 170)
    )

    mascara_amarilla = brillante & (
        (h >= 15) & (h <= 39)
    )

    mascara_verde = brillante & (
        (h >= 40) & (h <= 95)
    )

    cantidades = {
        "rojo": int(mascara_roja.sum()),
        "amarillo": int(mascara_amarilla.sum()),
        "verde": int(mascara_verde.sum())
    }

    ganador = max(
        cantidades,
        key=cantidades.get
    )

    valores = sorted(
        cantidades.values(),
        reverse=True
    )

    primero = valores[0]
    segundo = valores[1]

    minimo = max(
        5,
        int(h.size * 0.025)
    )

    if primero < minimo:
        return "desconocido"

    if primero < segundo * 1.8:
        return "desconocido"

    mascaras = {
        "rojo": mascara_roja,
        "amarillo": mascara_amarilla,
        "verde": mascara_verde
    }

    mascara = mascaras[ganador]

    ys, _ = np.where(mascara)

    centro_relativo = (
        float(ys.mean())
        / max(1, recorte.shape[0])
    )

    # Comprobación para semáforos verticales convencionales.

    if ganador == "rojo" and centro_relativo > 0.55:
        return "desconocido"

    if ganador == "verde" and centro_relativo < 0.40:
        return "desconocido"

    return ganador


def detectar_estado_semaforo(imagen_bgr, cajas):
    """Obtiene el color de los semáforos detectados."""

    candidatos = []

    for caja, confianza in cajas:

        estado = reconocer_color_semaforo(
            imagen_bgr,
            caja
        )

        if estado != "desconocido":
            candidatos.append(
                (estado, float(confianza), caja)
            )

    if not candidatos:
        return "desconocido", None

    colores = {
        candidato[0]
        for candidato in candidatos
    }

    if len(colores) > 1:
        return "desconocido", None

    candidatos.sort(
        key=lambda elemento: elemento[1],
        reverse=True
    )

    mejor = candidatos[0]

    return mejor[0], mejor[2]


# =====================================================
# FUNCIONES DE VEHÍCULOS
# =====================================================

def pie(caja):
    """Calcula el punto inferior central del vehículo."""

    x1, _, x2, y2 = caja

    return (
        (x1 + x2) // 2,
        y2
    )


def proporcion_solapamiento(caja_a, caja_b):
    """Calcula el solapamiento respecto a la caja menor."""

    x1 = max(caja_a[0], caja_b[0])
    y1 = max(caja_a[1], caja_b[1])
    x2 = min(caja_a[2], caja_b[2])
    y2 = min(caja_a[3], caja_b[3])

    interseccion = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    area_a = (
        max(1, caja_a[2] - caja_a[0])
        * max(1, caja_a[3] - caja_a[1])
    )

    area_b = (
        max(1, caja_b[2] - caja_b[0])
        * max(1, caja_b[3] - caja_b[1])
    )

    return interseccion / min(area_a, area_b)


def seleccionar_detecciones(base, local):
    """Combina los modelos evitando cajas duplicadas."""

    elegidas = list(local)

    for caja, clase, confianza in base:

        duplicada = any(
            proporcion_solapamiento(
                caja,
                otra_caja
            ) >= 0.45
            for otra_caja, _, _ in local
        )

        if not duplicada:
            elegidas.append(
                (caja, clase, confianza)
            )

    return filtrar_duplicados(elegidas)


# =====================================================
# REGISTRO DE POSIBLES INFRACCIONES
# =====================================================

def registrar_eventos(
    identificador,
    caja,
    estado,
    rojo,
    linea_y,
    poligono,
    direccion,
    tiempo_segundos,
    clase,
    modo,
    semaforo_automatico=False
):
    """Registra eventos para cada vehículo rastreado."""

    x, y = pie(caja)

    datos = estado.setdefault(
        identificador,
        {
            "pie_anterior": None,
            "dentro_consecutivos": 0,
            "cruzo_rojo": False,
            "invadio": False
        }
    )

    anterior = datos["pie_anterior"]

    eventos = []

    nombre = NOMBRES.get(
        clase,
        "Vehiculo"
    )

    # -------------------------------------------------
    # POSIBLE CRUCE CON LUZ ROJA
    # -------------------------------------------------

    if (
        modo == "semaforo"
        and rojo
        and anterior is not None
        and not datos["cruzo_rojo"]
    ):

        previo_y = anterior[1]

        if direccion == "abajo":

            cruzo = (
                previo_y < linea_y <= y
            )

        else:

            cruzo = (
                previo_y > linea_y >= y
            )

        if cruzo:

            datos["cruzo_rojo"] = True

            tipo = (
                "Posible paso con luz roja "
                "(detectado en video)"
                if semaforo_automatico
                else "Paso con luz roja (simulado)"
            )

            eventos.append({
                "tipo": tipo,
                "vehiculo": nombre,
                "id_vehiculo": identificador,
                "tiempo_segundos": round(
                    tiempo_segundos,
                    2
                )
            })

    # -------------------------------------------------
    # POSIBLE OCUPACIÓN DEL PASO PEATONAL
    # -------------------------------------------------

    if modo == "peatonal":

        dentro = cv2.pointPolygonTest(
            poligono,
            (float(x), float(y)),
            False
        ) >= 0

        if dentro:

            datos["dentro_consecutivos"] += 1

        else:

            datos["dentro_consecutivos"] = 0

    if (
        modo == "peatonal"
        and datos["dentro_consecutivos"] >= 2
        and not datos["invadio"]
    ):

        datos["invadio"] = True

        eventos.append({
            "tipo": (
                "Posible ocupacion de paso peatonal "
                "(zona manual)"
            ),
            "vehiculo": nombre,
            "id_vehiculo": identificador,
            "tiempo_segundos": round(
                tiempo_segundos,
                2
            )
        })

    datos["pie_anterior"] = (x, y)

    return eventos


# =====================================================
# SELECCIÓN MANUAL DEL PASO PEATONAL
# =====================================================

def seleccionar_paso_peatonal(
    archivo_video,
    segundo=2.0
):
    """Permite marcar cuatro puntos mediante OpenCV."""

    ruta = Path(archivo_video).resolve()

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No encuentro el video: {ruta}"
        )

    cap = cv2.VideoCapture(
        str(ruta)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir el video."
        )

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        max(0.0, segundo) * 1000
    )

    ok, frame = cap.read()

    if not ok:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        ok, frame = cap.read()

    cap.release()

    if not ok:
        raise RuntimeError(
            "No se pudo leer una imagen del video."
        )

    alto_original, ancho_original = frame.shape[:2]

    escala = min(
        850 / ancho_original,
        750 / alto_original,
        1.0
    )

    ancho = max(
        1,
        round(ancho_original * escala)
    )

    alto = max(
        1,
        round(alto_original * escala)
    )

    pantalla = cv2.resize(
        frame,
        (ancho, alto)
    )

    puntos = []

    nombre = "MARCAR PASO PEATONAL - 4 CLICS"

    def clic(evento, x, y, flags, param):

        if (
            evento == cv2.EVENT_LBUTTONDOWN
            and len(puntos) < 4
        ):
            puntos.append((x, y))

    try:

        cv2.namedWindow(
            nombre,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            nombre,
            ancho,
            alto
        )

        cv2.setMouseCallback(
            nombre,
            clic
        )

        while True:

            vista = pantalla.copy()

            if len(puntos) >= 2:

                cv2.polylines(
                    vista,
                    [np.array(puntos, np.int32)],
                    False,
                    (255, 220, 30),
                    2
                )

            if len(puntos) == 4:

                cv2.polylines(
                    vista,
                    [np.array(puntos, np.int32)],
                    True,
                    (255, 220, 30),
                    3
                )

            for indice, (x, y) in enumerate(
                puntos,
                start=1
            ):

                cv2.circle(
                    vista,
                    (x, y),
                    5,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    vista,
                    str(indice),
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3
                )

            mensaje = (
                f"Puntos {len(puntos)}/4. "
                "ENTER guardar - R reiniciar - ESC cancelar"
            )

            posicion_texto = (
                5,
                max(20, alto - 15)
            )

            cv2.putText(
                vista,
                mensaje,
                posicion_texto,
                cv2.FONT_HERSHEY_SIMPLEX,
                min(0.6, ancho / 850),
                (0, 0, 0),
                3
            )

            cv2.putText(
                vista,
                mensaje,
                posicion_texto,
                cv2.FONT_HERSHEY_SIMPLEX,
                min(0.6, ancho / 850),
                (255, 255, 255),
                1
            )

            cv2.imshow(
                nombre,
                vista
            )

            tecla = cv2.waitKey(50) & 0xFF

            if (
                tecla in (13, 10)
                and len(puntos) == 4
            ):
                break

            if tecla in (
                ord("r"),
                ord("R")
            ):
                puntos.clear()

            if tecla == 27:

                print(
                    "Seleccion cancelada."
                )

                return

    finally:

        cv2.destroyAllWindows()

    relativos = [
        [
            round(x / ancho, 6),
            round(y / alto, 6)
        ]
        for x, y in puntos
    ]

    area = cv2.contourArea(
        np.array(
            puntos,
            dtype=np.float32
        )
    )

    if area < 20:
        raise ValueError(
            "El poligono es demasiado pequeno."
        )

    ZONA_GUARDADA.write_text(
        json.dumps(
            {
                "video": ruta.name,
                "puntos": relativos,
                "aviso": (
                    "Zona seleccionada manualmente "
                    "para demostracion."
                )
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "Zona guardada:",
        ZONA_GUARDADA
    )

    print(
        "Puntos:",
        relativos
    )


# =====================================================
# LEER ZONA PEATONAL GUARDADA
# =====================================================

def leer_zona(
    video,
    ancho_real,
    alto_real,
    izq,
    arriba
):

    if not ZONA_GUARDADA.is_file():
        raise FileNotFoundError(
            "Primero marca el paso peatonal."
        )

    zona = json.loads(
        ZONA_GUARDADA.read_text(
            encoding="utf-8"
        )
    )

    if zona.get("video") != Path(video).name:
        raise ValueError(
            "La zona guardada pertenece a otro video."
        )

    puntos = zona.get(
        "puntos",
        []
    )

    if len(puntos) != 4:
        raise ValueError(
            "La zona necesita cuatro puntos."
        )

    return np.array(
        [
            (
                int(x * ancho_real) + izq,
                int(y * alto_real) + arriba
            )
            for x, y in puntos
        ],
        dtype=np.int32
    )


# =====================================================
# PROCESAMIENTO PRINCIPAL
# =====================================================

def procesar_infracciones(
    entrada,
    salida,
    semaforo="rojo",
    linea=0.55,
    direccion="abajo",
    segundos=20,
    modo="peatonal",
    sin_modelo_local=False,
    puntos_zona=None
):
    """Procesa un video y genera evidencias."""

    entrada = Path(entrada).resolve()

    salida = Path(salida).resolve()

    # VALIDACIONES

    if not entrada.is_file():
        raise FileNotFoundError(
            f"No encuentro el video: {entrada}"
        )

    if not MODELO_BASE.is_file():
        raise FileNotFoundError(
            f"No encuentro el modelo: {MODELO_BASE}"
        )

    if not 0 <= linea <= 1:
        raise ValueError(
            "La linea debe estar entre 0 y 1."
        )

    if modo not in {
        "peatonal",
        "semaforo"
    }:
        raise ValueError(
            "Modo incorrecto."
        )

    if semaforo not in {
        "rojo",
        "verde",
        "automatico"
    }:
        raise ValueError(
            "Estado de semaforo incorrecto."
        )

    if direccion not in {
        "arriba",
        "abajo"
    }:
        raise ValueError(
            "Direccion incorrecta."
        )

    salida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    evidencias = (
        salida.parent
        / (salida.stem + "_evidencias")
    )

    # CARGAR MODELOS YOLO

    modelo_base = YOLO(
        str(MODELO_BASE)
    )

    modelo_local = (
        YOLO(str(MODELO_LOCAL))
        if MODELO_LOCAL.is_file()
        and not sin_modelo_local
        else None
    )

    print(
        "Modelo de motokar/motocarguero:",
        "ACTIVO"
        if modelo_local is not None
        else "NO ENCONTRADO"
    )

    print(
        "Modo:",
        modo.upper()
    )

    if modo == "semaforo":

        print(
            "Semaforo:",
            semaforo.upper(),
            "(DETECCION EN VIDEO)"
            if semaforo == "automatico"
            else "(SIMULADO)"
        )

    else:

        print(
            "Paso peatonal: zona manual."
        )

    # ABRIR VIDEO

    lector = imageio_ffmpeg.read_frames(
        str(entrada),
        pix_fmt="bgr24"
    )

    escritor = None

    fotogramas = 0
    analizados = 0
    fps = 12.0

    pistas = {}
    proximo_id = 1

    estado = {}
    eventos = []
    ultimas = []

    # ESTADO DEL SEMÁFORO

    estado_luz = "desconocido"
    candidato_luz = "desconocido"
    repeticiones_luz = 0
    caja_luz = None

    fotogramas_luz = {
        "rojo": 0,
        "verde": 0,
        "amarillo": 0,
        "desconocido": 0
    }

    try:

        info = next(lector)

        original_w, original_h = info["size"]

        fps = float(
            info.get("fps") or 12
        )

        if not np.isfinite(fps) or fps <= 0:
            fps = 12.0

        # REDIMENSIONAR

        escala = min(
            960 / original_w,
            960 / original_h,
            1.0
        )

        real_w = max(
            16,
            int(original_w * escala)
        )

        real_h = max(
            16,
            int(original_h * escala)
        )

        ancho = (
            (real_w + 15) // 16
        ) * 16

        alto = (
            (real_h + 15) // 16
        ) * 16

        izq = (
            ancho - real_w
        ) // 2

        arriba = (
            alto - real_h
        ) // 2

        der = (
            ancho - real_w - izq
        )

        abajo = (
            alto - real_h - arriba
        )

        linea_y = (
            int(linea * real_h)
            + arriba
        )

        # ZONA PEATONAL

        if modo == "peatonal":

            if puntos_zona is not None:

                poligono = np.array(
                    [
                        (
                            int(float(x) * real_w) + izq,
                            int(float(y) * real_h) + arriba
                        )
                        for x, y in puntos_zona
                    ],
                    dtype=np.int32
                )

            else:

                poligono = leer_zona(
                    entrada,
                    real_w,
                    real_h,
                    izq,
                    arriba
                )

        else:

            poligono = None

        # CREAR VIDEO DE SALIDA

        escritor = imageio_ffmpeg.write_frames(
            str(salida),
            (ancho, alto),
            pix_fmt_in="bgr24",
            pix_fmt_out="yuv420p",
            fps=fps,
            codec="libx264",
            output_params=[
                "-preset",
                "ultrafast",
                "-movflags",
                "+faststart"
            ]
        )

        escritor.send(None)

        max_fotogramas = min(
            int(fps * segundos),
            900
        )

        # =================================================
        # RECORRER LOS FOTOGRAMAS
        # =================================================

        for datos in lector:

            if fotogramas >= max_fotogramas:
                break

            imagen = np.frombuffer(
                datos,
                dtype=np.uint8
            ).reshape(
                (original_h, original_w, 3)
            ).copy()

            imagen = cv2.resize(
                imagen,
                (real_w, real_h)
            )

            imagen = cv2.copyMakeBorder(
                imagen,
                arriba,
                abajo,
                izq,
                der,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

            tiempo_actual = fotogramas / fps

            # MODO SIMULADO O AUTOMÁTICO

            if semaforo == "automatico":

                rojo = (
                    estado_luz == "rojo"
                )

            else:

                rojo = (
                    semaforo == "rojo"
                    and tiempo_actual >= SEGUNDO_ROJO
                )

            pendientes = []

            # DETECCIÓN YOLO CADA DOS FOTOGRAMAS

            if fotogramas % 2 == 0:

                clases = [2, 3, 5, 7]

                if (
                    modo == "semaforo"
                    and semaforo == "automatico"
                ):
                    clases.append(9)

                base_result = modelo_base.predict(
                    imagen,
                    classes=clases,
                    conf=0.25,
                    imgsz=416,
                    device="cpu",
                    verbose=False
                )[0]

                base = [
                    (
                        tuple(
                            map(
                                int,
                                caja.xyxy[0].tolist()
                            )
                        ),
                        int(caja.cls[0]),
                        float(caja.conf[0])
                    )
                    for caja in base_result.boxes
                ]

                # MODELO PERSONALIZADO

                local = []

                if modelo_local is not None:

                    local_result = modelo_local.predict(
                        imagen,
                        conf=0.25,
                        imgsz=416,
                        device="cpu",
                        verbose=False
                    )[0]

                    local = [
                        (
                            tuple(
                                map(
                                    int,
                                    caja.xyxy[0].tolist()
                                )
                            ),
                            80 + int(caja.cls[0]),
                            float(caja.conf[0])
                        )
                        for caja in local_result.boxes
                    ]

                # RECONOCER SEMÁFORO

                if (
                    modo == "semaforo"
                    and semaforo == "automatico"
                ):

                    luces = [
                        (caja, confianza)
                        for caja, clase, confianza in base
                        if clase == 9
                    ]

                    observado, caja_observada = (
                        detectar_estado_semaforo(
                            imagen,
                            luces
                        )
                    )

                    if (
                        observado == candidato_luz
                        and observado != "desconocido"
                    ):
                        repeticiones_luz += 1

                    else:

                        candidato_luz = observado

                        repeticiones_luz = (
                            1
                            if observado != "desconocido"
                            else 0
                        )

                    if repeticiones_luz >= 2:

                        estado_luz = observado

                    else:

                        estado_luz = "desconocido"

                    caja_luz = caja_observada

                    rojo = (
                        estado_luz == "rojo"
                    )

                    fotogramas_luz[estado_luz] += 1

                    # QUITAR SEMÁFOROS DE LAS DETECCIONES
                    # DE VEHÍCULOS

                    base = [
                        elemento
                        for elemento in base
                        if elemento[1] != 9
                    ]

                # COMBINAR MODELOS

                detecciones = seleccionar_detecciones(
                    base,
                    local
                )

                analizados += 1

                # SEGUIMIENTO DE VEHÍCULOS

                ids, proximo_id = relacionar(
                    detecciones,
                    pistas,
                    analizados,
                    proximo_id
                )

                ultimas = [
                    (*caja, clase, confianza, id_)
                    for (caja, clase, confianza), id_
                    in zip(detecciones, ids)
                ]

                # DETECTAR POSIBLES INFRACCIONES

                for caja, identificador in zip(
                    detecciones,
                    ids
                ):

                    coordenadas, clase, _ = caja

                    nuevos = registrar_eventos(
                        identificador,
                        coordenadas,
                        estado,
                        rojo,
                        linea_y,
                        poligono,
                        direccion,
                        tiempo_actual,
                        clase,
                        modo,
                        semaforo == "automatico"
                    )

                    for evento in nuevos:

                        evidencias.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        nombre = (
                            f"evento_{len(eventos) + 1:03d}.jpg"
                        )

                        ruta_captura = (
                            evidencias / nombre
                        )

                        evento["imagen"] = str(
                            ruta_captura
                        )

                        pendientes.append(
                            ruta_captura
                        )

                        eventos.append(
                            evento
                        )

                        print(
                            f"ALERTA: {evento['tipo']} | "
                            f"{evento['vehiculo']} "
                            f"ID {identificador}"
                        )

            # =================================================
            # DIBUJAR PASO PEATONAL O SEMÁFORO
            # =================================================

            if modo == "peatonal":

                cv2.polylines(
                    imagen,
                    [poligono],
                    True,
                    (255, 180, 30),
                    3
                )

                posicion = tuple(
                    poligono[
                        np.argmin(
                            poligono[:, 1]
                        )
                    ]
                )

                cv2.putText(
                    imagen,
                    "PASO PEATONAL (ZONA MANUAL)",
                    (
                        int(posicion[0]),
                        max(
                            43,
                            int(posicion[1]) - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 180, 30),
                    2
                )

                cv2.rectangle(
                    imagen,
                    (0, 0),
                    (ancho, 33),
                    (24, 24, 24),
                    -1
                )

                cv2.putText(
                    imagen,
                    "MODO: PASO PEATONAL (DEMO)",
                    (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 220, 30),
                    2
                )

            else:

                # LÍNEA DE DETENCIÓN

                cv2.line(
                    imagen,
                    (izq, linea_y),
                    (izq + real_w, linea_y),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    imagen,
                    "LINEA DE DETENCION",
                    (
                        izq + 5,
                        max(
                            40,
                            linea_y - 12
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

                # COLOR DEL INDICADOR

                color = (
                    (0, 0, 255)
                    if rojo
                    else (0, 220, 0)
                )

                cv2.rectangle(
                    imagen,
                    (0, 0),
                    (ancho, 33),
                    (24, 24, 24),
                    -1
                )

                # TEXTO DEL SEMÁFORO

                if semaforo == "automatico":

                    texto_semaforo = (
                        "SEMAFORO: "
                        + estado_luz.upper()
                        + " (VIDEO)"
                    )

                    if caja_luz is not None:

                        lx1, ly1, lx2, ly2 = map(
                            int,
                            caja_luz
                        )

                        cv2.rectangle(
                            imagen,
                            (lx1, ly1),
                            (lx2, ly2),
                            color,
                            2
                        )

                else:

                    texto_semaforo = (
                        "SEMAFORO: ROJO (SIMULADO)"
                        if rojo
                        else "SEMAFORO: ANTES DEL ROJO (SIMULADO)"
                    )

                cv2.putText(
                    imagen,
                    texto_semaforo,
                    (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

            # =================================================
            # DIBUJAR VEHÍCULOS
            # =================================================

            for (
                x1, y1, x2, y2,
                clase, confianza, identificador
            ) in ultimas:

                color = (
                    50,
                    240,
                    80
                )

                cv2.rectangle(
                    imagen,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                nombre = NOMBRES.get(
                    clase,
                    "Vehiculo"
                )

                etiqueta = (
                    f"{nombre} ID {identificador} "
                    f"{confianza:.0%}"
                )

                cv2.putText(
                    imagen,
                    etiqueta,
                    (
                        x1,
                        max(45, y1 - 6)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                cv2.circle(
                    imagen,
                    pie(
                        (x1, y1, x2, y2)
                    ),
                    4,
                    (255, 255, 255),
                    -1
                )

            # =================================================
            # CONTADOR DE ALERTAS
            # =================================================

            if eventos:

                cv2.putText(
                    imagen,
                    f"ALERTAS (DEMO): {len(eventos)}",
                    (10, alto - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # GUARDAR IMÁGENES DE EVIDENCIA

            for ruta_captura in pendientes:

                cv2.imwrite(
                    str(ruta_captura),
                    imagen
                )

            # GUARDAR FOTOGRAMA EN VIDEO

            escritor.send(
                imagen.tobytes()
            )

            fotogramas += 1

    finally:

        lector.close()

        if escritor is not None:
            escritor.close()

    if fotogramas == 0:
        raise RuntimeError(
            "El video no contenia fotogramas validos."
        )

    # =====================================================
    # GENERAR RESUMEN
    # =====================================================

    resumen = {
        "modo": modo,

        "aviso": (
            "Prototipo experimental: no confirma "
            "infracciones legales. "
            "Semaforo simulado o estimado visualmente; "
            "zona peatonal manual."
        ),

        "semaforo": semaforo,

        "lecturas_semaforo": (
            fotogramas_luz
            if semaforo == "automatico"
            else None
        ),

        "aviso_semaforo": (
            "La identificacion visual del semaforo "
            "es experimental. Si no se ve claramente, "
            "no se registran cruces en rojo."
            if semaforo == "automatico"
            else "Estado simulado desde el segundo 5.5."
        ),

        "fotogramas": fotogramas,

        "fotogramas_analizados": analizados,

        "duracion_segundos": round(
            fotogramas / fps,
            2
        ),

        "total_eventos": len(eventos),

        "eventos": eventos
    }

    ruta_json = salida.with_suffix(
        ".json"
    )

    ruta_json.write_text(
        json.dumps(
            resumen,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "Video guardado:",
        salida
    )

    print(
        "Resumen guardado:",
        ruta_json
    )

    print(
        "Eventos detectados:",
        len(eventos)
    )

    return resumen


# =====================================================
# EJECUCIÓN DESDE TERMINAL
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "SmartCity AI - Detector de infracciones"
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Video de entrada"
    )

    parser.add_argument(
        "--modo",
        choices=[
            "peatonal",
            "semaforo"
        ],
        required=True
    )

    parser.add_argument(
        "--salida",
        default="resultado_infracciones_v2.mp4"
    )

    parser.add_argument(
        "--elegir-zona",
        action="store_true",
        help="Seleccionar zona peatonal"
    )

    parser.add_argument(
        "--segundo",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--semaforo",
        choices=[
            "rojo",
            "verde",
            "automatico"
        ],
        default="rojo"
    )

    parser.add_argument(
        "--linea",
        type=float,
        default=0.78
    )

    parser.add_argument(
        "--direccion",
        choices=[
            "arriba",
            "abajo"
        ],
        default="arriba"
    )

    parser.add_argument(
        "--segundos",
        type=float,
        default=20.0
    )

    parser.add_argument(
        "--sin-modelo-local",
        action="store_true",
        help="Desactivar modelo personalizado"
    )

    args = parser.parse_args()

    # SELECCIONAR ZONA PEATONAL

    if args.elegir_zona:

        if args.modo != "peatonal":

            parser.error(
                "--elegir-zona solo funciona "
                "en modo peatonal."
            )

        seleccionar_paso_peatonal(
            args.video,
            args.segundo
        )

        return

    # PROCESAR VIDEO

    procesar_infracciones(
        args.video,
        args.salida,
        args.semaforo,
        args.linea,
        args.direccion,
        args.segundos,
        args.modo,
        args.sin_modelo_local
    )


if __name__ == "__main__":
    main()
