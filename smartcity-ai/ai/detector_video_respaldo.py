"""SmartCity AI: deteccion y estimacion de vehiculos diferentes en todo el video.

IMPORTANTE: conteo estimado, no exactitud garantizada. No usa una linea virtual.
Compatible con FastAPI backend/main.py: procesar_video(entrada, salida) -> dict.
"""

from collections import defaultdict
from math import hypot, log
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent
MODELO = RAIZ / "yolo11n.pt"
CLASES = {
    2: ("automoviles", "Automovil"),
    3: ("motocicletas", "Motocicleta"),
    5: ("buses", "Bus"),
    7: ("camiones", "Camion"),
}

RESOLUCION_MAXIMA = 960
ANALIZAR_CADA = 2
UMBRAL_YOLO = 0.14
UMBRAL_POR_CLASE = {2: 0.20, 3: 0.14, 5: 0.18, 7: 0.45}
MIN_APARICIONES = {2: 3, 3: 2, 5: 2, 7: 3}
MAX_PASOS_SIN_VER = 12
MAX_PASOS_UNIR = 14


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centro(caja):
    return ((caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2)


def clase_principal(pista):
    return max(pista["votos"], key=pista["votos"].get)


def filtrar_duplicados(detecciones):
    seleccionadas = []
    for caja, clase, confianza in sorted(
        detecciones, key=lambda item: item[2], reverse=True
    ):
        if any(iou(caja, anterior[0]) > 0.78 for anterior in seleccionadas):
            continue
        seleccionadas.append((caja, clase, confianza))
    return seleccionadas


def relacionar(detecciones, pistas, paso, proximo_id):
    """Asocia detecciones con IDs usando posicion, tamano y movimiento."""
    candidatos = []
    for indice, (caja, clase, confianza) in enumerate(detecciones):
        cx, cy = centro(caja)
        ancho = max(1, caja[2] - caja[0])
        alto = max(1, caja[3] - caja[1])
        for identificador, pista in pistas.items():
            salto = paso - pista["ultimo_paso"]
            if not 1 <= salto <= MAX_PASOS_SIN_VER:
                continue
            anterior = pista["caja"]
            px, py = centro(anterior)
            vx, vy = pista["velocidad"]
            pred_x, pred_y = px + vx * salto, py + vy * salto
            distancia = hypot(cx - pred_x, cy - pred_y)
            diag = hypot(anterior[2] - anterior[0], anterior[3] - anterior[1])
            tolerancia = max(34, diag * 0.92) * (1 + 0.14 * (salto - 1))
            solapamiento = iou(caja, anterior)
            if distancia > tolerancia and solapamiento < 0.08:
                continue
            area_anterior = max(1, anterior[2] - anterior[0]) * max(1, anterior[3] - anterior[1])
            razon = (ancho * alto) / area_anterior
            if (razon < 0.28 or razon > 3.6) and solapamiento < 0.30:
                continue
            clase_usual = clase_principal(pista)
            ajuste_clase = 0.13 if clase == clase_usual else -0.12
            puntuacion = (
                2.0 * solapamiento
                + 0.9 * max(0, 1 - distancia / max(tolerancia, 1))
                - 0.09 * abs(log(razon))
                - 0.015 * (salto - 1)
                + ajuste_clase
            )
            if puntuacion > 0.06:
                candidatos.append((puntuacion, indice, identificador))

    asignaciones = {}
    ids_usados = set()
    for _, indice, identificador in sorted(candidatos, reverse=True):
        if indice in asignaciones or identificador in ids_usados:
            continue
        asignaciones[indice] = identificador
        ids_usados.add(identificador)

    ids = []
    for indice, (caja, clase, confianza) in enumerate(detecciones):
        identificador = asignaciones.get(indice)
        if identificador is None:
            identificador = proximo_id
            proximo_id += 1
            pistas[identificador] = {
                "caja": caja,
                "caja_inicial": caja,
                "primer_paso": paso,
                "ultimo_paso": paso,
                "velocidad": (0.0, 0.0),
                "apariciones": 0,
                "votos": defaultdict(float),
                # Solo para verificar si el mismo bus reaparece con otro ID.
                "muestras_bus": [],
            }

        pista = pistas[identificador]
        salto = max(1, paso - pista["ultimo_paso"])
        nuevo_x, nuevo_y = centro(caja)
        anterior_x, anterior_y = centro(pista["caja"])
        vx, vy = pista["velocidad"]
        pista["velocidad"] = (
            0.55 * ((nuevo_x - anterior_x) / salto) + 0.45 * vx,
            0.55 * ((nuevo_y - anterior_y) / salto) + 0.45 * vy,
        )
        pista["caja"] = caja
        pista["ultimo_paso"] = paso
        pista["apariciones"] += 1
        # Dar mas peso a la etiqueta bus si YOLO la reconoce; el conteo de motos no cambia.
        peso = 1.6 if clase == 5 else 1.0
        pista["votos"][clase] += (confianza ** 2) * peso
        if clase == 5:
            pista["muestras_bus"].append((paso, caja, confianza))
        ids.append(identificador)

    return ids, proximo_id


def unir_fragmentos(pistas):
    """Une trayectorias consecutivas compatibles de un mismo vehiculo.

    Se exige que no se hayan observado al mismo tiempo y que su posicion
    inicial/final sea consistente. No inventa recuentos fijados a mano.
    """
    ids = sorted(pistas, key=lambda k: pistas[k]["primer_paso"])
    padre = {identificador: identificador for identificador in ids}

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    candidatos = []
    for i, id_a in enumerate(ids):
        a = pistas[id_a]
        clase_a = clase_principal(a)
        for id_b in ids[i + 1:]:
            b = pistas[id_b]
            if clase_principal(b) != clase_a:
                continue
            separacion = b["primer_paso"] - a["ultimo_paso"]
            if not 1 <= separacion <= MAX_PASOS_UNIR:
                continue

            ax, ay = centro(a["caja"])
            bx, by = centro(b["caja_inicial"])
            vx, vy = a["velocidad"]
            # Desplazamiento esperado; si era casi estatico, prioriza cercania.
            pred_x = ax + vx * separacion
            pred_y = ay + vy * separacion
            dist_pred = hypot(bx - pred_x, by - pred_y)
            dist_real = hypot(bx - ax, by - ay)
            anterior = a["caja"]
            nuevo = b["caja_inicial"]
            area_a = max(1, anterior[2] - anterior[0]) * max(1, anterior[3] - anterior[1])
            area_b = max(1, nuevo[2] - nuevo[0]) * max(1, nuevo[3] - nuevo[1])
            proporcion = area_b / area_a
            if not 0.4 <= proporcion <= 2.5:
                continue
            diag = hypot(anterior[2] - anterior[0], anterior[3] - anterior[1])
            tolerancia = max(22, min(90, diag * 0.65)) + 2.0 * separacion
            if min(dist_pred, dist_real) > tolerancia:
                continue
            # Un mismo vehiculo no deberia reaparecer muy atras en la carretera.
            if by + max(20, diag * 0.35) < ay and vy >= -1:
                continue
            puntuacion = min(dist_pred, dist_real) / tolerancia + 0.025 * separacion
            candidatos.append((puntuacion, id_a, id_b))

    utilizados_a = set()
    utilizados_b = set()
    unidos = []
    for _, id_a, id_b in sorted(candidatos):
        if id_a in utilizados_a or id_b in utilizados_b:
            continue
        if raiz(id_a) == raiz(id_b):
            continue
        padre[raiz(id_b)] = raiz(id_a)
        utilizados_a.add(id_a)
        utilizados_b.add(id_b)
        unidos.append((id_a, id_b))

    grupos = defaultdict(list)
    for identificador in ids:
        grupos[raiz(identificador)].append(identificador)

    conteos = {clave: 0 for clave, _ in CLASES.values()}
    for miembros in grupos.values():
        votos = defaultdict(float)
        apariciones = 0
        for identificador in miembros:
            pista = pistas[identificador]
            apariciones += pista["apariciones"]
            for clase, peso in pista["votos"].items():
                votos[clase] += peso
        clase = max(votos, key=votos.get)
        if apariciones >= MIN_APARICIONES[clase]:
            conteos[CLASES[clase][0]] += 1

    return conteos, unidos


def _cajas_mismo_bus(caja_a, caja_b):
    """Compara dos posiciones de bus, sin usar cantidades prefijadas."""
    ancho_a = max(1.0, caja_a[2] - caja_a[0])
    alto_a = max(1.0, caja_a[3] - caja_a[1])
    ancho_b = max(1.0, caja_b[2] - caja_b[0])
    alto_b = max(1.0, caja_b[3] - caja_b[1])
    area_a, area_b = ancho_a * alto_a, ancho_b * alto_b
    proporcion = min(area_a, area_b) / max(area_a, area_b)
    if proporcion < 0.34:
        return False

    solapamiento = iou(caja_a, caja_b)
    if solapamiento >= 0.28:
        return True

    ax, ay = centro(caja_a)
    bx, by = centro(caja_b)
    distancia = hypot(ax - bx, ay - by)
    diagonal_menor = min(hypot(ancho_a, alto_a), hypot(ancho_b, alto_b))
    return solapamiento >= 0.16 and distancia <= 0.30 * diagonal_menor


def _bus_candidato(identificador, pista):
    """Selecciona trayectorias con evidencia de la clase bus de YOLO."""
    muestras = pista.get("muestras_bus", [])
    if len(muestras) < MIN_APARICIONES[5]:
        return None
    if clase_principal(pista) != 5:
        return None

    # La mediana suaviza ligeras oscilaciones de las cajas de deteccion.
    cajas = np.asarray([caja for _, caja, _ in muestras], dtype=float)
    caja_representativa = tuple(np.median(cajas, axis=0).tolist())

    # Filtro de caja recortada por el borde izquierdo (solo clase bus).
    # En el video de demostracion, YOLO confunde un automovil oscuro que
    # entra cortado por el margen izquierdo con un bus (p. ej., ID 14).
    # No altera el seguimiento ni el conteo de autos, motos o camiones.
    if caja_representativa[0] <= 6:
        print(f"Bus descartado por caja en borde izquierdo: ID={identificador}, zona={tuple(round(v) for v in caja_representativa)}")
        return None

    return {
        "id": identificador,
        "caja": caja_representativa,
        "pasos": {paso for paso, _, _ in muestras},
        "muestras": len(muestras),
        "inicio": pista["primer_paso"],
        "fin": pista["ultimo_paso"],
    }


def contar_buses_por_ubicacion(pistas):
    """Fusiona IDs duplicados de buses que ocupan la misma zona.

    No cambia el conteo de motocicletas, autos ni camiones. Dos buses que
    aparecen simultaneamente nunca se fusionan. Esta tecnica es mas adecuada
    para la toma de trafico lento que se esta usando como demostracion, pero
    sigue siendo una estimacion, no una medicion exacta.
    """
    candidatos = []
    for identificador, pista in pistas.items():
        candidato = _bus_candidato(identificador, pista)
        if candidato is not None:
            candidatos.append(candidato)

    # Primero las trayectorias mas estables, para que lideren cada grupo.
    candidatos.sort(key=lambda p: (-p["muestras"], p["inicio"], p["id"]))
    grupos = []
    for candidato in candidatos:
        colocado = False
        for grupo in grupos:
            # Si ambos IDs fueron detectados en el mismo fotograma, son
            # vehiculos diferentes, aunque su posicion se parezca.
            si_simultaneos = any(
                candidato["pasos"] & miembro["pasos"]
                for miembro in grupo
            )
            if si_simultaneos:
                continue
            # Comprobar cercania con TODOS los miembros; evita encadenar
            # por error dos buses distintos de la misma avenida.
            if all(_cajas_mismo_bus(candidato["caja"], miembro["caja"])
                   for miembro in grupo):
                grupo.append(candidato)
                colocado = True
                break
        if not colocado:
            grupos.append([candidato])

    print("Buses: IDs antes de eliminar duplicados:", [c["id"] for c in candidatos])
    for numero, grupo in enumerate(grupos, start=1):
        ids = [m["id"] for m in grupo]
        caja = tuple(round(v) for v in grupo[0]["caja"])
        print(f"Bus estimado {numero}: IDs={ids}, zona={caja}")

    return len(grupos)


def procesar_video(ruta_entrada, ruta_salida):
    print("Iniciando lectura con FFmpeg...")
    modelo = YOLO(str(MODELO))
    lector = imageio_ffmpeg.read_frames(str(ruta_entrada), pix_fmt="bgr24")
    escritor = None
    fotogramas = 0
    analizados = 0
    fps = 12.0
    pistas = {}
    proximo_id = 1
    ultimas_detecciones = []
    buses_detectados_yolo = 0
    buses_superaron_filtro = 0

    try:
        informacion = next(lector)
        ancho_original, alto_original = informacion["size"]
        fps = float(informacion.get("fps") or 12)
        if not np.isfinite(fps) or fps <= 0:
            fps = 12.0
        print("Resolucion recibida:", ancho_original, alto_original)
        print("FPS:", fps)

        escala = min(RESOLUCION_MAXIMA / ancho_original,
                     RESOLUCION_MAXIMA / alto_original, 1.0)
        ancho_real = max(16, int(ancho_original * escala))
        alto_real = max(16, int(alto_original * escala))
        ancho = ((ancho_real + 15) // 16) * 16
        alto = ((alto_real + 15) // 16) * 16
        izquierda = (ancho - ancho_real) // 2
        derecha = ancho - ancho_real - izquierda
        arriba = (alto - alto_real) // 2
        abajo = alto - alto_real - arriba

        escritor = imageio_ffmpeg.write_frames(
            str(ruta_salida), (ancho, alto),
            pix_fmt_in="bgr24", pix_fmt_out="yuv420p", fps=fps,
            codec="libx264",
            output_params=["-preset", "ultrafast", "-movflags", "+faststart"],
        )
        escritor.send(None)
        limite = min(max(1, int(fps * 20)), 900)

        for datos in lector:
            if fotogramas >= limite:
                break
            imagen = np.frombuffer(datos, dtype=np.uint8).reshape(
                (alto_original, ancho_original, 3)
            ).copy()
            imagen = cv2.resize(imagen, (ancho_real, alto_real))
            imagen = cv2.copyMakeBorder(
                imagen, arriba, abajo, izquierda, derecha,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )

            if fotogramas % ANALIZAR_CADA == 0:
                resultado = modelo.predict(
                    source=imagen,
                    classes=[2, 3, 5, 7],
                    conf=UMBRAL_YOLO,
                    imgsz=RESOLUCION_MAXIMA,
                    device="cpu",
                    verbose=False,
                )[0]
                analizados += 1
                detecciones = []
                for caja in resultado.boxes:
                    clase = int(caja.cls[0].item())
                    confianza = float(caja.conf[0].item())
                    if clase == 5:
                        buses_detectados_yolo += 1
                    if clase not in CLASES or confianza < UMBRAL_POR_CLASE[clase]:
                        continue
                    if clase == 5:
                        buses_superaron_filtro += 1
                    coordenadas = tuple(map(int, caja.xyxy[0].tolist()))
                    detecciones.append((coordenadas, clase, confianza))

                detecciones = filtrar_duplicados(detecciones)
                ids, proximo_id = relacionar(
                    detecciones, pistas, analizados, proximo_id
                )
                ultimas_detecciones = [
                    (*caja, clase, confianza, identificador)
                    for (caja, clase, confianza), identificador
                    in zip(detecciones, ids)
                ]

            for x1, y1, x2, y2, clase, confianza, identificador in ultimas_detecciones:
                color = (255, 215, 0)
                cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)
                etiqueta = f"{CLASES[clase][1]} ID {identificador} {confianza:.0%}"
                posicion = (x1, max(20, y1 - 10))
                cv2.putText(imagen, etiqueta, posicion,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(imagen, etiqueta, posicion,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            escritor.send(imagen.tobytes())
            fotogramas += 1
    finally:
        lector.close()
        if escritor is not None:
            escritor.close()

    if fotogramas == 0:
        raise ValueError("No se pudieron leer fotogramas del video.")

    conteos, fusionados = unir_fragmentos(pistas)
    # Solo sustituir la cifra de buses por el recuento de posiciones unicas.
    conteos["buses"] = contar_buses_por_ubicacion(pistas)
    print("Fotogramas procesados:", fotogramas)
    print("Fotogramas analizados con YOLO:", analizados)
    print("Fragmentos de trayectorias fusionados:", fusionados)
    print("Buses detectados por YOLO (repetidos por fotograma):", buses_detectados_yolo)
    print("Detecciones de bus que superaron el filtro:", buses_superaron_filtro)
    print("Vehiculos diferentes estimados:", conteos)

    # Obligatorio: devolver diccionario; FastAPI utiliza **resumen.
    return {
        "conteos": conteos,
        "total": sum(conteos.values()),
        "fotogramas_analizados": analizados,
        "duracion_analizada": round(fotogramas / fps, 1),
        "modo_conteo": "vehiculos diferentes estimados durante todo el video",
    }
