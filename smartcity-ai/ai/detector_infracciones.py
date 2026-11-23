"""SmartCity AI - prototipo de infracciones en video.

Detecta eventos potenciales, NO infracciones legales confirmadas.
El semaforo se configura manualmente (SIMULADO) y las zonas se aproximan
por coordenadas relativas. Ajustar las zonas al encuadre del video.

Ejecutar desde la raiz del proyecto:
python -m ai.detector_infracciones --video "video_prueba.mp4" --semaforo rojo

NO modifica ai/detector_video.py ni backend/main.py.
"""

import argparse
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

from ai.detector_video import filtrar_duplicados, relacionar

RAIZ = Path(__file__).resolve().parent.parent
MODELO_BASE = RAIZ / "yolo11n.pt"
MODELO_LOCAL = (
    RAIZ / "runs" / "detect" / "entrenamientos" /
    "modelo_motokar" / "weights" / "best.pt"
)
NOMBRES = {
    2: "Automovil", 3: "Motocicleta", 5: "Bus", 7: "Camion",
    80: "Motokar", 81: "Motocarguero",
}


def pie(caja):
    """Punto de contacto aproximado del vehiculo con la calzada."""
    x1, _, x2, y2 = caja
    return ((x1 + x2) // 2, y2)


def proporcion_solapamiento(caja_a, caja_b):
    """Area de interseccion / area de la caja mas pequena."""
    x1 = max(caja_a[0], caja_b[0])
    y1 = max(caja_a[1], caja_b[1])
    x2 = min(caja_a[2], caja_b[2])
    y2 = min(caja_a[3], caja_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, caja_a[2] - caja_a[0]) * max(1, caja_a[3] - caja_a[1])
    area_b = max(1, caja_b[2] - caja_b[0]) * max(1, caja_b[3] - caja_b[1])
    return inter / min(area_a, area_b)


def seleccionar_detecciones(base, local):
    """Evita, principalmente, que un motokar cuente tambien como motocicleta."""
    elegidas = list(local)
    for caja, clase, confianza in base:
        if any(
            proporcion_solapamiento(caja, otra_caja) >= 0.45
            for otra_caja, _, _ in local
        ):
            continue
        elegidas.append((caja, clase, confianza))
    return filtrar_duplicados(elegidas)


def registrar_eventos(identificador, caja, estado, rojo, linea_y,
                      zona, direccion, tiempo_segundos, clase):
    """Retorna eventos NUEVOS para un mismo vehiculo seguido en varios frames."""
    x, y = pie(caja)
    datos = estado.setdefault(identificador, {
        "pie_anterior": None,
        "dentro_consecutivos": 0,
        "cruzo_rojo": False,
        "invadio": False,
    })
    anterior = datos["pie_anterior"]
    eventos = []
    nombre = NOMBRES.get(clase, "Vehiculo")

    if False and rojo and anterior is not None and not datos["cruzo_rojo"]:
        previo_y = anterior[1]
        if direccion == "abajo":
            cruzo = previo_y < linea_y <= y
        else:
            cruzo = previo_y > linea_y >= y
        if cruzo:
            datos["cruzo_rojo"] = True
            eventos.append({
                "tipo": "Paso con luz roja (simulado)",
                "vehiculo": nombre, "id_vehiculo": identificador,
                "tiempo_segundos": round(tiempo_segundos, 2),
            })

    x1, y1, x2, y2 = zona
    dentro = x1 <= x <= x2 and y1 <= y <= y2
    datos["dentro_consecutivos"] = datos["dentro_consecutivos"] + 1 if dentro else 0
    # Prototipo: ocupacion sostenida de zona mientras el estado simulado es rojo.
    if rojo and datos["dentro_consecutivos"] >= 2 and not datos["invadio"]:
        datos["invadio"] = True
        eventos.append({
            "tipo": "Posible invasion de paso peatonal (simulado)",
            "vehiculo": nombre, "id_vehiculo": identificador,
            "tiempo_segundos": round(tiempo_segundos, 2),
        })

    datos["pie_anterior"] = (x, y)
    return eventos


def procesar_infracciones(
    entrada, salida, semaforo="rojo", linea=0.55,
    zona_inicio=0.58, zona_fin=0.78,
    x_inicio=0.05, x_fin=0.95, direccion="abajo", segundos=20,
):
    """Procesa un video, guarda MP4, JSON y capturas de eventos."""
    entrada = Path(entrada).resolve()
    salida = Path(salida).resolve()
    if not entrada.is_file():
        raise FileNotFoundError(f"No encuentro el video: {entrada}")
    if not MODELO_BASE.is_file():
        raise FileNotFoundError(f"No encuentro el modelo base: {MODELO_BASE}")
    if not (0 <= linea <= 1 and 0 <= zona_inicio < zona_fin <= 1 and
            0 <= x_inicio < x_fin <= 1):
        raise ValueError("Los porcentajes de linea y zona deben estar entre 0 y 1")
    if semaforo not in {"rojo", "verde"}:
        raise ValueError("El semaforo debe ser rojo o verde")
    if direccion not in {"arriba", "abajo"}:
        raise ValueError("La direccion debe ser arriba o abajo")

    salida.parent.mkdir(parents=True, exist_ok=True)
    evidencias = salida.parent / (salida.stem + "_evidencias")
    modelo_base = YOLO(str(MODELO_BASE))
    modelo_local = YOLO(str(MODELO_LOCAL)) if MODELO_LOCAL.is_file() else None
    print("Modelo de motokar/motocarguero:", "ACTIVO" if modelo_local else "NO ENCONTRADO")
    print("Semaforo", semaforo.upper(), "(SIMULADO; no se reconoce desde el video)")

    lector = imageio_ffmpeg.read_frames(str(entrada), pix_fmt="bgr24")
    escritor = None
    fotogramas = 0
    analizados = 0
    fps = 12.0
    pistas = {}
    proximo_id = 1
    estado = {}
    eventos = []
    ultimas = []
    try:
        info = next(lector)
        original_w, original_h = info["size"]
        fps = float(info.get("fps") or 12)
        if not np.isfinite(fps) or fps <= 0:
            fps = 12.0
        escala = min(960 / original_w, 960 / original_h, 1.0)
        real_w = max(16, int(original_w * escala))
        real_h = max(16, int(original_h * escala))
        ancho = ((real_w + 15) // 16) * 16
        alto = ((real_h + 15) // 16) * 16
        izq, arriba = (ancho - real_w) // 2, (alto - real_h) // 2
        der, abajo = ancho - real_w - izq, alto - real_h - arriba

        linea_y = int(linea * real_h) + arriba
        zona = (int(x_inicio * real_w) + izq, int(zona_inicio * real_h) + arriba,
                int(x_fin * real_w) + izq, int(zona_fin * real_h) + arriba)

        escritor = imageio_ffmpeg.write_frames(
            str(salida), (ancho, alto), pix_fmt_in="bgr24",
            pix_fmt_out="yuv420p", fps=fps, codec="libx264",
            output_params=["-preset", "ultrafast", "-movflags", "+faststart"],
        )
        escritor.send(None)
        max_fotogramas = min(int(fps * segundos), 900)
        for datos in lector:
            if fotogramas >= max_fotogramas:
                break
            imagen = np.frombuffer(datos, dtype=np.uint8).reshape(
                (original_h, original_w, 3)
            ).copy()
            imagen = cv2.resize(imagen, (real_w, real_h))
            imagen = cv2.copyMakeBorder(
                imagen, arriba, abajo, izq, der,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
            rojo = semaforo == "rojo"
            pendientes = []
            if fotogramas % 2 == 0:
                base_result = modelo_base.predict(
                    imagen, classes=[2, 3, 5, 7], conf=0.25,
                    imgsz=416, device="cpu", verbose=False,
                )[0]
                base = [
                    (tuple(map(int, caja.xyxy[0].tolist())),
                     int(caja.cls[0]), float(caja.conf[0]))
                    for caja in base_result.boxes
                ]
                local = []
                if modelo_local:
                    local_result = modelo_local.predict(
                        imagen, conf=0.25, imgsz=416,
                        device="cpu", verbose=False,
                    )[0]
                    local = [
                        (tuple(map(int, caja.xyxy[0].tolist())),
                         80 + int(caja.cls[0]), float(caja.conf[0]))
                        for caja in local_result.boxes
                    ]
                detecciones = seleccionar_detecciones(base, local)
                analizados += 1
                ids, proximo_id = relacionar(
                    detecciones, pistas, analizados, proximo_id
                )
                ultimas = [(*caja, clase, conf, id_)
                           for (caja, clase, conf), id_ in zip(detecciones, ids)]
                for caja, identificador in zip(detecciones, ids):
                    coordenadas, clase, _ = caja
                    nuevos = registrar_eventos(
                        identificador, coordenadas, estado,
                        rojo, linea_y, zona, direccion,
                        fotogramas / fps, clase,
                    )
                    for evento in nuevos:
                        evidencias.mkdir(parents=True, exist_ok=True)
                        nombre = f"evento_{len(eventos)+1:03d}.jpg"
                        evento["imagen"] = str(evidencias / nombre)
                        pendientes.append(evidencias / nombre)
                        eventos.append(evento)
                        print(f"ALERTA: {evento['tipo']} | {evento['vehiculo']} ID {identificador}")

            # Dibujar areas aproximadas; NO son lineas de la calle detectadas por IA.
            cv2.rectangle(imagen, (zona[0], zona[1]), (zona[2], zona[3]), (255, 180, 30), 2)
            cv2.putText(imagen, "ZONA PEATONAL MANUAL", (zona[0], max(18, zona[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 30), 2)
            cv2.line(imagen, (izq, linea_y), (izq + real_w, linea_y), (0, 0, 255), 3)
            cv2.putText(imagen, "LINEA DE DETENCION", (izq + 5, max(40, linea_y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            estado_color = (0, 0, 255) if rojo else (0, 220, 0)
            cv2.rectangle(imagen, (0, 0), (ancho, 33), (24, 24, 24), -1)
            cv2.putText(imagen, f"SEMAFORO: {semaforo.upper()} (SIMULADO)",
                        (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.64, estado_color, 2)
            for x1, y1, x2, y2, clase, conf, identificador in ultimas:
                color = (50, 240, 80)
                cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)
                nombre = NOMBRES.get(clase, "Vehiculo")
                cv2.putText(imagen, f"{nombre} ID {identificador} {conf:.0%}",
                            (x1, max(45, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
                cv2.circle(imagen, pie((x1, y1, x2, y2)), 4, (255, 255, 255), -1)
            if eventos:
                cv2.putText(imagen, f"ALERTAS (DEMO): {len(eventos)}",
                            (10, alto - 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
            for ruta_captura in pendientes:
                cv2.imwrite(str(ruta_captura), imagen)
            escritor.send(imagen.tobytes())
            fotogramas += 1
    finally:
        lector.close()
        if escritor is not None:
            escritor.close()

    if fotogramas == 0:
        raise RuntimeError("El video no contenia fotogramas validos")
    resumen = {
        "modo": "demostracion (semaforo y zonas configurados manualmente)",
        "semaforo": semaforo,
        "fotogramas": fotogramas,
        "fotogramas_analizados": analizados,
        "duracion_segundos": round(fotogramas / fps, 2),
        "total_eventos": len(eventos),
        "eventos": eventos,
    }
    ruta_json = salida.with_suffix(".json")
    ruta_json.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Video guardado en: {salida}")
    print(f"Resumen guardado en: {ruta_json}")
    print(f"Eventos detectados: {len(eventos)}")
    return resumen


def main():
    parser = argparse.ArgumentParser(description="Prototipo de infracciones para SmartCity AI")
    parser.add_argument("--video", required=True, help="Ruta del video de entrada")
    parser.add_argument("--salida", default="resultado_infracciones.mp4")
    parser.add_argument("--semaforo", choices=["rojo", "verde"], default="rojo")
    parser.add_argument("--linea", type=float, default=0.55, help="Altura de la linea: 0.0 - 1.0")
    parser.add_argument("--zona-inicio", type=float, default=0.58)
    parser.add_argument("--zona-fin", type=float, default=0.78)
    parser.add_argument("--x-inicio", type=float, default=0.05)
    parser.add_argument("--x-fin", type=float, default=0.95)
    parser.add_argument("--direccion", choices=["arriba", "abajo"], default="abajo")
    parser.add_argument("--segundos", type=float, default=20.0)
    args = parser.parse_args()
    procesar_infracciones(
        args.video, args.salida, args.semaforo, args.linea,
        args.zona_inicio, args.zona_fin,
        args.x_inicio, args.x_fin, args.direccion, args.segundos,
    )


if __name__ == "__main__":
    main()
