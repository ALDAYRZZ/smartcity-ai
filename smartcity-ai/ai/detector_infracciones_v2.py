"""SmartCity AI - prototipo de infracciones v2 (modos separados).

Peatonal: seleccion manual de 4 esquinas de un POLIGONO sobre el paso peatonal.
Semaforo: cruce de linea con estado de luz SIMULADO por argumento.
Los eventos son posibles infracciones para DEMOSTRACION, no pruebas legales.
Este archivo no modifica el detector de video ni el backend existentes.
"""

import argparse
import json
import sys
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
                      poligono, direccion, tiempo_segundos, clase, modo):
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

    if modo == "semaforo" and rojo and anterior is not None and not datos["cruzo_rojo"]:
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

    if modo == "peatonal":
        # Punto de contacto aproximado: centro de la base de la caja del vehiculo.
        dentro = cv2.pointPolygonTest(poligono, (float(x), float(y)), False) >= 0
        datos["dentro_consecutivos"] = datos["dentro_consecutivos"] + 1 if dentro else 0
    # La zona peatonal NO necesita un semaforo: en el video hay una senal STOP.
    if modo == "peatonal" and datos["dentro_consecutivos"] >= 2 and not datos["invadio"]:
        datos["invadio"] = True
        eventos.append({
            "tipo": "Posible ocupacion de paso peatonal (zona manual)",
            "vehiculo": nombre, "id_vehiculo": identificador,
            "tiempo_segundos": round(tiempo_segundos, 2),
        })

    datos["pie_anterior"] = (x, y)
    return eventos


ZONA_GUARDADA = RAIZ / "zona_peatonal.json"


def seleccionar_paso_peatonal(archivo_video, segundo=2.0):
    """El usuario marca 4 esquinas sobre una imagen REAL del video.

    Guardamos las coordenadas como proporciones 0-1. La seleccion se vuelve
    a aplicar al video original con independencia de su resolucion.
    """
    ruta = Path(archivo_video).resolve()
    if not ruta.is_file():
        raise FileNotFoundError(f"No encuentro el video: {ruta}")
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise RuntimeError("No pude abrir el video para seleccionar la zona")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, segundo) * 1000)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("No pude leer una imagen del video")

    alto_original, ancho_original = frame.shape[:2]
    escala = min(850 / ancho_original, 750 / alto_original, 1.0)
    w = max(1, round(ancho_original * escala))
    h = max(1, round(alto_original * escala))
    pantalla = cv2.resize(frame, (w, h))
    puntos = []
    nombre = "MARCAR PASO PEATONAL - 4 CLICS"

    def clic(evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
            puntos.append((x, y))

    try:
        cv2.namedWindow(nombre, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nombre, w, h)
        cv2.setMouseCallback(nombre, clic)
        while True:
            vista = pantalla.copy()
            if len(puntos) >= 2:
                cv2.polylines(vista, [np.array(puntos, np.int32)], False, (255, 220, 30), 2)
            if len(puntos) == 4:
                cv2.polylines(vista, [np.array(puntos, np.int32)], True, (255, 220, 30), 3)
            for indice, (x, y) in enumerate(puntos, start=1):
                cv2.circle(vista, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(vista, str(indice), (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(vista, f"Puntos {len(puntos)}/4. ENTER guardar - R reiniciar - ESC cancelar",
                        (5, max(20, h - 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        min(0.6, w / 850), (0, 0, 0), 3)
            cv2.putText(vista, f"Puntos {len(puntos)}/4. ENTER guardar - R reiniciar - ESC cancelar",
                        (5, max(20, h - 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        min(0.6, w / 850), (255, 255, 255), 1)
            cv2.imshow(nombre, vista)
            tecla = cv2.waitKey(50) & 0xFF
            if tecla in (13, 10) and len(puntos) == 4:
                break
            if tecla in (ord('r'), ord('R')):
                puntos.clear()
            if tecla == 27:
                print("Seleccion cancelada, sin cambios")
                return
    finally:
        cv2.destroyAllWindows()

    relativos = [[round(x / w, 6), round(y / h, 6)] for x, y in puntos]
    area = cv2.contourArea(np.array(puntos, dtype=np.float32))
    if area < 20:
        raise ValueError("El poligono es demasiado pequeno; repite la seleccion")
    ZONA_GUARDADA.write_text(json.dumps({
        "video": ruta.name,
        "puntos": relativos,
        "aviso": "Zona de demostracion seleccionada manualmente; no implica infraccion legal"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Zona peatonal guardada en:", ZONA_GUARDADA)
    print("Puntos:", relativos)
    print("Ahora ejecuta el analisis en modo peatonal")


def leer_zona(video, ancho_real, alto_real, izq, arriba):
    if not ZONA_GUARDADA.is_file():
        raise FileNotFoundError(
            "Primero marca el paso peatonal con --modo peatonal --elegir-zona"
        )
    zona = json.loads(ZONA_GUARDADA.read_text(encoding="utf-8"))
    if zona.get("video") != Path(video).name:
        raise ValueError(
            "La zona guardada pertenece a otro video. Repite --elegir-zona."
        )
    puntos = zona.get("puntos", [])
    if len(puntos) != 4:
        raise ValueError("La zona guardada debe tener exactamente 4 puntos")
    return np.array([
        (int(x * ancho_real) + izq, int(y * alto_real) + arriba)
        for x, y in puntos
    ], dtype=np.int32)


def procesar_infracciones(
    entrada, salida, semaforo="rojo", linea=0.55,
    direccion="abajo", segundos=20, modo="peatonal", sin_modelo_local=False,
):
    """Procesa un video, guarda MP4, JSON y capturas de eventos."""
    entrada = Path(entrada).resolve()
    salida = Path(salida).resolve()
    if not entrada.is_file():
        raise FileNotFoundError(f"No encuentro el video: {entrada}")
    if not MODELO_BASE.is_file():
        raise FileNotFoundError(f"No encuentro el modelo base: {MODELO_BASE}")
    if not 0 <= linea <= 1:
        raise ValueError("La linea debe estar entre 0 y 1")
    if modo not in {"peatonal", "semaforo"}:
        raise ValueError("El modo debe ser peatonal o semaforo")
    if semaforo not in {"rojo", "verde"}:
        raise ValueError("El semaforo debe ser rojo o verde")
    if direccion not in {"arriba", "abajo"}:
        raise ValueError("La direccion debe ser arriba o abajo")

    salida.parent.mkdir(parents=True, exist_ok=True)
    evidencias = salida.parent / (salida.stem + "_evidencias")
    modelo_base = YOLO(str(MODELO_BASE))
    modelo_local = YOLO(str(MODELO_LOCAL)) if MODELO_LOCAL.is_file() and not sin_modelo_local else None
    print("Modelo de motokar/motocarguero:", "ACTIVO" if modelo_local else "NO ENCONTRADO")
    print("Modo:", modo.upper())
    if modo == "semaforo":
        print("Semaforo", semaforo.upper(), "(SIMULADO; no se reconoce desde el video)")
    else:
        print("Paso peatonal dibujado manualmente (no requiere semaforo)")

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
        poligono = leer_zona(entrada, real_w, real_h, izq, arriba) if modo == "peatonal" else None

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
                        rojo, linea_y, poligono, direccion,
                        fotogramas / fps, clase, modo,
                    )
                    for evento in nuevos:
                        evidencias.mkdir(parents=True, exist_ok=True)
                        nombre = f"evento_{len(eventos)+1:03d}.jpg"
                        evento["imagen"] = str(evidencias / nombre)
                        pendientes.append(evidencias / nombre)
                        eventos.append(evento)
                        print(f"ALERTA: {evento['tipo']} | {evento['vehiculo']} ID {identificador}")

            # Solo dibuja lo que corresponde al modo seleccionado.
            if modo == "peatonal":
                cv2.polylines(imagen, [poligono], True, (255, 180, 30), 3)
                posicion = tuple(poligono[np.argmin(poligono[:, 1])])
                cv2.putText(imagen, "PASO PEATONAL (ZONA MANUAL)",
                            (int(posicion[0]), max(43, int(posicion[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 30), 2)
                cv2.rectangle(imagen, (0, 0), (ancho, 33), (24, 24, 24), -1)
                cv2.putText(imagen, "MODO: PASO PEATONAL (DEMO)",
                            (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 30), 2)
            else:
                cv2.line(imagen, (izq, linea_y), (izq + real_w, linea_y), (0, 0, 255), 3)
                cv2.putText(imagen, "LINEA DE DETENCION", (izq + 5, max(40, linea_y - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                color = (0, 0, 255) if rojo else (0, 220, 0)
                cv2.rectangle(imagen, (0, 0), (ancho, 33), (24, 24, 24), -1)
                cv2.putText(imagen, f"SEMAFORO: {semaforo.upper()} (SIMULADO)",
                            (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
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
        "modo": modo,
        "aviso": "Prototipo: no confirma infracciones legales; zona/estado manual",
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
    parser = argparse.ArgumentParser(description="SmartCity AI - infracciones, 2 modos (DEMO)")
    parser.add_argument("--video", required=True, help="Video de entrada")
    parser.add_argument("--modo", choices=["peatonal", "semaforo"], required=True)
    parser.add_argument("--salida", default="resultado_infracciones_v2.mp4")
    parser.add_argument("--elegir-zona", action="store_true",
                        help="Seleccionar con 4 clics la zona peatonal, sin procesar")
    parser.add_argument("--segundo", type=float, default=2.0,
                        help="Instante del video para marcar la zona")
    parser.add_argument("--semaforo", choices=["rojo", "verde"], default="rojo")
    parser.add_argument("--linea", type=float, default=0.78)
    parser.add_argument("--direccion", choices=["arriba", "abajo"], default="arriba")
    parser.add_argument("--segundos", type=float, default=20.0)
    parser.add_argument("--sin-modelo-local", action="store_true",
                        help="Evita etiquetas de motokar falsas al usar un video sin motokares")
    args = parser.parse_args()
    if args.elegir_zona:
        if args.modo != "peatonal":
            parser.error("--elegir-zona solo se usa con --modo peatonal")
        seleccionar_paso_peatonal(args.video, args.segundo)
        return
    procesar_infracciones(
        args.video, args.salida, args.semaforo, args.linea,
        args.direccion, args.segundos, args.modo, args.sin_modelo_local
    )


if __name__ == "__main__":
    main()
