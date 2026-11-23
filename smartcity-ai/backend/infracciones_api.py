
"""API de infracciones para SmartCity AI.

Permite analizar videos desde la pagina web.
Modos disponibles:
- Semaforo: cruce de linea con luz simulada.
- Peatonal: ocupacion de una zona marcada manualmente.

Las alertas son experimentales.
"""

import json
import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from ai.detector_infracciones_web import procesar_infracciones

from backend.historial_db import (
    inicializar_bd,
    guardar_infracciones,
    obtener_infracciones
)
# ============================================
# CONFIGURACION
# ============================================

router = APIRouter()

LOGGER = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent.parent

RESULTADOS = RAIZ / "backend" / "resultados"

ZONA_GUARDADA = RAIZ / "zona_peatonal.json"

LIMITE_VIDEO = 30 * 1024 * 1024

EXTENSIONES = {".mp4", ".webm"}


# ============================================
# VALIDAR ZONA PEATONAL
# ============================================

def _validar_puntos(puntos):
    """Valida las cuatro esquinas del paso peatonal."""

    if not isinstance(puntos, list) or len(puntos) != 4:
        raise ValueError(
            "Marca cuatro esquinas del paso peatonal."
        )

    normalizados = []

    for par in puntos:

        if not isinstance(par, (list, tuple)) or len(par) != 2:
            raise ValueError(
                "Cada punto debe tener X e Y."
            )

        try:
            x = float(par[0])
            y = float(par[1])

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Coordenadas no validas."
            ) from error

        if not (
            np.isfinite(x)
            and np.isfinite(y)
            and 0 <= x <= 1
            and 0 <= y <= 1
        ):
            raise ValueError(
                "Los puntos deben estar dentro del video."
            )

        normalizados.append([x, y])

    poligono = np.asarray(
        normalizados,
        dtype=np.float32
    )

    if (
        cv2.contourArea(poligono) < 0.0001
        or not cv2.isContourConvex(poligono)
    ):
        raise ValueError(
            "Los cuatro puntos deben rodear la zona "
            "en orden, sin cruzarse."
        )

    return normalizados


# ============================================
# CONSULTAR ZONA PEATONAL GUARDADA
# ============================================

@router.get("/zona-peatonal-guardada")
def zona_peatonal_guardada(nombre: str):
    """Recupera los puntos dibujados anteriormente."""

    nombre = Path(
        nombre.replace("\\", "/")
    ).name

    if not ZONA_GUARDADA.exists():
        return {"disponible": False}

    try:
        datos = json.loads(
            ZONA_GUARDADA.read_text(
                encoding="utf-8"
            )
        )

        if datos.get("video") != nombre:
            return {"disponible": False}

        puntos = _validar_puntos(
            datos.get("puntos")
        )

        return {
            "disponible": True,
            "puntos": puntos
        }

    except (ValueError, json.JSONDecodeError, OSError):

        LOGGER.warning(
            "No se pudo leer la zona peatonal guardada",
            exc_info=True
        )

        return {"disponible": False}


# ============================================
# ANALIZAR INFRACCIONES DESDE LA WEB
# ============================================

@router.post("/analizar-infracciones")
async def analizar_infracciones(

    archivo: UploadFile = File(...),

    modo: str = Form(...),

    semaforo: str = Form("rojo"),

    linea: float = Form(0.78),

    direccion: str = Form("arriba"),

    sin_modelo_local: bool = Form(True),

    puntos_zona: str = Form(""),

):
    """Recibe un video y devuelve los eventos detectados."""

    # VALIDACIONES

    if modo not in {"semaforo", "peatonal"}:
        raise HTTPException(
            status_code=400,
            detail="Selecciona semaforo o paso peatonal."
        )


    if semaforo not in {"rojo", "verde", "automatico"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "Estado de semáforo inválido. "
                "Opciones: rojo, verde o automatico."
            )
        )
    if direccion not in {"arriba", "abajo"}:
        raise HTTPException(
            status_code=400,
            detail="Selecciona la direccion del cruce."
        )

    if not 0 <= linea <= 1:
        raise HTTPException(
            status_code=400,
            detail="La altura de la linea debe estar entre 0 y 1."
        )

    nombre_original = Path(
        (archivo.filename or "").replace("\\", "/")
    ).name

    extension = Path(nombre_original).suffix.lower()

    if extension not in EXTENSIONES:
        raise HTTPException(
            status_code=400,
            detail="Selecciona un video MP4 o WebM."
        )

    # ============================================
    # RECUPERAR LA ZONA PEATONAL
    # ============================================

    puntos = None

    if modo == "peatonal":

        if puntos_zona.strip():

            try:
                puntos = _validar_puntos(
                    json.loads(puntos_zona)
                )

            except (ValueError, json.JSONDecodeError) as error:

                raise HTTPException(
                    status_code=400,
                    detail=str(error)
                ) from error

        elif ZONA_GUARDADA.exists():

            try:
                datos = json.loads(
                    ZONA_GUARDADA.read_text(
                        encoding="utf-8"
                    )
                )

                if datos.get("video") == nombre_original:

                    puntos = _validar_puntos(
                        datos.get("puntos")
                    )

            except (
                OSError,
                ValueError,
                json.JSONDecodeError
            ):
                pass

        if puntos is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Primero marca cuatro puntos "
                    "del paso peatonal."
                )
            )

    # ============================================
    # LEER EL VIDEO SUBIDO
    # ============================================

    try:
        contenido = await archivo.read(
            LIMITE_VIDEO + 1
        )

    finally:
        await archivo.close()

    if not contenido:
        raise HTTPException(
            status_code=400,
            detail="El video esta vacio."
        )

    if len(contenido) > LIMITE_VIDEO:
        raise HTTPException(
            status_code=413,
            detail="El video supera el limite de 30 MB."
        )

    # ============================================
    # GUARDAR VIDEO TEMPORALMENTE
    # ============================================

    RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    identificador = uuid4().hex

    entrada = (
        RESULTADOS /
        f"entrada_infracciones_{identificador}{extension}"
    )

    salida = (
        RESULTADOS /
        f"infracciones_{identificador}.mp4"
    )

    entrada.write_bytes(contenido)

    # ============================================
    # EJECUTAR DETECTOR YOLO
    # ============================================

    try:

        resumen = await run_in_threadpool(
            procesar_infracciones,
            entrada,
            salida,
            semaforo,
            linea,
            direccion,
            20,
            modo,
            sin_modelo_local,
            puntos,
        )

    except Exception as error:

        LOGGER.exception(
            "Error al analizar infracciones"
        )

        salida.unlink(missing_ok=True)

        salida.with_suffix(
            ".json"
        ).unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=(
                "No se pudo procesar el video. "
                "Revisa el formato y la zona marcada."
            )
        ) from error

    finally:

        entrada.unlink(
            missing_ok=True
        )

    # ============================================
    # PREPARAR EVIDENCIAS PARA LA PAGINA WEB
    # ============================================

    eventos = []

    for evento in resumen.get("eventos", []):

        dato = dict(evento)

        imagen = dato.pop(
            "imagen",
            None
        )

        if imagen:

            ruta = Path(
                imagen
            ).resolve()

            try:

                relativo = ruta.relative_to(
                    RESULTADOS.resolve()
                )

                dato["imagen_url"] = (
                    "/resultados/" + relativo.as_posix()
                )

            except ValueError:
                pass

        eventos.append(dato)

    # ============================================
    # GUARDAR INFRACCIONES EN SQLITE
    # ============================================

    inicializar_bd()

    video_url = f"/resultados/{salida.name}"

    cantidad_guardada = await run_in_threadpool(
        guardar_infracciones,
        identificador,
        nombre_original,
        modo,
        eventos,
        video_url
    )

    LOGGER.info(
        "Infracciones guardadas en SQLite: %s",
        cantidad_guardada
    )
    # ============================================
    # RESPUESTA FINAL A LA PAGINA WEB
    # ============================================

    return {

        "mensaje": (
            "Analisis de infracciones "
            "de demostracion terminado"
        ),

        "modo": modo,

        "video_url": (
            f"/resultados/{salida.name}"
        ),

        "total_eventos": len(eventos),

        "eventos": eventos,

        "aviso": (
            "Alertas experimentales. "
            "La luz del semaforo y las zonas "
            "son manuales; no son pruebas legales."
        ),

    }
@router.get("/historial-infracciones")
def historial_infracciones():
    inicializar_bd()
    return obtener_infracciones()
