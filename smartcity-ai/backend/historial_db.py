
import sqlite3
import json
from pathlib import Path
from datetime import datetime

RUTA_BD = Path(__file__).resolve().parent / "smartcity_historial.db"


def conectar():
    conexion = sqlite3.connect(RUTA_BD, timeout=10)
    conexion.row_factory = sqlite3.Row
    return conexion


def inicializar_bd():
    with conectar() as conexion:

        # Historial existente
        conexion.execute("""
            CREATE TABLE IF NOT EXISTS analisis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                archivo TEXT NOT NULL,
                tipo TEXT NOT NULL,
                fecha TEXT NOT NULL,
                total INTEGER NOT NULL,
                conteos TEXT NOT NULL,
                resultado_url TEXT,
                estado TEXT DEFAULT 'Completado'
            )
        """)

        # Nueva tabla para infracciones
        conexion.execute("""
            CREATE TABLE IF NOT EXISTS infracciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analisis_id TEXT NOT NULL,
                archivo TEXT NOT NULL,
                fecha TEXT NOT NULL,
                tipo_infraccion TEXT NOT NULL,
                vehiculo TEXT,
                id_vehiculo INTEGER,
                tiempo_segundos REAL,
                imagen_url TEXT,
                video_url TEXT,
                modo TEXT,
                estado TEXT DEFAULT 'Posible infracción'
            )
        """)


def fecha_actual():
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def guardar_analisis(
    archivo,
    tipo,
    conteos,
    total,
    resultado_url=None
):
    with conectar() as conexion:
        conexion.execute("""
            INSERT INTO analisis
            (archivo, tipo, fecha, total, conteos, resultado_url)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            archivo,
            tipo,
            fecha_actual(),
            total,
            json.dumps(conteos, ensure_ascii=False),
            resultado_url
        ))


def guardar_infracciones(
    analisis_id,
    archivo,
    modo,
    eventos,
    video_url
):
    """Guarda las alertas detectadas durante un análisis."""

    fecha = fecha_actual()

    registros = []

    for evento in eventos:
        registros.append((
            analisis_id,
            archivo,
            fecha,
            evento.get("tipo", "Posible infracción"),
            evento.get("vehiculo", "Vehículo"),
            evento.get("id_vehiculo"),
            evento.get("tiempo_segundos"),
            evento.get("imagen_url"),
            video_url,
            modo
        ))

    if not registros:
        return 0

    with conectar() as conexion:
        conexion.executemany("""
            INSERT INTO infracciones (
                analisis_id,
                archivo,
                fecha,
                tipo_infraccion,
                vehiculo,
                id_vehiculo,
                tiempo_segundos,
                imagen_url,
                video_url,
                modo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, registros)

    return len(registros)


def obtener_infracciones():
    with conectar() as conexion:

        filas = conexion.execute("""
            SELECT *
            FROM infracciones
            ORDER BY id DESC
            LIMIT 300
        """).fetchall()

    registros = [dict(fila) for fila in filas]

    return {
        "total": len(registros),
        "registros": registros
    }


def obtener_historial():
    with conectar() as conexion:

        filas = conexion.execute("""
            SELECT *
            FROM analisis
            ORDER BY id DESC
            LIMIT 300
        """).fetchall()

        total_infracciones = conexion.execute("""
            SELECT COUNT(*)
            FROM infracciones
        """).fetchone()[0]

    registros = []

    for fila in filas:
        registro = dict(fila)

        try:
            registro["conteos"] = json.loads(
                registro["conteos"]
            )
        except (ValueError, TypeError):
            registro["conteos"] = {}

        registros.append(registro)

    return {
        "resumen": {
            "total": len(registros),
            "imagenes": sum(
                r["tipo"] == "imagen"
                for r in registros
            ),
            "videos": sum(
                r["tipo"] == "video"
                for r in registros
            ),
            "infracciones": total_infracciones,
            "ultimo": (
                registros[0]["fecha"]
                if registros else None
            )
        },
        "registros": registros
    }
