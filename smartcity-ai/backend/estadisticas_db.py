
# ==========================================
# SMARTCITY AI - ESTADÍSTICAS
# Obtener datos desde SQLite
# ==========================================

import json
from datetime import datetime, timedelta

from backend.historial_db import conectar


def obtener_estadisticas():

    # Consultar los análisis guardados en SQLite
    with conectar() as conexion:
        registros = conexion.execute("""
            SELECT tipo, fecha, total, conteos
            FROM analisis
            ORDER BY id DESC
        """).fetchall()

    # Resumen general
    total_analisis = len(registros)
    total_imagenes = 0
    total_videos = 0
    total_vehiculos = 0

    ultimo_analisis = None

    # Conteo por tipo de vehículo
    vehiculos = {
        "automoviles": 0,
        "motocicletas": 0,
        "buses": 0,
        "camiones": 0
    }

    # Análisis realizados en los últimos 7 días
    hoy = datetime.now().date()

    dias = [
        hoy - timedelta(days=i)
        for i in range(6, -1, -1)
    ]

    analisis_por_dia = {
        dia.isoformat(): 0
        for dia in dias
    }

    # Procesar todos los registros
    for registro in registros:

        tipo = registro["tipo"]

        if tipo == "imagen":
            total_imagenes += 1

        elif tipo == "video":
            total_videos += 1

        total_vehiculos += int(registro["total"] or 0)

        # Leer los conteos guardados por YOLO
        conteos = json.loads(registro["conteos"])

        for categoria in vehiculos:
            vehiculos[categoria] += int(
                conteos.get(categoria, 0)
            )

        # Último análisis registrado
        if ultimo_analisis is None:
            ultimo_analisis = registro["fecha"]

        # Agrupar los análisis por fecha
        try:
            fecha = datetime.fromisoformat(
                registro["fecha"]
            ).date()

            clave = fecha.isoformat()

            if clave in analisis_por_dia:
                analisis_por_dia[clave] += 1

        except (ValueError, TypeError):
            pass

    # Datos para los gráficos
    grafico_dias = [
        {
            "fecha": dia,
            "analisis": cantidad
        }
        for dia, cantidad in analisis_por_dia.items()
    ]

    return {
        "resumen": {
            "total_analisis": total_analisis,
            "imagenes": total_imagenes,
            "videos": total_videos,
            "total_vehiculos": total_vehiculos,
            "ultimo_analisis": ultimo_analisis
        },

        "vehiculos": vehiculos,

        "tipos_analisis": {
            "imagenes": total_imagenes,
            "videos": total_videos
        },

        "analisis_por_dia": grafico_dias
    }
