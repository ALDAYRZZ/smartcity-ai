
"use strict";

// ESTADÍSTICAS DE SMARTCITY AI

document.addEventListener("DOMContentLoaded", cargarEstadisticas);

function colocarTexto(id, texto) {
    const elemento = document.getElementById(id);
    if (elemento) {
        elemento.textContent = texto;
    }
}

function numero(valor) {
    return Number(valor) || 0;
}

async function cargarEstadisticas() {
    const estado = "est-estado-conexion";

    try {
        // Obtener datos reales de SQLite mediante FastAPI
        const respuesta = await fetch("/estadisticas", {
            cache: "no-store"
        });

        if (!respuesta.ok) {
            throw new Error("Error HTTP " + respuesta.status);
        }

        const datos = await respuesta.json();

        const resumen = datos.resumen;
        const vehiculos = datos.vehiculos;
        const tipos = datos.tipos_analisis;
        const dias = datos.analisis_por_dia;

        // ==================================
        // TARJETAS SUPERIORES
        // ==================================

        colocarTexto(
            "est-total-analisis",
            resumen.total_analisis
        );

        colocarTexto(
            "est-total-imagenes",
            resumen.imagenes
        );

        colocarTexto(
            "est-total-videos",
            resumen.videos
        );

        colocarTexto(
            "est-total-vehiculos",
            resumen.total_vehiculos
        );

        let ultimaFecha = "Sin registros";

        if (resumen.ultimo_analisis) {
            ultimaFecha = new Date(
                resumen.ultimo_analisis
            ).toLocaleString("es-PE", {
                timeZone: "America/Lima",
                day: "2-digit",
                month: "2-digit",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit"
            });
        }

        colocarTexto(
            "est-ultimo-analisis",
            ultimaFecha
        );

        // ==================================
        // LEYENDA DEL GRÁFICO CIRCULAR
        // ==================================

        const imagenes = numero(tipos.imagenes);
        const videos = numero(tipos.videos);
        const total = imagenes + videos;

        const porcentajeImagenes = total
            ? Math.round(imagenes / total * 100)
            : 0;

        const porcentajeVideos = total
            ? Math.round(videos / total * 100)
            : 0;

        colocarTexto("est-total-dona", total);

        colocarTexto(
            "est-leyenda-imagenes",
            `${imagenes} (${porcentajeImagenes}%)`
        );

        colocarTexto(
            "est-leyenda-videos",
            `${videos} (${porcentajeVideos}%)`
        );

        // ==================================
        // RESUMEN INFERIOR
        // ==================================

        colocarTexto(
            "est-texto-resumen",
            `Se registraron ${resumen.total_analisis} análisis ` +
            `con un total de ${resumen.total_vehiculos} ` +
            `detecciones de vehículos: ` +
            `${numero(vehiculos.automoviles)} automóviles, ` +
            `${numero(vehiculos.motocicletas)} motocicletas, ` +
            `${numero(vehiculos.buses)} buses y ` +
            `${numero(vehiculos.camiones)} camiones.`
        );

        // ==================================
        // COMPROBAR LA BIBLIOTECA CHART.JS
        // ==================================

        if (typeof Chart === "undefined") {
            colocarTexto(
                estado,
                "Los datos cargaron, pero falta Chart.js."
            );
            return;
        }

        Chart.defaults.color = "#c5ddea";
        Chart.defaults.font.family = "Segoe UI, Arial, sans-serif";

        // ==================================
        // GRÁFICO 1: VEHÍCULOS
        // ==================================

        new Chart(
            document.getElementById("grafico-vehiculos"),
            {
                type: "bar",
                data: {
                    labels: [
                        "Automóviles",
                        "Motocicletas",
                        "Buses",
                        "Camiones"
                    ],
                    datasets: [{
                        label: "Detecciones",
                        data: [
                            numero(vehiculos.automoviles),
                            numero(vehiculos.motocicletas),
                            numero(vehiculos.buses),
                            numero(vehiculos.camiones)
                        ],
                        backgroundColor: [
                            "#12c9ed",
                            "#9958f4",
                            "#ffbc36",
                            "#00d7a0"
                        ],
                        borderRadius: 7
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { precision: 0 }
                        }
                    }
                }
            }
        );

        // ==================================
        // GRÁFICO 2: TIPOS DE ANÁLISIS
        // ==================================

        new Chart(
            document.getElementById("grafico-tipos"),
            {
                type: "doughnut",
                data: {
                    labels: ["Imágenes", "Videos"],
                    datasets: [{
                        data: [imagenes, videos],
                        backgroundColor: [
                            "#268cff",
                            "#9958f4"
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: "72%",
                    plugins: {
                        legend: { display: false }
                    }
                }
            }
        );

        // ==================================
        // GRÁFICO 3: ANÁLISIS POR DÍA
        // ==================================

        const fechas = dias.map(dia => {
            const partes = dia.fecha.split("-");
            return `${partes[2]}/${partes[1]}`;
        });

        const cantidades = dias.map(dia =>
            numero(dia.analisis)
        );

        new Chart(
            document.getElementById("grafico-dias"),
            {
                type: "line",
                data: {
                    labels: fechas,
                    datasets: [{
                        label: "Análisis realizados",
                        data: cantidades,
                        borderColor: "#12c9ed",
                        backgroundColor: "rgba(18,201,237,0.16)",
                        fill: true,
                        tension: 0.3,
                        borderWidth: 3,
                        pointRadius: 5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { precision: 0 }
                        }
                    }
                }
            }
        );

        colocarTexto(
            estado,
            "● Base de datos conectada · Estadísticas actualizadas"
        );

    } catch (error) {
        console.error("Error de estadísticas:", error);

        colocarTexto(
            estado,
            "Error al cargar las estadísticas. Revisa FastAPI."
        );
    }
}
