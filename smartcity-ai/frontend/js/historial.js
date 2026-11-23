
/* ==========================================
   SMARTCITY AI - HISTORIAL
   Conexión con FastAPI y SQLite
   ========================================== */

document.addEventListener("DOMContentLoaded", () => {

    // ELEMENTOS DE LA PÁGINA

    const estadoBD = document.getElementById("estado-bd");

    const totalAnalisis = document.getElementById("total-analisis");
    const totalImagenes = document.getElementById("total-imagenes");
    const totalVideos = document.getElementById("total-videos");
    const ultimoAnalisis = document.getElementById("ultimo-analisis");

    const tabla = document.getElementById("tabla-historial");
    const sinRegistros = document.getElementById("sin-registros");
    const cantidadRegistros = document.getElementById("cantidad-registros");

    const buscar = document.getElementById("buscar");
    const filtroTipo = document.getElementById("filtro-tipo");
    const filtroFecha = document.getElementById("filtro-fecha");
    const botonFiltrar = document.getElementById("aplicar-filtros");

    const modal = document.getElementById("modal-detalles");
    const cerrarModal = document.getElementById("cerrar-modal");
    const contenidoDetalles = document.getElementById("contenido-detalles");

    // REGISTROS QUE RECIBIMOS DE SQLITE
    let registros = [];


    // ======================================
    // FORMATEAR FECHA
    // ======================================

    function formatearFecha(fecha) {

        if (!fecha) {
            return "Sin registros";
        }

        const valor = new Date(fecha);

        if (Number.isNaN(valor.getTime())) {
            return "Fecha no disponible";
        }

        return valor.toLocaleString("es-PE", {
            day: "2-digit",
            month: "2-digit",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit"
        });
    }


    // ======================================
    // CONECTAR CON FASTAPI Y SQLITE
    // ======================================

    async function cargarHistorial() {

        estadoBD.textContent = "Conectando...";

        try {

            const respuesta = await fetch("/historial", {
                method: "GET",
                cache: "no-store"
            });

            if (!respuesta.ok) {
                throw new Error(
                    "Error HTTP " + respuesta.status
                );
            }

            const datos = await respuesta.json();

            // Verificar que recibimos los datos esperados
            if (!datos.resumen ||
                !Array.isArray(datos.registros)) {
                throw new Error("Respuesta de datos incorrecta");
            }

            registros = datos.registros;

            // Actualizar las cuatro tarjetas
            totalAnalisis.textContent = datos.resumen.total;
            totalImagenes.textContent = datos.resumen.imagenes;
            totalVideos.textContent = datos.resumen.videos;

            ultimoAnalisis.textContent = formatearFecha(
                datos.resumen.ultimo
            );

            // Estado de conexión
            estadoBD.textContent = "● Base de datos conectada";

            // Mostrar los análisis
            aplicarFiltros();

            console.log(
                "SmartCity AI: historial cargado correctamente",
                registros
            );

        } catch (error) {

            console.error(
                "Error al cargar el historial:",
                error
            );

            estadoBD.textContent = "Error de conexión";

            cantidadRegistros.textContent =
                "No se pudieron cargar los registros";

        }
    }


    // ======================================
    // FILTRAR LOS REGISTROS
    // ======================================

    function aplicarFiltros() {

        const texto = buscar.value
            .toLowerCase()
            .trim();

        const tipo = filtroTipo.value;
        const fecha = filtroFecha.value;

        const ahora = new Date();

        const filtrados = registros.filter(registro => {

            // Buscar por nombre de archivo
            const nombre = String(
                registro.archivo || ""
            ).toLowerCase();

            if (!nombre.includes(texto)) {
                return false;
            }

            // Filtrar por imagen o video
            if (tipo !== "todos" &&
                registro.tipo !== tipo) {
                return false;
            }

            // Filtrar por fecha
            if (fecha !== "todas") {

                const fechaRegistro = new Date(registro.fecha);

                if (Number.isNaN(fechaRegistro.getTime())) {
                    return false;
                }

                // Inicio del día actual
                const inicioHoy = new Date(
                    ahora.getFullYear(),
                    ahora.getMonth(),
                    ahora.getDate()
                );

                if (fecha === "hoy") {

                    if (fechaRegistro < inicioHoy ||
                        fechaRegistro > ahora) {
                        return false;
                    }
                }

                if (fecha === "7dias") {

                    const inicioPeriodo = new Date(
                        inicioHoy
                    );

                    inicioPeriodo.setDate(
                        inicioPeriodo.getDate() - 6
                    );

                    if (fechaRegistro < inicioPeriodo ||
                        fechaRegistro > ahora) {
                        return false;
                    }
                }
            }

            return true;
        });

        mostrarRegistros(filtrados);
    }


    // ======================================
    // MOSTRAR REGISTROS EN LA TABLA
    // ======================================

    function mostrarRegistros(lista) {

        // Limpiar contenido anterior
        tabla.replaceChildren();

        cantidadRegistros.textContent =
            `Mostrando ${lista.length} registros`;

        // Sin registros
        if (lista.length === 0) {

            sinRegistros.hidden = false;

            const tituloVacio = sinRegistros.querySelector("h3");
            const textoVacio = sinRegistros.querySelector("p");

            if (registros.length === 0) {
                tituloVacio.textContent =
                    "Aún no hay análisis guardados";

                textoVacio.textContent =
                    "Los resultados de imágenes y videos aparecerán aquí.";
            } else {
                tituloVacio.textContent =
                    "No se encontraron resultados";

                textoVacio.textContent =
                    "Prueba con otros filtros de búsqueda.";
            }

            return;
        }

        sinRegistros.hidden = true;

        // Crear una fila por cada análisis
        lista.forEach(registro => {

            const fila = document.createElement("tr");

            // ARCHIVO
            const celdaArchivo = document.createElement("td");
            celdaArchivo.textContent =
                registro.archivo || "Archivo sin nombre";

            // TIPO
            const celdaTipo = document.createElement("td");
            celdaTipo.textContent =
                registro.tipo === "imagen"
                    ? "Imagen"
                    : registro.tipo === "video"
                        ? "Video"
                        : "Otro";

            // FECHA
            const celdaFecha = document.createElement("td");
            celdaFecha.textContent =
                formatearFecha(registro.fecha);

            // DETECCIONES
            const celdaTotal = document.createElement("td");

            const numero = Number(registro.total || 0);

            celdaTotal.textContent =
                `${numero} vehículos`;

            // ESTADO
            const celdaEstado = document.createElement("td");
            celdaEstado.textContent =
                registro.estado || "Completado";

            // BOTÓN VER DETALLES
            const celdaDetalles = document.createElement("td");

            const boton = document.createElement("button");
            boton.type = "button";
            boton.className = "boton-detalles";
            boton.textContent = "Ver detalles";

            boton.addEventListener("click", () => {
                abrirDetalles(registro);
            });

            celdaDetalles.appendChild(boton);

            // AGREGAR CELDAS A LA FILA
            fila.append(
                celdaArchivo,
                celdaTipo,
                celdaFecha,
                celdaTotal,
                celdaEstado,
                celdaDetalles
            );

            tabla.appendChild(fila);
        });
    }


    // ======================================
    // MOSTRAR DETALLES DE UN ANÁLISIS
    // ======================================

    function abrirDetalles(registro) {

        contenidoDetalles.replaceChildren();

        function agregarDato(etiqueta, valor) {

            const parrafo = document.createElement("p");

            const negrita = document.createElement("strong");
            negrita.textContent = etiqueta + ": ";

            const texto = document.createTextNode(
                String(valor)
            );

            parrafo.append(negrita, texto);

            contenidoDetalles.appendChild(parrafo);
        }

        agregarDato(
            "Archivo",
            registro.archivo || "Sin nombre"
        );

        agregarDato(
            "Tipo",
            registro.tipo === "imagen" ? "Imagen" : "Video"
        );

        agregarDato(
            "Fecha",
            formatearFecha(registro.fecha)
        );

        agregarDato(
            "Total de vehículos",
            registro.total || 0
        );

        // DESGLOSE DE VEHÍCULOS
        const conteos = registro.conteos || {};

        agregarDato(
            "Automóviles",
            conteos.automoviles || 0
        );

        agregarDato(
            "Motocicletas",
            conteos.motocicletas || 0
        );

        agregarDato(
            "Buses",
            conteos.buses || 0
        );

        agregarDato(
            "Camiones",
            conteos.camiones || 0
        );

        agregarDato(
            "Estado",
            registro.estado || "Completado"
        );

        // ENLACE AL VIDEO PROCESADO
        if (registro.tipo === "video" &&
            registro.resultado_url) {

            try {

                const url = new URL(
                    registro.resultado_url,
                    window.location.href
                );

                // Solo admitir archivos del servidor local
                if (
                    ["127.0.0.1", "localhost"].includes(url.hostname) &&
                    url.port === "8000" &&
                    url.pathname.startsWith("/resultados/")
                ) {

                    const enlace = document.createElement("a");

                    enlace.href = url.href;
                    enlace.target = "_blank";
                    enlace.rel = "noopener noreferrer";

                    enlace.textContent = "▶ Abrir video analizado";

                    const parrafo = document.createElement("p");
                    parrafo.appendChild(enlace);

                    contenidoDetalles.appendChild(parrafo);
                }

            } catch (error) {
                console.warn(
                    "Enlace de video no disponible",
                    error
                );
            }
        }

        modal.hidden = false;
    }


    // ======================================
    // CERRAR VENTANA DE DETALLES
    // ======================================

    function cerrarDetalles() {
        modal.hidden = true;
    }

    cerrarModal.addEventListener(
        "click",
        cerrarDetalles
    );

    // Cerrar al hacer clic fuera de la ventana
    modal.addEventListener("click", evento => {

        if (evento.target === modal) {
            cerrarDetalles();
        }
    });

    // Cerrar con tecla ESC
    document.addEventListener("keydown", evento => {

        if (evento.key === "Escape") {
            cerrarDetalles();
        }
    });


    // ======================================
    // EVENTOS DE LOS FILTROS
    // ======================================

    botonFiltrar.addEventListener(
        "click",
        aplicarFiltros
    );

    buscar.addEventListener("input", aplicarFiltros);

    filtroTipo.addEventListener("change", aplicarFiltros);

    filtroFecha.addEventListener("change", aplicarFiltros);


    // ======================================
    // INICIAR HISTORIAL
    // ======================================

    cargarHistorial();

});
