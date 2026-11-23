
const archivoVideo = document.getElementById("archivoVideo");
const reproductorVideo = document.getElementById("reproductorVideo");
const videoVacio = document.getElementById("videoVacio");

const botonAnalizarVideo = document.getElementById("botonAnalizarVideo");
const botonLimpiarVideo = document.getElementById("botonLimpiarVideo");
const descargarVideo = document.getElementById("descargarVideo");

const nombreVideo = document.getElementById("nombreVideo");
const mensajeVideo = document.getElementById("mensajeVideo");
const estadoVideo = document.getElementById("estadoVideo");
const estadoProceso = document.getElementById("estadoProceso");
const duracionVideo = document.getElementById("duracionVideo");
const totalVideo = document.getElementById("totalVideo");

const API_VIDEO = "http://127.0.0.1:8000/analizar-video";

let videoSeleccionado = null;
let urlTemporal = null;
let urlResultado = null;
let procesando = false;

function escribir(id, valor) {
    const elemento = document.getElementById(id);
    if (elemento) {
        elemento.textContent = valor;
    }
}

function reiniciarResultadosVideo() {
    escribir("videoAutos", "—");
    escribir("videoMotos", "—");
    escribir("videoBuses", "—");
    escribir("videoCamiones", "—");
    escribir("videoMotokares", "—");
    escribir("videoMotocargueros", "—");

    totalVideo.textContent = "—";
    descargarVideo.hidden = true;
    descargarVideo.removeAttribute("href");
    urlResultado = null;
}

// SELECCIONAR VIDEO
archivoVideo.addEventListener("change", () => {
    const archivo = archivoVideo.files[0];

    if (!archivo) return;

    const extension = archivo.name.split(".").pop().toLowerCase();

    if (!["mp4", "webm"].includes(extension)) {
        mensajeVideo.textContent = "Selecciona un MP4 o WebM.";
        archivoVideo.value = "";
        return;
    }

    if (archivo.size > 30 * 1024 * 1024) {
        mensajeVideo.textContent = "El video no debe superar 30 MB.";
        archivoVideo.value = "";
        return;
    }

    if (urlTemporal) {
        URL.revokeObjectURL(urlTemporal);
    }

    videoSeleccionado = archivo;
    urlTemporal = URL.createObjectURL(archivo);

    reproductorVideo.src = urlTemporal;
    reproductorVideo.hidden = false;
    videoVacio.hidden = true;
    reproductorVideo.load();

    nombreVideo.textContent = archivo.name;
    estadoVideo.textContent = "Video cargado";
    estadoProceso.textContent = "Listo";
    mensajeVideo.textContent = "Presiona Analizar video para comenzar.";

    botonAnalizarVideo.disabled = false;
    reiniciarResultadosVideo();
});

reproductorVideo.addEventListener("loadedmetadata", () => {
    if (Number.isFinite(reproductorVideo.duration)) {
        duracionVideo.textContent =
            reproductorVideo.duration.toFixed(1) + " s";
    }
});

// LIMPIAR
botonLimpiarVideo.addEventListener("click", () => {
    if (procesando) return;

    videoSeleccionado = null;
    archivoVideo.value = "";

    reproductorVideo.pause();
    reproductorVideo.removeAttribute("src");
    reproductorVideo.load();

    reproductorVideo.hidden = true;
    videoVacio.hidden = false;

    if (urlTemporal) {
        URL.revokeObjectURL(urlTemporal);
        urlTemporal = null;
    }

    nombreVideo.textContent = "Ningún archivo seleccionado";
    mensajeVideo.textContent = "Selecciona un video para comenzar.";

    estadoVideo.textContent = "Sin video";
    estadoProceso.textContent = "Pendiente";
    duracionVideo.textContent = "—";

    botonAnalizarVideo.disabled = true;
    reiniciarResultadosVideo();
});

// ANALIZAR CON YOLO
botonAnalizarVideo.addEventListener("click", async () => {
    if (!videoSeleccionado || procesando) return;

    procesando = true;
    botonAnalizarVideo.disabled = true;
    botonLimpiarVideo.disabled = true;
    botonAnalizarVideo.textContent = "Analizando...";

    estadoVideo.textContent = "Procesando";
    estadoProceso.textContent = "Ejecutando YOLO";
    mensajeVideo.textContent =
        "La IA está analizando el video. No cierres esta página.";

    reiniciarResultadosVideo();

    const formulario = new FormData();
    formulario.append("archivo", videoSeleccionado);

    try {
        const respuesta = await fetch(API_VIDEO, {
            method: "POST",
            body: formulario,
        });

        const datos = await respuesta.json();

        if (!respuesta.ok) {
            throw new Error(
                datos.detail || "Error al analizar el video."
            );
        }

        // Mostrar video con recuadros
        urlResultado = datos.video_url;

        reproductorVideo.pause();
        reproductorVideo.src = urlResultado;
        reproductorVideo.load();

        reproductorVideo.hidden = false;
        videoVacio.hidden = true;

        // Mostrar detecciones
        escribir("videoAutos", datos.conteos.automoviles);
        escribir("videoMotos", datos.conteos.motocicletas);
        escribir("videoBuses", datos.conteos.buses);
        escribir("videoCamiones", datos.conteos.camiones);

        // Categorías pendientes de entrenamiento
        escribir("videoMotokares", "—");
        escribir("videoMotocargueros", "—");

        totalVideo.textContent = datos.total;

        estadoVideo.textContent = "Completado";
        estadoProceso.textContent = "Análisis terminado";
        duracionVideo.textContent =
            datos.duracion_analizada + " s procesados";

        mensajeVideo.textContent =
            `Se registraron ${datos.total} detecciones ` +
            `en ${datos.fotogramas_analizados} fotogramas analizados.`;

        descargarVideo.href = urlResultado;
        descargarVideo.hidden = false;

    } catch (error) {
        estadoVideo.textContent = "Error";
        estadoProceso.textContent = "No completado";

        mensajeVideo.textContent =
            error instanceof TypeError
                ? "No hay conexión con FastAPI. Comprueba el servidor."
                : error.message;
    } finally {
        procesando = false;
        botonAnalizarVideo.disabled = false;
        botonLimpiarVideo.disabled = false;
        botonAnalizarVideo.textContent = "Analizar video";
    }
});

// DESCARGAR EL VIDEO PROCESADO
descargarVideo.addEventListener("click", async (evento) => {
    evento.preventDefault();

    if (!urlResultado) return;

    try {
        const respuesta = await fetch(urlResultado);

        if (!respuesta.ok) {
            throw new Error("Error al descargar.");
        }

        const contenido = await respuesta.blob();
        const urlDescarga = URL.createObjectURL(contenido);

        const enlace = document.createElement("a");
        enlace.href = urlDescarga;
        enlace.download = "smartcity-video-analizado.mp4";

        document.body.appendChild(enlace);
        enlace.click();
        enlace.remove();

        setTimeout(() => {
            URL.revokeObjectURL(urlDescarga);
        }, 1000);

    } catch (error) {
        mensajeVideo.textContent =
            "No se pudo descargar el video procesado.";
    }
});
