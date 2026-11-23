
const archivoImagen = document.getElementById("archivoImagen");
const zonaCarga = document.getElementById("zonaCarga");
const nombreArchivo = document.getElementById("nombreArchivo");
const botonAnalizar = document.getElementById("botonAnalizar");
const botonLimpiar = document.getElementById("botonLimpiar");
const imagenVista = document.getElementById("imagenVista");
const vistaVacia = document.getElementById("vistaVacia");
const estadoAnalisis = document.getElementById("estadoAnalisis");
const aviso = document.getElementById("aviso");
const etiquetaResultado = document.getElementById("etiquetaResultado");
const descargarResultado = document.getElementById("descargarResultado");

const API = "http://127.0.0.1:8000/analizar-imagen";

let archivoSeleccionado = null;
let imagenTemporal = null;

function cambiarTexto(id, valor) {
    const elemento = document.getElementById(id);
    if (elemento) elemento.textContent = valor;
}

function reiniciarResultados() {
    cambiarTexto("totalAutos", "—");
    cambiarTexto("totalMotos", "—");
    cambiarTexto("totalBuses", "—");
    cambiarTexto("totalCamiones", "—");
    cambiarTexto("totalMotokares", "—");
    cambiarTexto("totalMotocargueros", "—");
    cambiarTexto("totalVehiculos", "—");

    etiquetaResultado.textContent = "Sin analizar";

    if (descargarResultado) {
        descargarResultado.hidden = true;
        descargarResultado.removeAttribute("href");
    }
}

function limpiar() {
    archivoImagen.value = "";
    archivoSeleccionado = null;

    if (imagenTemporal) {
        URL.revokeObjectURL(imagenTemporal);
        imagenTemporal = null;
    }

    imagenVista.removeAttribute("src");
    imagenVista.hidden = true;
    vistaVacia.hidden = false;

    nombreArchivo.textContent = "Ningún archivo seleccionado";
    estadoAnalisis.textContent = "Esperando imagen";
    aviso.textContent = "Selecciona una imagen para comenzar.";

    botonAnalizar.disabled = true;
    reiniciarResultados();
}

function cargarImagen(archivo) {
    if (!archivo) return;

    const permitidos = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];

    if (!permitidos.includes(archivo.type)) {
        limpiar();
        aviso.textContent = "Selecciona un archivo JPG, PNG o WebP.";
        return;
    }

    if (archivo.size > 10 * 1024 * 1024) {
        limpiar();
        aviso.textContent = "La imagen no puede superar los 10 MB.";
        return;
    }

    if (imagenTemporal) {
        URL.revokeObjectURL(imagenTemporal);
    }

    archivoSeleccionado = archivo;
    imagenTemporal = URL.createObjectURL(archivo);

    imagenVista.src = imagenTemporal;
    imagenVista.hidden = false;
    vistaVacia.hidden = true;

    nombreArchivo.textContent = archivo.name;
    estadoAnalisis.textContent = "Imagen cargada";
    aviso.textContent = "Imagen lista para analizar.";

    botonAnalizar.disabled = false;
    reiniciarResultados();
}

// Seleccionar imagen desde Windows
archivoImagen.addEventListener("change", () => {
    cargarImagen(archivoImagen.files[0]);
});

// Arrastrar imagen
zonaCarga.addEventListener("dragover", (evento) => {
    evento.preventDefault();
    zonaCarga.classList.add("arrastrando");
});

zonaCarga.addEventListener("dragleave", () => {
    zonaCarga.classList.remove("arrastrando");
});

zonaCarga.addEventListener("drop", (evento) => {
    evento.preventDefault();
    zonaCarga.classList.remove("arrastrando");
    cargarImagen(evento.dataTransfer.files[0]);
});

// Limpiar
botonLimpiar.addEventListener("click", limpiar);

// ANALIZAR CON YOLO
botonAnalizar.addEventListener("click", async () => {

    if (!archivoSeleccionado) {
        aviso.textContent = "Primero selecciona una imagen.";
        return;
    }

    const formulario = new FormData();
    formulario.append("archivo", archivoSeleccionado);

    botonAnalizar.disabled = true;
    botonAnalizar.textContent = "Analizando...";
    estadoAnalisis.textContent = "Procesando con YOLO...";
    aviso.textContent = "Analizando la fotografía.";
    reiniciarResultados();

    try {
        const respuesta = await fetch(API, {
            method: "POST",
            body: formulario,
        });

        const datos = await respuesta.json();

        if (!respuesta.ok) {
            throw new Error(
                datos.detail || "No se pudo analizar la imagen."
            );
        }

        // Mostrar foto procesada
        imagenVista.src = datos.imagen_resultado;
        imagenVista.hidden = false;
        vistaVacia.hidden = true;

        // Mostrar cantidades reales
        cambiarTexto("totalAutos", datos.conteos.automoviles);
        cambiarTexto("totalMotos", datos.conteos.motocicletas);
        cambiarTexto("totalBuses", datos.conteos.buses);
        cambiarTexto("totalCamiones", datos.conteos.camiones);
        cambiarTexto("totalVehiculos", datos.total);

        // Motokares y motocargueros siguen pendientes
        cambiarTexto("totalMotokares", "—");
        cambiarTexto("totalMotocargueros", "—");

        estadoAnalisis.textContent = "Análisis terminado";
        etiquetaResultado.textContent = "Completado";

        aviso.textContent =
            `Se detectaron ${datos.total} vehículos. ` +
            "Los motokares y motocargueros aún no se reconocen por separado.";

        // Habilitar descarga
        if (descargarResultado) {
            descargarResultado.href = datos.imagen_resultado;
            descargarResultado.hidden = false;
        }

    } catch (error) {
        etiquetaResultado.textContent = "Error";
        estadoAnalisis.textContent = "No se pudo analizar";

        aviso.textContent =
            error instanceof TypeError
                ? "No se pudo conectar con FastAPI. Verifica que el servidor esté encendido."
                : error.message;
    } finally {
        botonAnalizar.disabled = false;
        botonAnalizar.textContent = "Analizar imagen";
    }
});
