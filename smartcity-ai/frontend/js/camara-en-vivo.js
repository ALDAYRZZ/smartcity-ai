
"use strict";

// ==========================================
// SMARTCITY AI - CÁMARA EN VIVO CON YOLO
// ==========================================

// ELEMENTOS DE LA PÁGINA
const video = document.getElementById("camara");
const dispositivo = document.getElementById("dispositivo");

const btnIniciar = document.getElementById("iniciar");
const btnDetener = document.getElementById("detener");
const btnActualizar = document.getElementById("actualizar");

const pantallaVacia = document.getElementById("pantalla-vacia");
const sello = document.getElementById("sello");

const estado = document.getElementById("estado");
const mensaje = document.getElementById("mensaje");

const fuente = document.getElementById("fuente");
const conexion = document.getElementById("conexion");

const estadoIA = document.getElementById("estado-procesamiento");
const etiquetaIA = document.querySelector(".cam-etiqueta");
const resumenIA = document.querySelector(".cam-resultados-titulo span");

// CONFIGURACIÓN
const ENDPOINT_IA = "/analizar-camara";
const LADO_MAXIMO = 512;
const PAUSA_ENTRE_ANALISIS_MS = 350;

// VARIABLES
let transmision = null;
let iniciando = false;
let modoActual = "vehiculos";

let temporizadorIA = null;
let solicitudIA = null;
let iaDisponible = true;

// ==========================================
// 1. CREAR CAPA PARA LOS RECUADROS
// ==========================================

const pantalla = document.querySelector(".cam-pantalla");

const capaIA = document.createElement("canvas");
capaIA.className = "cam-capa-ia";
capaIA.style.objectFit = "contain";
capaIA.setAttribute("aria-hidden", "true");

pantalla.appendChild(capaIA);

const pincel = capaIA.getContext("2d");

// Lienzo para capturar fotogramas
const captura = document.createElement("canvas");
const contextoCaptura = captura.getContext("2d");

// ==========================================
// 2. MOSTRAR MENSAJES
// ==========================================

function mostrarMensaje(texto, esError = false) {
    mensaje.textContent = texto;
    mensaje.classList.toggle("error", esError);
}

// ==========================================
// 3. ESTADO DEL PROCESAMIENTO IA
// ==========================================

function mostrarEstadoIA(texto, activa = false) {

    if (estadoIA) {
        estadoIA.textContent = texto;
        estadoIA.classList.toggle(
            "cam-pendiente",
            !activa
        );
    }

    if (etiquetaIA) {
        etiquetaIA.textContent = activa
            ? `YOLO activo · ${
                modoActual === "personas"
                    ? "Personas"
                    : "Vehículos"
              }`
            : "Vista previa · Sin análisis IA";
    }

    if (resumenIA) {
        resumenIA.textContent = activa
            ? "Objetos visibles en el último fotograma analizado"
            : "Conteos pendientes de YOLO";
    }
}

// ==========================================
// 4. ESTADO DE LA CÁMARA
// ==========================================

function cambiarEstado(conectada) {

    estado.innerHTML = conectada
        ? '<span class="punto"></span> Conectada'
        : '<span class="punto"></span> Sin conectar';

    estado.classList.toggle("activo", conectada);

    conexion.textContent = conectada
        ? "Conectada"
        : "Desconectada";

    pantallaVacia.hidden = conectada;
    sello.hidden = !conectada;

    btnIniciar.disabled = conectada || iniciando;
    btnDetener.disabled = !conectada;
}

// ==========================================
// 5. BUSCAR CÁMARAS
// ==========================================

async function buscarCamaras() {

    if (!navigator.mediaDevices?.enumerateDevices) {
        mostrarMensaje(
            "Abre esta página desde http://127.0.0.1:8000.",
            true
        );
        return;
    }

    try {

        const anterior = dispositivo.value;

        const dispositivos =
            await navigator.mediaDevices.enumerateDevices();

        const camaras = dispositivos.filter(
            item => item.kind === "videoinput"
        );

        dispositivo.replaceChildren(
            new Option("Cámara predeterminada", "")
        );

        camaras.forEach((camara, indice) => {

            dispositivo.add(
                new Option(
                    camara.label || `Cámara ${indice + 1}`,
                    camara.deviceId
                )
            );
        });

        const existe = [...dispositivo.options].some(
            opcion => opcion.value === anterior
        );

        if (existe) {
            dispositivo.value = anterior;
        }

        if (!transmision) {
            mostrarMensaje(
                camaras.length
                    ? `Se encontraron ${camaras.length} cámara(s).`
                    : "No se encontraron cámaras disponibles."
            );
        }

    } catch (error) {

        mostrarMensaje(
            "No se pudieron buscar cámaras: " + error.message,
            true
        );
    }
}

// ==========================================
// 6. REINICIAR CONTADORES
// ==========================================

function reiniciarContadores() {

    const categorias = [
        "personas",
        "automoviles",
        "motocicletas",
        "buses",
        "camiones"
    ];

    categorias.forEach(id => {

        const elemento = document.getElementById(id);

        if (elemento) {
            elemento.textContent = "—";
        }
    });
}

// ==========================================
// 7. LIMPIAR RECUADROS
// ==========================================

function limpiarRecuadros() {

    pincel.clearRect(
        0,
        0,
        capaIA.width,
        capaIA.height
    );
}

// ==========================================
// 8. DETENER ANÁLISIS YOLO
// ==========================================

function detenerAnalisis() {

    if (temporizadorIA !== null) {
        clearTimeout(temporizadorIA);
        temporizadorIA = null;
    }

    if (solicitudIA) {
        solicitudIA.abort();
        solicitudIA = null;
    }

    limpiarRecuadros();
    reiniciarContadores();

    mostrarEstadoIA("Sin análisis IA");
}

// ==========================================
// 9. DETENER CÁMARA
// ==========================================

function detenerCamara() {

    detenerAnalisis();

    if (transmision) {

        transmision.getTracks().forEach(
            pista => pista.stop()
        );

        transmision = null;
    }

    video.pause();
    video.srcObject = null;

    fuente.textContent = "Sin seleccionar";

    cambiarEstado(false);
}

// ==========================================
// 10. ACTUALIZAR CONTADORES
// ==========================================

function actualizarContadores(conteos = {}) {

    const categorias = [
        "personas",
        "automoviles",
        "motocicletas",
        "buses",
        "camiones"
    ];

    categorias.forEach(id => {

        const elemento = document.getElementById(id);

        if (elemento) {
            elemento.textContent = String(
                conteos[id] ?? 0
            );
        }
    });
}

// ==========================================
// 11. DIBUJAR DETECCIONES DE YOLO
// ==========================================

// FastAPI devolverá coordenadas normalizadas
// entre 0 y 1 para cada recuadro.

function dibujarDetecciones(detecciones = []) {

    const ancho = video.videoWidth;
    const alto = video.videoHeight;

    if (!ancho || !alto) return;

    if (
        capaIA.width !== ancho ||
        capaIA.height !== alto
    ) {
        capaIA.width = ancho;
        capaIA.height = alto;
    }

    limpiarRecuadros();

    pincel.lineWidth = Math.max(
        2,
        Math.round(ancho / 300)
    );

    pincel.font =
        `${Math.max(13, Math.round(ancho / 52))}px Arial`;

    detecciones.forEach(deteccion => {

        if (
            !Array.isArray(deteccion.caja) ||
            deteccion.caja.length !== 4
        ) {
            return;
        }

        const [x1, y1, x2, y2] =
            deteccion.caja.map(Number);

        if (
            ![x1, y1, x2, y2].every(Number.isFinite)
        ) {
            return;
        }

        const x = Math.max(
            0,
            Math.min(ancho, x1 * ancho)
        );

        const y = Math.max(
            0,
            Math.min(alto, y1 * alto)
        );

        const derecha = Math.max(
            0,
            Math.min(ancho, x2 * ancho)
        );

        const abajo = Math.max(
            0,
            Math.min(alto, y2 * alto)
        );

        const w = derecha - x;
        const h = abajo - y;

        if (w <= 0 || h <= 0) return;

        const etiqueta = String(
            deteccion.etiqueta || "Objeto"
        );

        const confianza = Number(
            deteccion.confianza
        );

        const texto = Number.isFinite(confianza)
            ? `${etiqueta} ${Math.round(confianza * 100)}%`
            : etiqueta;

        // Recuadro celeste
        pincel.strokeStyle = "#14c9eb";
        pincel.strokeRect(x, y, w, h);

        // Fondo de etiqueta
        const anchoTexto =
            pincel.measureText(texto).width + 12;

        const altoTexto = Math.max(
            21,
            Math.round(ancho / 35)
        );

        const arriba = y >= altoTexto
            ? y - altoTexto
            : y;

        pincel.fillStyle = "#06283b";

        pincel.fillRect(
            x,
            arriba,
            Math.min(anchoTexto, ancho - x),
            altoTexto
        );

        // Texto
        pincel.fillStyle = "#ffffff";

        pincel.fillText(
            texto,
            x + 6,
            arriba + altoTexto - 6
        );
    });
}

// ==========================================
// 12. PROGRAMAR ANÁLISIS
// ==========================================

function programarAnalisis(
    retraso = PAUSA_ENTRE_ANALISIS_MS
) {

    if (
        !transmision ||
        !iaDisponible ||
        temporizadorIA !== null
    ) {
        return;
    }

    temporizadorIA = setTimeout(() => {

        temporizadorIA = null;

        analizarFotograma();

    }, retraso);
}

// ==========================================
// 13. ENVIAR FOTOGRAMA A FASTAPI
// ==========================================

async function analizarFotograma() {

    if (
        !transmision ||
        !iaDisponible ||
        solicitudIA
    ) {
        return;
    }

    if (
        video.readyState <
            HTMLMediaElement.HAVE_CURRENT_DATA ||
        !video.videoWidth
    ) {
        programarAnalisis(500);
        return;
    }

    // Reducir tamaño para la CPU
    const escala = Math.min(
        1,
        LADO_MAXIMO / Math.max(
            video.videoWidth,
            video.videoHeight
        )
    );

    captura.width = Math.round(
        video.videoWidth * escala
    );

    captura.height = Math.round(
        video.videoHeight * escala
    );

    contextoCaptura.drawImage(
        video,
        0,
        0,
        captura.width,
        captura.height
    );

    const modoSolicitado = modoActual;
    const flujoSolicitado = transmision;

    const imagen = await new Promise(resolve => {
        captura.toBlob(
            resolve,
            "image/jpeg",
            0.72
        );
    });

    if (
        !imagen ||
        transmision !== flujoSolicitado
    ) {
        return;
    }

    const datos = new FormData();

    datos.append(
        "archivo",
        imagen,
        "fotograma.jpg"
    );

    datos.append(
        "modo",
        modoSolicitado
    );

    const controlador = new AbortController();

    solicitudIA = controlador;

    try {

        const respuesta = await fetch(
            ENDPOINT_IA,
            {
                method: "POST",
                body: datos,
                signal: controlador.signal
            }
        );

        if (!respuesta.ok) {

            const error = new Error(
                `Servidor respondió ${respuesta.status}`
            );

            error.status = respuesta.status;
            throw error;
        }

        const resultado = await respuesta.json();

        // Ignorar respuestas antiguas al cambiar modo
        if (
            controlador.signal.aborted ||
            modoActual !== modoSolicitado ||
            transmision !== flujoSolicitado
        ) {
            return;
        }

        dibujarDetecciones(
            resultado.detecciones ?? []
        );

        actualizarContadores(
            resultado.conteos ?? {}
        );

        mostrarEstadoIA("YOLO activo", true);

        mostrarMensaje(
            "Cámara conectada. " +
            "Detección de YOLO en funcionamiento."
        );

    } catch (error) {

        if (error.name === "AbortError") return;

        limpiarRecuadros();

        mostrarEstadoIA("Sin análisis IA");

        if (
            error.status === 404 ||
            error.status === 405
        ) {

            // La ruta aún no se creó en FastAPI
            iaDisponible = false;

            mostrarMensaje(
                "La cámara funciona, pero falta " +
                "conectar /analizar-camara en FastAPI.",
                true
            );

        } else {

            mostrarMensaje(
                "No se pudo analizar la imagen: " +
                error.message,
                true
            );
        }

    } finally {

        if (solicitudIA === controlador) {
            solicitudIA = null;
        }

        if (
            transmision === flujoSolicitado &&
            iaDisponible
        ) {
            programarAnalisis(350);
        }
    }
}

// ==========================================
// 14. INICIAR CÁMARA
// ==========================================

async function iniciarCamara() {

    if (iniciando || transmision) return;

    if (!navigator.mediaDevices?.getUserMedia) {

        mostrarMensaje(
            "Chrome no permite acceder a la cámara.",
            true
        );

        return;
    }

    iniciando = true;
    btnIniciar.disabled = true;

    mostrarMensaje("Conectando con la cámara...");

    let nuevoFlujo = null;

    try {

        const idSeleccionado = dispositivo.value;

        const configuracion = {

            audio: false,

            video: idSeleccionado
                ? {
                    deviceId: {
                        exact: idSeleccionado
                    }
                }
                : {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 15 }
                }
        };

        nuevoFlujo =
            await navigator.mediaDevices.getUserMedia(
                configuracion
            );

        transmision = nuevoFlujo;
        video.srcObject = nuevoFlujo;

        await video.play();

        const pista =
            nuevoFlujo.getVideoTracks()[0];

        fuente.textContent =
            pista?.label || "Cámara conectada";

        iniciando = false;

        cambiarEstado(true);

        // Activar análisis cuando exista el backend
        iaDisponible = true;

        reiniciarContadores();

        programarAnalisis(400);

        mostrarMensaje(
            "Cámara conectada. " +
            "Preparando el análisis de YOLO..."
        );

        // Desconexión del dispositivo
        pista?.addEventListener(
            "ended",
            () => {

                if (transmision === nuevoFlujo) {

                    detenerCamara();

                    mostrarMensaje(
                        "La cámara se desconectó."
                    );
                }
            },
            { once: true }
        );

        await buscarCamaras();

    } catch (error) {

        if (nuevoFlujo) {

            nuevoFlujo.getTracks().forEach(
                pista => pista.stop()
            );
        }

        transmision = null;
        video.srcObject = null;

        iniciando = false;

        cambiarEstado(false);

        const errores = {

            NotAllowedError:
                "Permiso denegado. Autoriza la cámara en Chrome.",

            NotFoundError:
                "No se encontró la cámara seleccionada.",

            NotReadableError:
                "La cámara está ocupada por otra aplicación.",

            OverconstrainedError:
                "La cámara seleccionada no está disponible."
        };

        mostrarMensaje(
            errores[error.name] ||
            "No se pudo iniciar: " + error.message,
            true
        );
    }
}

// ==========================================
// 15. BOTONES
// ==========================================

btnIniciar.addEventListener(
    "click",
    iniciarCamara
);

btnDetener.addEventListener(
    "click",
    () => {

        detenerCamara();

        mostrarMensaje(
            "Cámara detenida correctamente."
        );
    }
);

btnActualizar.addEventListener(
    "click",
    buscarCamaras
);

// CAMBIAR DISPOSITIVO
dispositivo.addEventListener(
    "change",
    () => {

        if (transmision) {

            detenerCamara();

            mostrarMensaje(
                "Cambiaste de dispositivo. " +
                "Pulsa Iniciar cámara."
            );
        }
    }
);

// ==========================================
// 16. CAMBIAR MODO
// ==========================================

// El HTML ya tiene los botones
// modo-vehiculos y modo-personas.
// Este evento lo envía su script.

window.addEventListener(
    "smartcity:cambio-modo",
    evento => {

        const nuevo = evento.detail?.modo;

        if (
            nuevo !== "personas" &&
            nuevo !== "vehiculos"
        ) {
            return;
        }

        modoActual = nuevo;

        detenerAnalisis();

        iaDisponible = true;

        if (transmision) {

            mostrarMensaje(
                `Modo ${modoActual} seleccionado. ` +
                "Reiniciando el análisis..."
            );

            programarAnalisis(250);
        }
    }
);

// ==========================================
// 17. SALIR Y RECONOCER DISPOSITIVOS
// ==========================================

window.addEventListener(
    "pagehide",
    detenerCamara
);

navigator.mediaDevices?.addEventListener?.(
    "devicechange",
    buscarCamaras
);

// ==========================================
// 18. INICIALIZACIÓN
// ==========================================

cambiarEstado(false);
reiniciarContadores();
mostrarEstadoIA("Sin análisis IA");
buscarCamaras();
