
const botonMovil = document.getElementById("menuMovil");
const navegacion = document.getElementById("navegacion");

// Abrir o cerrar el menú en celulares
botonMovil.addEventListener("click", () => {
    const abierto = navegacion.classList.toggle("mostrar");

    botonMovil.setAttribute("aria-expanded", abierto);
    botonMovil.textContent = abierto ? "✕" : "☰";
});

// Abrir submenús al hacer clic
const botonesMenu = document.querySelectorAll(".boton-menu");

botonesMenu.forEach((boton) => {
    boton.setAttribute("aria-expanded", "false");

    boton.addEventListener("click", () => {
        const contenedor = boton.parentElement;
        const abierto = contenedor.classList.contains("abierto");

        document.querySelectorAll(".desplegable").forEach((menu) => {
            menu.classList.remove("abierto");
            menu.querySelector(".boton-menu")
                .setAttribute("aria-expanded", "false");
        });

        if (!abierto) {
            contenedor.classList.add("abierto");
            boton.setAttribute("aria-expanded", "true");
        }
    });
});

// Cerrar submenús al pulsar Escape
document.addEventListener("keydown", (evento) => {
    if (evento.key === "Escape") {
        document.querySelectorAll(".desplegable").forEach((menu) => {
            menu.classList.remove("abierto");
        });
    }
});
