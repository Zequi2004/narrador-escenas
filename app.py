import os
import json
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

# Cargar variables de entorno si existe un archivo .env local
load_dotenv()

# =====================================================================
# 1. ESQUEMA DE SALIDA ESTRUCTURADA (PYDANTIC)
# =====================================================================
class ResultadoEscena(BaseModel):
    escena_narrada: str = Field(
        description="La escena narrativa final enriquecida en formato Markdown. Debe conservar intacto el 100% de las líneas de diálogo originales."
    )
    escenario_inventado: str = Field(
        description="Breve resumen de qué elementos de utilería, atmósfera y actitudes dedujo e inventó la IA para rellenar los huecos de la descripción."
    )
    analisis_subtexto: str = Field(
        description="Análisis en viñetas explicando las decisiones de lenguaje corporal, microexpresiones y ritmo aplicadas."
    )


# =====================================================================
# 2. SYSTEM PROMPT CON MOTOR DE AUTO-DEDUCCIÓN ("MODO VAGO")
# =====================================================================
def construir_system_prompt(
    nivel_entorno: str,
    nivel_tension: str,
    punto_de_vista: str,
    formato_dialogo: str,
    genero: str,
    clima_hora: str
) -> str:
    guia_entorno = {
        "Bajo": "Descripciones breves, ágiles y al hueso. Solo menciona objetos cuando un personaje los toca o mira.",
        "Medio": "Equilibrio orgánico. Intercala atmósfera, luces, sonidos y texturas en las pausas naturales del diálogo.",
        "Alto": "Inmersión sensorial profunda y cinematográfica. El ambiente, el clima y los objetos refuerzan constantemente la tensión."
    }

    guia_tension = {
        "Bajo": "Lenguaje corporal relajado, cotidiano y naturalista.",
        "Sutil": "Microexpresiones involuntarias, desvíos de mirada, silencios incómodos y uso inconsciente de objetos cercanos (utilería).",
        "Dramático": "Tensión física intensa. Contradicción fuerte entre lo que dicen y lo que hacen sus cuerpos; gestos bruscos o contenidamente explosivos."
    }

    return f"""Eres un novelista de élite y director de escena experto en transformar borradores mínimos en prosa digna de publicación.
El autor que te consulta es directo y suele dar descripciones muy cortas, vagas o incluso dejar campos vacíos. TU TRABAJO ES HACER EL TRABAJO PESADO DE IMAGINACIÓN POR ÉL.

### REGLAS INQUEBRANTABLES:
1. MOTOR DE AUTO-DEDUCCIÓN Y EXPANSIÓN DE ESCENARIO:
   - Si el "Entorno" es muy escueto (ej. "afuera de la casa", "en una habitación") o está en Modo Automático, EXPÁNDELO tú mismo inventando utilería específica y coherente con el diálogo (ej. si hablan de un hechizo, añade rastro de ozono o maná en el aire, tierra removida, un bastón o libro, el crujido de los escalones de madera).
   - Si "Personajes" está vacío o solo menciona al narrador, EXTRAELOS automáticamente de los nombres en el diálogo en bruto y deduce su actitud, postura y emociones por lo que dicen.
   - Si el "Subtexto" está vacío o es mínimo, DEDUCELO de la fricción entre las líneas de diálogo.

2. FIDELIDAD DEL 100% AL DIÁLOGO:
   - Respeta palabra por palabra el 100% de las líneas de diálogo provistas por el autor.
   - NO cambies palabras, no agregues muletillas dentro del parlamento ni resumas lo que dicen. (Solo puedes corregir ortografía evidente si falta una tilde o signo de apertura).
   - Aplica el formato de diálogo: {formato_dialogo}.

3. CERO CLICHÉS DE LENGUAJE CORPORAL:
   - PROHIBIDO usar frases gastadas como: "frunció el ceño", "tragó saliva", "suspiró pesadamente", "un escalofrío recorrió su espalda", "apretó los puños", "se mordió el labio" o "sus ojos se abrieron como platos".
   - Usa acciones físicas concretas con el entorno (sacudirse el polvo de la ropa, apoyar la bota en un escalón, desviar la vista hacia un objeto, rascarse la nuca, limpiar una herramienta).

4. PARÁMETROS ACTIVOS DE LA ESCENA:
   - Género / Atmósfera: {genero}
   - Momento y Clima: {clima_hora}
   - Punto de vista (POV): {punto_de_vista}. (Si es 3ª limitada o 1ª persona y el autor no especificó quién es el protagonista, elige como focalizador al primer personaje que habla).
   - Nivel de entorno ({nivel_entorno}): {guia_entorno[nivel_entorno]}
   - Nivel de lenguaje corporal ({nivel_tension}): {guia_tension[nivel_tension]}
"""


def construir_user_prompt(
    entorno: str,
    subtexto: str,
    personajes: str,
    dialogo_bruto: str
) -> str:
    entorno_txt = entorno.strip() if entorno.strip() else "AUTOMÁTICO (Dedúcelo del diálogo e inventa un escenario rico con utilería)."
    subtexto_txt = subtexto.strip() if subtexto.strip() else "AUTOMÁTICO (Dedúcelo de la intención oculta en las líneas de diálogo)."
    personajes_txt = personajes.strip() if personajes.strip() else "AUTOMÁTICO (Identifícalos por los nombres en el diálogo y deduce cómo actúan)."

    return f"""Convierte este borrador rápido en una escena literaria completa. Rellena con creatividad cualquier detalle faltante:

- **Entorno / Ubicación (Expandir si es breve):** {entorno_txt}
- **Subtexto / Tensión:** {subtexto_txt}
- **Personajes:** {personajes_txt}

### DIÁLOGO EN BRUTO (MANTENER LAS LÍNEAS INTACTAS):
{dialogo_bruto}
"""


# =====================================================================
# 3. GESTIÓN DE ESTADO Y API KEY SEGURA
# =====================================================================
def inicializar_estado():
    if "escena_generada" not in st.session_state:
        st.session_state.escena_generada = None
    if "escenario_inventado" not in st.session_state:
        st.session_state.escenario_inventado = None
    if "analisis_generado" not in st.session_state:
        st.session_state.analisis_generado = None
    if "historial" not in st.session_state:
        st.session_state.historial = []


def obtener_api_key_servidor() -> str:
    """Obtiene la API Key de Streamlit Secrets o de las variables de entorno sin exponerla."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip()


# =====================================================================
# 4. INTERFAZ PRINCIPAL DE STREAMLIT
# =====================================================================
def main():
    st.set_page_config(
        page_title="Estudio de Escenas IA",
        page_icon="🖋️",
        layout="wide"
    )
    inicializar_estado()

    # Valores por defecto blindados (evita cualquier error de variable no definida)
    api_key_input = obtener_api_key_servidor()
    modelo = "gemini-2.5-flash"
    nivel_entorno = "Medio"
    nivel_tension = "Sutil"
    punto_de_vista = "3ª persona limitada"
    formato_dialogo = "Raya literaria tradicional en español (—)"
    temperatura = 0.8

    # --- PANEL LATERAL ---
    with st.sidebar:
        st.header("⚙️ Configuración")

        if api_key_input:
            st.success("🔒 API Key conectada de forma segura.")
        else:
            api_key_input = st.text_input(
                "Gemini API Key",
                type="password",
                help="Ingresa tu clave de Google AI Studio o configúrala en Secrets."
            )

        modelo = st.selectbox(
            "Modelo de IA",
            options=[
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-1.5-pro",
                "gemini-1.5-flash"
            ],
            index=0
        )

        st.divider()
        st.subheader("🎛️ Controles Narrativos")

        punto_de_vista = st.selectbox(
            "Punto de vista (POV)",
            options=[
                "3ª persona limitada",
                "3ª persona omnisciente",
                "1ª persona"
            ]
        )

        nivel_entorno = st.select_slider(
            "Cantidad de descripción del lugar",
            options=["Bajo", "Medio", "Alto"],
            value="Medio"
        )

        nivel_tension = st.select_slider(
            "Intensidad de gestos y cuerpo",
            options=["Bajo", "Sutil", "Dramático"],
            value="Sutil"
        )

        formato_dialogo = st.radio(
            "Formato de diálogo",
            options=[
                "Raya literaria tradicional en español (—)",
                "Comillas (\" \")"
            ],
            index=0
        )

        temperatura = st.slider(
            "Creatividad de la IA para rellenar detalles",
            min_value=0.2,
            max_value=1.2,
            value=0.80,
            step=0.05
        )

    # --- PANEL PRINCIPAL ---
    st.title("🖋️ Enriquecedor de Escenas (Modo Rápido)")
    st.caption(
        "💡 **Apto para descripciones mínimas:** Solo el diálogo es obligatorio. Si ponés dos palabras en el lugar o dejás campos en blanco, la IA deduce los personajes, inventa la utilería y arma la atmósfera sola."
    )

    # Selectores rápidos de 1 clic para no tener que escribir ambientación
    col_gen, col_cli = st.columns(2)
    with col_gen:
        genero = st.selectbox(
            "🎭 Género / Onda de la historia (1 clic)",
            options=[
                "Automático (Deducir por lo que hablan)",
                "Fantasía / Magia / Aventura",
                "Acción / Tensión pre-combate",
                "Comedia / Charla casual con picardía",
                "Drama / Misterio / Sospecha",
                "Ciencia Ficción / Urbano"
            ]
        )
    with col_cli:
        clima_hora = st.selectbox(
            "🌤️ Momento y Clima (1 clic)",
            options=[
                "Automático (El que mejor quede con la escena)",
                "Atardecer con brisa",
                "Noche cerrada / Oscuridad",
                "Día soleado y despejado",
                "Lluvia / Tormenta / Humedad",
                "Interior cerrado con luz tenue"
            ]
        )

    col_contexto, col_dialogo = st.columns([1, 1.3], gap="large")

    with col_contexto:
        st.subheader("1. Boceto Rápido (Opcional)")
        entorno = st.text_input(
            "¿Dónde están? (Aunque sea 2 palabras, o dejalo vacío)",
            placeholder="Ej: Afuera de la casa / En una taberna / (Vacío = Automático)"
        )
        subtexto = st.text_input(
            "¿Qué pasa por debajo? (Opcional)",
            placeholder="Ej: Sospecha que su amigo es un chismoso / (Vacío = Automático)"
        )
        personajes = st.text_input(
            "¿Algún detalle de los personajes? (Opcional)",
            placeholder="Si lo dejás vacío, los saca solos del diálogo."
        )

    with col_dialogo:
        st.subheader("2. Tu Diálogo en Bruto (Lo único obligatorio)")
        dialogo_bruto = st.text_area(
            "Escribí quién habla y qué dice:",
            height=220,
            placeholder=(
                'John: "Rudeus, que haces?"\n'
                'Rudeus: "Eh nada"\n'
                'John: "Mentiroso, estabas espiandome mientras hacía el hechizo, no?"'
            )
        )

    btn_generar = st.button("✨ Armar Escena Completa", type="primary", use_container_width=True)

    # --- LÓGICA DE LLAMADA A LA API ---
    if btn_generar:
        if not api_key_input:
            st.error("⚠️ Falta configurar la Gemini API Key en el panel izquierdo o en los Secrets de Streamlit.")
            st.stop()

        if not dialogo_bruto.strip():
            st.warning("⚠️ ¡Escribí aunque sea un par de líneas de diálogo para que tenga algo que transformar!")
            st.stop()

        with st.spinner("Deduciendo personajes, inventando utilería del lugar y armando la escena..."):
            try:
                client = genai.Client(api_key=api_key_input)

                sys_prompt = construir_system_prompt(
                    nivel_entorno=nivel_entorno,
                    nivel_tension=nivel_tension,
                    punto_de_vista=punto_de_vista,
                    formato_dialogo=formato_dialogo,
                    genero=genero,
                    clima_hora=clima_hora
                )
                usr_prompt = construir_user_prompt(
                    entorno=entorno,
                    subtexto=subtexto,
                    personajes=personajes,
                    dialogo_bruto=dialogo_bruto
                )

                response = client.models.generate_content(
                    model=modelo,
                    contents=usr_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_prompt,
                        temperature=temperatura,
                        response_mime_type="application/json",
                        response_schema=ResultadoEscena,
                    )
                )

                datos = json.loads(response.text)
                escena_final = datos.get("escena_narrada", "")
                inventado = datos.get("escenario_inventado", "")
                analisis = datos.get("analisis_subtexto", "")

                st.session_state.escena_generada = escena_final
                st.session_state.escenario_inventado = inventado
                st.session_state.analisis_generado = analisis

                # Guardar en el historial de la sesión
                st.session_state.historial.insert(0, {
                    "entorno": entorno.strip() or "Escenario Auto-deducido",
                    "escena": escena_final,
                    "inventado": inventado
                })

            except APIError as e:
                st.error(f"❌ Error de la API de Gemini: {e}")
            except Exception as e:
                st.error(f"❌ Ocurrió un error al procesar la escena: {e}")

    # --- SALIDA Y HERRAMIENTAS ---
    if st.session_state.escena_generada:
        st.divider()
        st.subheader("📖 Tu Escena Lista")

        # Mostrar qué rellenó automáticamente la IA
        if st.session_state.escenario_inventado:
            st.info(f"🧠 **Lo que la IA dedujo y agregó por vos:** {st.session_state.escenario_inventado}")

        tab_lectura, tab_copiar, tab_historial = st.tabs([
            "✨ Vista de Lectura",
            "📋 Copiar con 1 Clic",
            f"🕒 Historial ({len(st.session_state.historial)})"
        ])

        with tab_lectura:
            st.markdown(st.session_state.escena_generada)

        with tab_copiar:
            st.caption("Tocá el iconito de copiar arriba a la derecha del cuadro gris:")
            st.code(st.session_state.escena_generada, language="markdown")

        with tab_historial:
            for idx, item in enumerate(st.session_state.historial):
                with st.expander(f"Versión #{len(st.session_state.historial) - idx} — {item['entorno']}", expanded=(idx == 0)):
                    st.markdown(item["escena"])

        col_dl_md, col_dl_txt, _ = st.columns([1, 1, 2])
        with col_dl_md:
            st.download_button(
                label="⬇️ Descargar .MD",
                data=st.session_state.escena_generada,
                file_name="escena_enriquecida.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_dl_txt:
            st.download_button(
                label="⬇️ Descargar .TXT",
                data=st.session_state.escena_generada,
                file_name="escena_enriquecida.txt",
                mime="text/plain",
                use_container_width=True
            )

        with st.expander("🔍 Ver desglose de gestos, utilería y subtexto", expanded=False):
            st.markdown(st.session_state.analisis_generado)


if __name__ == "__main__":
    main()
