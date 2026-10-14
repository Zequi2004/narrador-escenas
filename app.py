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
    analisis_subtexto: str = Field(
        description="Análisis técnico en viñetas explicando las decisiones de lenguaje corporal, uso de utilería, anclaje sensorial y ritmo narrativo."
    )


# =====================================================================
# 2. CONSTRUCCIÓN DEL SYSTEM PROMPT Y USER PROMPT
# =====================================================================
def construir_system_prompt(
    nivel_entorno: str,
    nivel_tension: str,
    punto_de_vista: str,
    formato_dialogo: str
) -> str:
    """Genera las instrucciones del sistema inyectando los parámetros de control literario."""
    
    guia_entorno = {
        "Bajo": "Pinceladas sensoriales mínimas y funcionales. Solo menciona el entorno cuando un personaje interactúa físicamente con él.",
        "Medio": "Equilibrio orgánico. Integra iluminación, acústica, temperatura y texturas en las pausas naturales del diálogo para sostener la atmósfera.",
        "Alto": "Inmersión sensorial profunda. El entorno actúa como un espejo psicológico del subtexto (sonidos de fondo, olores, texturas, contrastes de luz y sombra constantes)."
    }

    guia_tension = {
        "Bajo": "Lenguaje corporal contenido y naturalista. Acciones cotidianas con las manos o la postura que revelan el estado interno sin dramatismo.",
        "Sutil": "Microexpresiones involuntarias, desvíos de mirada, cambios en la cadencia respiratoria y manipulación inconsciente de objetos cercanos (utilería).",
        "Dramático": "Tensión física de alto impacto. Contradicción marcada entre lo que el personaje dice y lo que hace su cuerpo; rigidez muscular, silencios densos e interacción brusca o tensa con el espacio."
    }

    return f"""Eres un editor literario de élite y novelista experto en prosa inmersiva, subtexto dramático y puesta en escena (*blocking* narrativo).
Tu objetivo es transformar un diálogo en bruto en una escena literaria rica, viviente y orgánica.

### REGLAS INQUEBRANTABLES DE EJECUCIÓN:
1. FIDELIDAD ABSOLUTA AL DIÁLOGO:
   - Debes respetar el 100% de las palabras exactas de cada línea de diálogo provista por el autor.
   - Está ESTRICTAMENTE PROHIBIDO parafrasear, recortar, añadir muletillas o alterar el texto hablado.
   - Convierte el formato de entrada (`Personaje: "Línea"`) al estándar literario solicitado ({formato_dialogo}), integrando acotaciones narrativas (*action beats*) antes, en medio o después de los parlamentos según lo pida el ritmo.

2. PROHIBICIÓN ESTRICTA DE CLICHÉS SOMÁTICOS:
   - Queda PROHIBIDO usar frases gastadas como: "frunció el ceño", "tragó saliva", "suspiró pesadamente", "un escalofrío recorrió su espalda", "sus ojos se abrieron como platos", "apretó los puños", "se mordió el labio" o "el silencio se podía cortar con un cuchillo".
   - En lugar de nombrar la emoción de forma genérica, muéstrala mediante acciones específicas con objetos del entorno (utilería) y microexpresiones involuntarias (ej. el pulgar raspando el borde de una taza desportillada, el retraso de un segundo antes de soltar el picaporte, la vista clavada en una mancha de grasa).

3. RITMO Y CADENCIA ORGÁNICA:
   - No sobrecargues cada línea con un párrafo descriptivo.
   - Si el intercambio es rápido o cortante, deja líneas sucesivas casi limpias para preservar la velocidad verbal.
   - Inserta las pausas sensoriales y de acción física exactamente donde haya silencios lógicos, dudas, ocultamiento de información o cambios de poder en la conversación.

4. PARÁMETROS TÉCNICOS DE LA ESCENA:
   - Punto de vista (POV): {punto_de_vista}. Respeta rigurosamente las restricciones de percepción de esta focalización (no describas pensamientos internos de otros personajes si usas 1ª persona o 3ª persona limitada).
   - Nivel de detalle del entorno ({nivel_entorno}): {guia_entorno[nivel_entorno]}
   - Nivel de tensión física y corporal ({nivel_tension}): {guia_tension[nivel_tension]}
"""


def construir_user_prompt(
    entorno: str,
    subtexto: str,
    personajes: str,
    dialogo_bruto: str
) -> str:
    """Estructura los datos de entrada de la escena para el modelo."""
    return f"""Transforma el siguiente material en una escena narrativa completa siguiendo todas tus directrices editoriales:

- **Entorno / Ubicación:** {entorno}
- **Subtexto / Emoción subyacente:** {subtexto}
- **Personajes presentes (y su estado/rol):** {personajes}

### DIÁLOGO EN BRUTO A INTEGRAR (RESPETAR 100% TEXTUAL):
{dialogo_bruto}
"""


# =====================================================================
# 3. CONFIGURACIÓN DE LA INTERFAZ (STREAMLIT)
# =====================================================================
def inicializar_estado():
    """Inicializa las variables en session_state para persistir los resultados."""
    if "escena_generada" not in st.session_state:
        st.session_state.escena_generada = None
    if "analisis_generado" not in st.session_state:
        st.session_state.analisis_generado = None


def main():
    st.set_page_config(
        page_title="Enriquecedor de Escenas Literarias",
        page_icon="🖋️",
        layout="wide"
    )
    inicializar_estado()

    # --- PANEL LATERAL (CONFIGURACIÓN) ---
    with st.sidebar:
        st.header("⚙️ Configuración")

        # 1. Gestión de API Key
        env_api_key = os.environ.get("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Gemini API Key",
            value=env_api_key,
            type="password",
            help="Ingresa tu clave de Google AI Studio o configúrala en la variable de entorno GEMINI_API_KEY."
        )
        if env_api_key and api_key_input == env_api_key:
            st.caption("✅ Cargada automáticamente desde `GEMINI_API_KEY`.")

        st.divider()

        # 2. Selector de Modelo
        modelo = st.selectbox(
            "Modelo de IA",
            options=[
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash"
            ],
            index=0,
            help="Los modelos 'pro' ofrecen mayor sutileza literaria y manejo de subtexto; los 'flash' son más rápidos."
        )

        st.divider()
        st.subheader("🎛️ Parámetros Narrativos")

        # 3. Sliders y Controles Literarios
        nivel_entorno = st.select_slider(
            "Nivel de detalle del entorno",
            options=["Bajo", "Medio", "Alto"],
            value="Medio"
        )

        nivel_tension = st.select_slider(
            "Nivel de tensión física / corporal",
            options=["Bajo", "Sutil", "Dramático"],
            value="Sutil"
        )

        punto_de_vista = st.selectbox(
            "Punto de vista (POV)",
            options=[
                "3ª persona limitada",
                "3ª persona omnisciente",
                "1ª persona"
            ]
        )

        formato_dialogo = st.radio(
            "Formato de diálogo de salida",
            options=[
                "Raya literaria tradicional en español (—)",
                "Comillas anglosajonas (\" \")"
            ],
            index=0
        )

        temperatura = st.slider(
            "Creatividad (Temperatura)",
            min_value=0.1,
            max_value=1.2,
            value=0.75,
            step=0.05,
            help="Valores entre 0.6 y 0.85 logran el mejor balance entre riqueza léxica y coherencia."
        )

    # --- PANEL PRINCIPAL ---
    st.title("🖋️ Estudio de Enriquecimiento de Escenas")
    st.markdown(
        "Transforma diálogos en bruto en prosa inmersiva con **anclaje sensorial**, **utilería** y **microexpresiones**, manteniendo intactas tus líneas originales."
    )

    col_contexto, col_dialogo = st.columns([1, 1.2], gap="large")

    with col_contexto:
        st.subheader("1. Contexto de la Escena")
        entorno = st.text_input(
            "Entorno / Ubicación",
            placeholder="Ej. Taller mecánico de noche bajo lluvia intensa, olor a aceite quemado y tubo fluorescente parpadeando."
        )
        subtexto = st.text_input(
            "Subtexto / Emoción subyacente",
            placeholder="Ej. Tensión contenida, sospecha de traición; ambos fingen normalidad."
        )
        personajes = st.text_input(
            "Personajes presentes (indica el focalizador si usas 1ª o 3ª limitada)",
            placeholder="Ej. Clara (desconfiada, POV), Marcos (nervioso, limpiando piezas de un motor)."
        )

    with col_dialogo:
        st.subheader("2. Diálogo en Bruto")
        dialogo_bruto = st.text_area(
            "Líneas de diálogo (Formato sugerido: Personaje: \"Línea\")",
            height=210,
            placeholder=(
                'Clara: "Pensé que habías cerrado a las ocho."\n'
                'Marcos: "El distribuidor de este sedán me dio más trabajo de la cuenta."\n'
                'Clara: "Suena el teléfono de la oficina desde que entré. ¿No vas a atender?"\n'
                'Marcos: "Es el contestador automático. Déjalo que salte."'
            )
        )

    btn_generar = st.button("✨ Generar Escena Narrativa", type="primary", use_container_width=True)

    # --- LÓGICA DE LLAMADA A LA API ---
    if btn_generar:
        if not api_key_input.strip():
            st.error("⚠️ Debes ingresar una Gemini API Key válida en el panel lateral.")
            st.stop()

        if not all([entorno.strip(), subtexto.strip(), personajes.strip(), dialogo_bruto.strip()]):
            st.warning("⚠️ Por favor, completa todos los campos de contexto y el diálogo en bruto antes de generar.")
            st.stop()

        with st.spinner("Orquestando puesta en escena, anclaje sensorial y subtexto..."):
            try:
                client = genai.Client(api_key=api_key_input.strip())

                sys_prompt = construir_system_prompt(
                    nivel_entorno=nivel_entorno,
                    nivel_tension=nivel_tension,
                    punto_de_vista=punto_de_vista,
                    formato_dialogo=formato_dialogo
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

                datos_respuesta = json.loads(response.text)
                st.session_state.escena_generada = datos_respuesta.get("escena_narrada", "")
                st.session_state.analisis_generado = datos_respuesta.get("analisis_subtexto", "")

            except APIError as e:
                st.error(f"❌ Error de la API de Gemini: {e}")
            except Exception as e:
                st.error(f"❌ Ocurrió un error inesperado al procesar la escena: {e}")

    # --- SALIDA Y HERRAMIENTAS ---
    if st.session_state.escena_generada:
        st.divider()
        st.subheader("📖 Resultado de la Escena")

        tab_lectura, tab_copiar = st.tabs(["Vista Editorial", "Copiar al Portapapeles (Texto Plano)"])

        with tab_lectura:
            st.markdown(st.session_state.escena_generada)

        with tab_copiar:
            st.caption("Haz clic en el ícono de la esquina superior derecha del recuadro para copiar todo el texto con un clic:")
            st.code(st.session_state.escena_generada, language="markdown")

        # Botones de descarga (.md y .txt)
        col_dl_md, col_dl_txt, _ = st.columns([1, 1, 2])
        with col_dl_md:
            st.download_button(
                label="⬇️ Descargar como .MD",
                data=st.session_state.escena_generada,
                file_name="escena_enriquecida.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_dl_txt:
            st.download_button(
                label="⬇️ Descargar como .TXT",
                data=st.session_state.escena_generada,
                file_name="escena_enriquecida.txt",
                mime="text/plain",
                use_container_width=True
            )

        # Acordeón desplegable con el análisis técnico
        with st.expander("🔍 Análisis de Decisiones: Lenguaje Corporal, Utilería y Subtexto", expanded=False):
            st.markdown(st.session_state.analisis_generado)


if __name__ == "__main__":
    main()