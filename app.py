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
   - En lugar de nombrar la emoción de forma genérica, muéstrala mediante acciones específicas con objetos del entorno (utilería) y microexpresiones involuntarias.

3. RITMO Y CADENCIA ORGÁNICA:
   - No sobrecargues cada línea con un párrafo descriptivo.
   - Si el intercambio es rápido o cortante, deja líneas sucesivas casi limpias para preservar la velocidad verbal.
   - Inserta las pausas sensoriales y de acción física exactamente donde haya silencios lógicos, dudas, ocultamiento de información o cambios de poder en la conversación.

4. PARÁMETROS TÉCNICOS DE LA ESCENA:
   - Punto de vista (POV): {punto_de_vista}. Respeta rigurosamente las restricciones de percepción de esta focalización.
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


def obtener_api_key_servidor() -> str:
    """Busca la API Key en st.secrets (Streamlit Cloud) o en variables de entorno."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


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

        # 1. Gestión de API Key (Oculta si ya está en el servidor)
        env_api_key = obtener_api_key_servidor()
        if env_api_key:
            api_key_input = env_api_key
            st.success("🔒 API Key conectada de forma segura.")
        else:
            api_key_input = st.text_input(
                "Gemini API Key",
                type="password",
                help="Ingresa tu clave de Google AI Studio."
            )

        st.divider()

        # 2. Selector de Modelo
        modelo = st.selectbox(
            "Modelo de IA",
            options=[
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-1.5-pro",
                "gemini-1.5-flash"
            ],
            index=0,
            help="Los modelos 'pro' ofrecen mayor sutileza literaria; los 'flash'
