import base64
import requests, aiohttp
import json
import os
import re

class GeminiVisionClient:
    """
    Cliente para interactuar con la API de Gemini 1.5 Flash para análisis de imágenes.
    """

    # El prompt optimizado como una constante de clase
    _OPTIMIZED_PROMPT = """
    Eres un asistente experto en análisis de imágenes. Tu tarea es analizar detalladamente la imagen provista y extraer las características de la persona principal, su género, una estimación de su edad, una descripción general y una puntuación de certeza para género y edad.

    La salida debe ser siempre un objeto JSON estrictamente formateado de la siguiente manera:

    {
    "gender": "string (masculino/femenino/no definido)",
    "age": "string (rango de edad estimado, ej: '0-1', '2-4', '5-10', '11-17', '18-25', '26-40', '41-60', '60+')",
    "features": [
        "string (característica 1)",
        "string (característica 2)",
        "..."
    ],
    "description": "string (descripción concisa de la persona y el contexto)",
    "score": [
        float (certeza del género, de 0.0 a 1.0),
        float (certeza de la edad, de 0.0 a 1.0)
    ]
    }

    Consideraciones para la respuesta:
    - `gender`: Responde 'masculino', 'femenino' o 'no definido' si no es posible determinarlo con certeza.
    - `age`: Proporciona un rango de edad estimado. Si es un niño muy pequeño, sé específico con rangos cortos (ej: '0-1', '2-4'). Para adultos, usa rangos más amplios.
    - `features`: Lista de puntos clave observables sobre la persona, su vestimenta, objetos con los que interactúa y su apariencia física. Sé descriptivo pero conciso.
    - `description`: Una oración o dos que resuman lo que se ve en la imagen centrado en la persona y su acción/contexto principal.
    - `score`: Un array de dos valores flotantes, donde el primer valor es la certeza del género y el segundo es la certeza de la edad. Utiliza valores entre 0.0 y 1.0. 1.0 es certeza total, 0.5 es 50/50, etc.

    Analiza la siguiente imagen:
"""
    _API_BASE_URL   = "https://generativelanguage.googleapis.com/v1beta/models/"
    _MODEL_NAME     = "gemini-1.5-flash" # Modelo actual recomendado

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("La API Key no puede estar vacía.")
        self.api_key = api_key
        self.api_endpoint = f"{self._API_BASE_URL}{self._MODEL_NAME}:generateContent?key={self.api_key}"

    def _image_to_base64(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"El archivo de imagen no se encontró en: {image_path}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _extract_json_from_markdown(self, text):
        """
        Extrae el objeto JSON de un bloque markdown ```json ... ``` o limpia el texto para intentar parsear el JSON.
        """
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1)
        match = re.search(r"```\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1)
        text = text.strip()
        return text

    async def analyze_image(self, image_path: str, mime_type: str = "image/jpeg") -> dict:
        try:
            image_base64 = self._image_to_base64(image_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Error al codificar la imagen a Base64: {e}")
            return {"error": f"Error al codificar la imagen: {e}"}

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self._OPTIMIZED_PROMPT},
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.api_endpoint, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    response_json = await response.json()

                    if 'candidates' in response_json and response_json['candidates']:
                        first_candidate_content = response_json['candidates'][0]['content']['parts'][0]['text']
                        json_string = self._extract_json_from_markdown(first_candidate_content)
                        parsed_json_data = json.loads(json_string)

                        if 'usageMetadata' in response_json:
                            parsed_json_data['usageMetadata'] = response_json['usageMetadata']
                        if 'modelVersion' in response_json:
                            parsed_json_data['modelVersion'] = response_json['modelVersion']

                        return parsed_json_data
                    else:
                        return {"error": "La respuesta de la API no contiene el formato esperado de 'candidates'.", "api_response": response_json}
            except aiohttp.ClientError as e:
                print(f"Error en la solicitud HTTP: {e}")
                return {"error": f"Error de red o API: {e}"}
            except json.JSONDecodeError as e:
                print(f"Error al decodificar la respuesta JSON de la API: {e}")
                # No uses response.text directamente aquí porque no es awaitable fuera del bloque 'with'
                return {"error": f"Error al parsear JSON: {e}"}
            except Exception as e:
                print(f"Ocurrió un error inesperado: {e}")
                return {"error": f"Error inesperado: {e}"}