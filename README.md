# AI Senior Care - Documentación Técnica (v1.0)

## Descripción General
**AI Senior Care** es una plataforma de asistencia inteligente diseñada para brindar apoyo a adultos mayores mediante el uso de inteligencia artificial (IA) y procesamiento de lenguaje natural (NLP) en español. Este proyecto ofrece un agente conversacional que detecta intenciones (e.g., recordatorios, citas) y emociones (e.g., ansiedad, tristeza) a partir de texto ingresado por el usuario, generando acciones automatizadas como alertas, recordatorios o mensajes de apoyo. La solución está pensada para entornos residenciales, promoviendo autonomía, seguridad y bienestar emocional.

## Problema que Resuelve
El envejecimiento de la población, especialmente en Colombia, presenta desafíos como el aislamiento social y la sobrecarga en cuidadores. AI Senior Care aborda estos problemas al proporcionar una herramienta accesible que:
- Facilita la comunicación natural con adultos mayores.
- Detecta estados emocionales para ofrecer acompañamiento empático.
- Automatiza tareas rutinarias, reduciendo la dependencia de asistencia humana.

## Tecnologías Clave
- **NLP y BERT en español:** Utiliza modelos como `dccuchile/bert-base-spanish-wwm-cased` para análisis de texto, entrenados con Hugging Face Transformers.
- **FastAPI:** Framework para el backend, orquestando la API y endpoints.
- **TailwindCSS:** Estilizado responsive para la interfaz de usuario (UI).
- **PyTorch y Scikit-learn:** Soporte para entrenamiento y métricas.
- **dateparser:** Procesamiento de fechas y horas en texto.

## Beneficios y Impacto Esperado
- **Beneficios:** Mejora del bienestar emocional, reducción de soledad, y apoyo continuo con mínima intervención humana.
- **Impacto:** Establece una base para soluciones escalables en residencias geriátricas, alineada con leyes de protección de datos como la Ley 1581 de 2012 en Colombia, con potencial de integración con APIs como Google Calendar y WhatsApp.

## Palabras Clave
IA, NLP, BERT, cuidado senior, procesamiento de lenguaje natural, FastAPI, TailwindCSS, salud emocional, Colombia.

## Instalación y Configuración
### Requisitos
- Python 3.11
- Al menos 4GB de RAM y 2GB de espacio libre
- Dependencias instaladas (ver `requirements.txt` y `requirements-py311.txt`)

### Pasos
1. **Crear entorno virtual:**
   ```bash
   python -m venv .venv311
   ```
2. **Activar entorno:**
   - Windows: `.\\.venv311\\Scripts\\Activate.ps1`
   - Linux/Mac: `source .venv311/bin/activate`
3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-py311.txt
   ```
4. **Entrenar modelos:**
   ```bash
   python services/nlp_intent/train_intent.py
   python services/nlp_emotion/train_emotion.py
   ```
5. **Ejecutar la API:**
   ```bash
   python -m services.gateway.app
   ```
6. **Acceder a la UI:**
   Abre tu navegador en `http://127.0.0.1:8000`.

## Estructura del Proyecto
```
ai-senior-care/
├─ services/
│  ├─ gateway/          → API principal con FastAPI
│  ├─ nlp_intent/       → Modelos y entrenamiento de intenciones
│  ├─ nlp_emotion/      → Modelos y entrenamiento de emociones
│  ├─ common/           → Esquemas compartidos (Pydantic)
│  └─ ui/               → Interfaz de usuario (HTML, TailwindCSS)
├─ data/                → Datasets CSV (e.g., sample_intent.csv, sample_emotion.csv)
├─ out/                 → Modelos entrenados
└─ requirements.txt     → Lista de dependencias
```

## Uso
1. Ingresa texto en la UI (e.g., "Recuérdame tomar mi pastilla mañana a las 8").
2. Haz clic en "Procesar" o usa Ctrl+Enter.
3. Observa los resultados: intención, emoción y acciones sugeridas (e.g., agendar en Google Calendar).

## Resultados
- **Métricas:** Accuracy >85%, F1 Macro >83%, latencia <1s (medido el 25/10/2025).
- **Casos de uso:** Soporta textos como "Me siento ansioso" con respuestas empáticas y "Tengo sangre en la nariz" con alertas médicas.

## Integraciones Futuras
- Google Calendar para agendamiento automático.
- Whisper para STT/TTS (voz a texto y texto a voz).
- Agentes WhatsApp para mensajes, búsqueda de contactos y análisis RAG diario.
- Dashboards para familiares y administradores con resúmenes emocionales.

## Consideraciones Éticas y Legales
- Cumple con la Ley 1581 de 2012 de Colombia para protección de datos.
- Mitiga sesgos con datasets diversificados y auditorías planificadas.
- Prioriza transparencia y empatía en interacciones.

## Contribuciones
Este proyecto aporta una solución inicial para el cuidado senior con IA en español, con un diseño modular y ético, ideal para residencias y familias.

## Licencia
[MIT License](LICENSE) - Libre para uso y modificación, sujeto a términos.

## Contacto
Para preguntas o colaboraciones, contacta a [tu correo o repositorio GitHub].

---

*Última actualización: 25 de octubre de 2025, 10:18 PM -05*
