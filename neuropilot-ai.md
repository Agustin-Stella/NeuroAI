# NeuroPilot AI

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](LEGAL.md).


## AI-powered neurological signal analysis assistant

Version: 1.0  
Estado: Diseño inicial  
Autor: Agustín Stella  

---

# 1. Visión del proyecto

NeuroPilot AI es una plataforma de inteligencia artificial orientada a asistir en el análisis de señales neurológicas, comenzando con electroencefalogramas (EEG) para detección y análisis de actividad epiléptica.

El objetivo no es reemplazar al profesional médico, sino crear una herramienta de asistencia capaz de:

- reducir tiempos de análisis;
- detectar patrones difíciles de identificar;
- proporcionar explicaciones interpretables;
- generar reportes preliminares;
- facilitar investigación y aprendizaje.

La visión a largo plazo es convertirse en una plataforma general de IA aplicada a neurociencia.

---

# 2. Motivación

Actualmente, el análisis de EEG requiere:

- mucho tiempo por estudio;
- especialistas altamente capacitados;
- interpretación manual;
- experiencia acumulada.

Los modelos actuales de IA tienen buenos resultados experimentales, pero presentan problemas:

- falta de explicabilidad;
- dificultad para integrarse al flujo médico;
- poca confianza del usuario;
- problemas de generalización entre datasets.

NeuroPilot busca solucionar esto creando una capa de inteligencia explicable y usable.

---

# 3. Objetivo principal

Construir un asistente de IA capaz de analizar señales EEG y ayudar a identificar actividad neurológica anormal.

Primera aplicación:

## Detección y análisis de crisis epilépticas mediante EEG.

---

# 4. Alcance inicial (MVP)

## Entrada

El sistema recibirá:

- archivos EEG en formato EDF;
- señales multicanal;
- metadata básica del estudio.

Ejemplo:

- cantidad de canales;
- frecuencia de muestreo;
- duración;
- paciente anonimizado.

---

# 5. Funcionalidades principales

## 5.1 Visualizador EEG

El sistema debe permitir:

- cargar un EEG;
- visualizar canales;
- navegar temporalmente;
- hacer zoom;
- seleccionar regiones.

Debe funcionar como un visor médico simplificado.

---

## 5.2 Preprocesamiento automático

Pipeline:

```
Entrada EEG
        ↓
Limpieza:
  - eliminación de ruido;
  - filtrado;
  - normalización;
  - detección de artefactos.
        ↓
Datos preparados para IA.
```

---

## 5.3 Detección automática de eventos

Modelo inicial:

Clasificación binaria:

- normal;
- actividad epiléptica.

Salida:

Ejemplo:

```
Evento detectado:
  Inicio:     13:42:10
  Fin:        13:42:34
  Confianza:  92%
```

---

## 5.4 Localización temporal

El sistema debe indicar:

- cuándo comienza el evento;
- cuánto dura;
- cuándo termina.

Ejemplo:

```
EEG completo:

0s ---------------------- 3600s

         [CRISIS]
          1200s
          1240s
```

---

## 5.5 Interpretabilidad del modelo

La IA debe explicar sus decisiones.

Ejemplo:

"El modelo detectó actividad sospechosa debido a:

- patrón rítmico;
- aumento de frecuencia;
- actividad localizada en canales temporales."

Tecnologías posibles:

- SHAP;
- Grad-CAM;
- Attention visualization.

---

## 5.6 Generación automática de reportes

Crear un resumen:

Ejemplo:

```
Análisis NeuroPilot AI

Eventos encontrados:
3

Evento principal:
13:42:10 - 13:42:34

Canales relevantes:
T3
T5

Confianza:
89%

Observaciones:
Actividad compatible con patrón epiléptico.
```

---

# 6. Funcionalidades futuras

## 6.1 Asistente conversacional sobre EEG

Integrar LLM.

Ejemplos:

Usuario:

"¿Qué anomalías encontraste?"

IA:

"Se detectaron tres eventos compatibles con actividad epiléptica..."

---

## 6.2 Comparación histórica

Comparar:

- EEG anterior;
- EEG actual;
- evolución temporal.

---

## 6.3 Multimodalidad

Agregar:

- resonancias;
- historia clínica;
- medicación;
- síntomas.

Objetivo:

Crear un modelo neuroclínico completo.

---

## 6.4 Modo educativo

Para estudiantes:

- practicar interpretación;
- comparar con IA;
- aprender patrones.

---

## 6.5 Plataforma de investigación

Permitir:

- cargar datasets;
- entrenar modelos;
- comparar arquitecturas;
- reproducir experimentos.

---

# 7. Arquitectura propuesta

## Frontend

Tecnologías:

- React
- TypeScript
- Tailwind
- Recharts / Plotly

Responsabilidades:

- dashboard;
- visualización EEG;
- interacción usuario.

---

## Backend

Tecnologías:

- Python
- FastAPI
- PostgreSQL
- Redis
- Celery

Responsabilidades:

- gestión usuarios;
- procesamiento EEG;
- ejecución modelos;
- generación reportes.

---

## Machine Learning

Stack:

- Python
- PyTorch
- MNE-Python
- NumPy
- SciPy

---

# 8. Arquitectura ML

Pipeline:

```
EEG Raw
        ↓
Preprocessing
        ↓
Feature extraction
        ↓
Deep Learning Model
        ↓
Prediction
        ↓
Explainability
        ↓
Report
```

---

# 9. Modelos a investigar

Orden:

## Modelo base

CNN

Objetivo:

Crear baseline.

---

## Segundo modelo

CNN + LSTM

Objetivo:

Capturar información temporal.

---

## Modelo avanzado

Transformer para señales.

Objetivo:

Aprender relaciones temporales complejas.

---

# 10. Métricas importantes

No usar solamente accuracy.

Evaluar:

- Precision
- Recall
- F1 Score
- Sensibilidad
- Especificidad
- False Positive Rate
- False Negative Rate

En medicina:

Un falso negativo puede ser más grave que un falso positivo.

---

# 11. Dataset inicial

Datasets públicos:

- CHB-MIT Scalp EEG Database
- Temple University EEG Corpus

Todos los datos deben ser anonimizados.

---

# 12. Roadmap

## Fase 0 - Investigación

Duración:
1 mes

Objetivos:

- estudiar EEG;
- estudiar epilepsia;
- entender datasets;
- documentar problema.

Resultado:

Documento técnico.

---

## Fase 1 - Exploración de datos

Duración:
1-2 meses

Crear:

- notebooks;
- visualización EEG;
- análisis estadístico.

Resultado:

Primer análisis funcional.

---

## Fase 2 - Primer modelo IA

Duración:
2-3 meses

Crear:

- pipeline ML;
- entrenamiento;
- evaluación.

Resultado:

Modelo capaz de detectar eventos.

---

## Fase 3 - Plataforma

Duración:
3 meses

Crear:

- backend;
- frontend;
- dashboard;
- carga EEG.

Resultado:

Demo completa.

---

## Fase 4 - IA avanzada

Duración:
6 meses

Agregar:

- Transformers;
- explicabilidad;
- optimización;
- comparación modelos.

---

## Fase 5 - Investigación

Crear:

- paper;
- documentación;
- repositorio público;
- presentación.

---

# 13. Principios del proyecto

## No crear una caja negra

Toda predicción debe ser explicable.

---

## Calidad antes que velocidad

Priorizar:

- código limpio;
- documentación;
- reproducibilidad.

---

## Mentalidad científica

Toda mejora debe medirse.

No importa que el modelo "parezca funcionar".

Debe demostrarse.

---

## Impacto real

El objetivo final es crear una herramienta útil para:

- médicos;
- investigadores;
- estudiantes;
- pacientes.

---

# 14. Estructura del repositorio

```
neuropilot-ai/

docs/
├── vision.md
├── architecture.md
├── research.md

backend/

frontend/

ml/
├── notebooks/
├── models/
├── training/

data/

tests/

README.md
```

---

# 15. Primera versión esperada

La primera versión completa debe permitir:

1. Subir EEG.
2. Visualizar señal.
3. Procesarla.
4. Ejecutar modelo.
5. Detectar eventos.
6. Mostrar explicación.
7. Generar reporte.

---

# 16. Objetivo final

Crear un proyecto que combine:

- Ingeniería de Software.
- Inteligencia Artificial.
- Ciencia de Datos.
- Neurociencia.

No como proyecto académico, sino como una plataforma con potencial real de investigación y aplicación clínica.
