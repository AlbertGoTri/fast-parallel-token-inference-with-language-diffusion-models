# ✅ Refactorización Completada: Ollama → Mistral-7B-Instruct

## Resumen de Cambios

Tu código ha sido completamente refactorizado para eliminar la dependencia de Ollama y usar **Mistral-7B-Instruct** (gratuito, local, sin servidor externo).

---

## Archivos Nuevos Creados

### 1. **mistral_judge_provider.py** 
Nuevo provider que carga Mistral-7B-Instruct desde Hugging Face y evalúa respuestas:
- Descarga automática la primera vez (~15GB)
- Se cachea localmente en `~/.cache/huggingface/hub/`
- Carga en memoria solo cuando se necesita
- Evaluación Yes/No para cada assertion

### 2. **promptfooconfig_mistral.yaml**
Nueva configuración de Promptfoo que usa Mistral:
- Mismo 12 prompts + ~5 assertions cada uno
- Usa `mistral_judge_provider.evaluate_yes_no()` en lugar de llamadas HTTP a Ollama
- Idéntico en propósito, pero 100% local

### 3. **MISTRAL_SETUP.md**
Documentación completa sobre:
- Cómo funciona la arquitectura
- Troubleshooting
- Comparación Ollama vs Mistral
- Cómo revertir si es necesario

### 4. **validate_setup.py**
Script de validación que verifica:
- Node.js, Python, NPX instalados
- Paquetes Python requeridos
- Archivos de configuración presentes
- Estado del servidor LLaDA

---

## Archivos Modificados

### 1. **run_evaluation.ps1** 
✂️ **Eliminadas:**
- Función `Test-OllamaRunning()`
- Check de Ollama antes de ejecutar
- Referencias a `$OllamaUrl = "http://127.0.0.1:11434"`
- Mensajes sobre "Ollama judges"

✏️ **Modificados:**
- Ahora usa `promptfooconfig_mistral.yaml` por defecto
- Mensajes actualizados: "Mistral-7B-Instruct (local, no external service)"
- Comando: `npx promptfoo eval -c promptfooconfig_mistral.yaml`

### 2. **README.md** (en evaluation/promptfoo/)
✏️ **Actualizado completamente:**
- Nueva descripción de la arquitectura
- Sin referencias a Ollama
- Requisitos de sistema (VRAM para Mistral)
- Pasos manuales usando Mistral

### 3. **EVALUATION_GUIDE.md** (en evaluation/)
✏️ **Extensión con dos opciones:**
- **Option A (Recomendado):** Mistral-7B-Instruct
- **Option B (Legacy):** Ollama (instrucciones conservadas)
- Ambas completamente documentadas

---

## Cómo Usar

### Opción 1: Automático (RECOMENDADO)
```powershell
# Desde raíz de LLaDA
cd c:\Users\Gotri\Documents\tfg\LLaDA
.\evaluation\promptfoo\run_evaluation.ps1
```

El script:
1. Verifica que Node.js esté instalado ✓
2. Inicia el servidor de LLaDA (si no está corriendo)
3. Descarga Mistral la primera vez (~15GB, se cachea luego)
4. Ejecuta evaluación (~15-30 minutos)
5. Genera reporte HTML

### Opción 2: Validar primero
```powershell
cd c:\Users\Gotri\Documents\tfg\LLaDA\evaluation\promptfoo
python validate_setup.py

# Verificará:
# ✓ Node.js, Python, NPX
# ✓ Paquetes Python (torch, transformers)
# ✓ Archivos de configuración
# ✓ Servidor LLaDA
```

### Opción 3: Manual
```bash
# Terminal 1: Servidor LLaDA
python serve_llada.py

# Terminal 2: Evaluación
cd evaluation/promptfoo
npx promptfoo eval -c promptfooconfig_mistral.yaml

# Terminal 2: Generar reporte
python generate_report.py
```

---

## ¿Qué Desapareció?

❌ **Ollama**
- No necesita estar instalado
- No necesita `ollama pull llama3.1:8b`
- No necesita ejecutar `ollama serve`
- No hay puertos 11434 escuchando

✨ **Ventaja:**
- Un servidor menos que gestionar
- No hay problemas legales con Hugging Face (Apache 2.0)
- Sin límites de rate-limiting
- 100% offline después de descargar Mistral

---

## Requisitos de Sistema

| Componente | Antes (Ollama) | Ahora (Mistral) |
|------------|----------------|-----------------|
| VRAM | LLaDA ~16GB | LLaDA ~16GB + Mistral ~15GB |
| GPU | Recomendado | Recomendado |
| Instalaciones | Ollama + Python | Solo Python |
| Servicios externos | Ollama server | Ninguno |
| Descargas | llama3.1:8b | Mistral-7B-Instruct |
| Licencia | - | Apache 2.0 |

---

## En Caso de Problemas

### "CUDA out of memory"
Mistral + LLaDA necesitan ~30GB VRAM juntas. Opciones:
1. GPU con ≥24GB (RTX 4090, A100)
2. Ejecutar en CPU (lento pero funciona)
3. Usar máquina remota

### "Descarga se interrumpe"
Mistral es ~15GB. Si la conexión se interrumpe:
```python
# Prueba descarga manual:
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"
```

### "Evaluación tarda mucho"
Si está en CPU (sin GPU), es normal (~1 min/assertion). Para acelerar:
- Usar GPU: `torch_dtype=torch.float16`
- Reducir assertions en `promptfooconfig_mistral.yaml`

---

## Siguiente Paso

Ejecuta ahora:

```powershell
.\evaluation\promptfoo\run_evaluation.ps1
```

O primero valida:

```powershell
python evaluation/promptfoo/validate_setup.py
```

---

## Archivo Original Conservado

Si necesitas revertir a Ollama:
- `promptfooconfig.yaml` sigue ahí (versión original con Ollama)
- Simplemente usa: `npx promptfoo eval -c promptfooconfig.yaml`
- (Pero necesitarás `ollama serve` corriendo)

---

## Documentación Completa

- **MISTRAL_SETUP.md** - Detalles técnicos de Mistral
- **README.md** - Resumen rápido
- **EVALUATION_GUIDE.md** - Guía completa (secciones para Mistral y Ollama)
- **validate_setup.py** - Script de validación
