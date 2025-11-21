# Process Filtering

Il sistema AZIMUTH supporta il **filtraggio dinamico dei processi** per permettere di addestrare il controller su subset di processi senza modificare la configurazione globale.

## Come Funziona

### 1. Definizione dei Processi (processes_config.py)

Tutti i processi disponibili sono definiti in `PROCESSES`:

```python
PROCESSES = [
    {'name': 'laser', ...},
    {'name': 'plasma', ...},
    {'name': 'galvanic', ...},
    {'name': 'microetch', ...}
]
```

### 2. Selezione dei Processi (controller_config.py)

Il campo `process_names` in `CONTROLLER_CONFIG` controlla quali processi usare:

```python
CONTROLLER_CONFIG = {
    'process_names': ['laser', 'plasma'],  # Usa solo questi 2
    ...
}
```

**Opzioni:**
- `None`: Usa TUTTI i processi definiti in `PROCESSES`
- `['laser', 'plasma']`: Usa solo laser e plasma
- `['laser', 'plasma', 'galvanic', 'microetch']`: Usa tutti e 4

### 3. Filtraggio Automatico

Il training automaticamente filtra i processi:

```python
# In train_controller.py
process_names = CONTROLLER_CONFIG.get('process_names', None)
selected_processes = get_filtered_processes(process_names)
# selected_processes contiene solo i processi richiesti
```

## Esempi d'Uso

### Solo 2 Processi (Development)

```python
# controller_config.py
CONTROLLER_CONFIG = {
    'process_names': ['laser', 'plasma'],
    ...
}
```

**Vantaggi:**
- Training più veloce
- Debugging più semplice
- Meno memoria richiesta

### Tutti i Processi (Production)

```python
# controller_config.py
CONTROLLER_CONFIG = {
    'process_names': None,  # O ometti completamente il campo
    ...
}
```

**Vantaggi:**
- Catena completa
- Risultati più realistici
- Propagazione completa dell'incertezza

### Custom Subset

```python
# controller_config.py
CONTROLLER_CONFIG = {
    'process_names': ['laser', 'galvanic'],  # Salta plasma
    ...
}
```

⚠️ **Attenzione:** Questo crea una catena non standard! Assicurati che abbia senso dal punto di vista causale.

## Ordine dei Processi

**L'ordine in `process_names` determina la catena**:

```python
['laser', 'plasma', 'galvanic']  # Laser → Plasma → Galvanic
['plasma', 'laser', 'galvanic']  # Plasma → Laser → Galvanic (inusuale!)
```

Di solito vuoi mantenere l'ordine standard: **Laser → Plasma → Galvanic → Microetch**

## Validazione

Il sistema valida automaticamente:

✅ **Successo:**
```python
process_names = ['laser', 'plasma']  # Entrambi esistono in PROCESSES
```

❌ **Errore:**
```python
process_names = ['laser', 'invalid']  # 'invalid' non esiste
# ValueError: Process 'invalid' not found in PROCESSES. Available: ['galvanic', 'laser', 'microetch', 'plasma']
```

## API

### get_filtered_processes()

```python
from controller_optimization.configs.processes_config import get_filtered_processes

# Tutti i processi
all_procs = get_filtered_processes(None)

# Solo alcuni processi
filtered = get_filtered_processes(['laser', 'plasma'])

# Ritorna: lista di configurazioni processi
# [
#     {'name': 'laser', 'input_dim': 2, ...},
#     {'name': 'plasma', 'input_dim': 2, ...}
# ]
```

**Parametri:**
- `process_names` (list[str] | None): Nomi dei processi da includere
  - `None`: ritorna tutti i processi
  - `['a', 'b']`: ritorna solo processi 'a' e 'b' nell'ordine specificato

**Returns:**
- `list`: Lista di configurazioni processi filtrata

**Raises:**
- `ValueError`: Se un processo specificato non esiste in `PROCESSES`

## Test

Esegui il test per verificare il funzionamento:

```bash
python controller_optimization/test_process_filtering.py
```

Output atteso:
```
TEST: Process Filtering
1. Test con process_names=None (tutti i processi):
   ✓ PASS: Ritorna tutti i processi
2. Test con process_names=['laser', 'plasma']:
   ✓ PASS: Filtra correttamente
3. Test con ordine personalizzato ['plasma', 'laser']:
   ✓ PASS: Ordine mantenuto
...
✓ TUTTI I TEST PASSATI!
```

## Workflow Tipico

### Development Phase

1. **Prototipo veloce**: Usa 2 processi
   ```python
   'process_names': ['laser', 'plasma']
   ```

2. **Test incrementale**: Aggiungi processi uno alla volta
   ```python
   'process_names': ['laser', 'plasma', 'galvanic']
   ```

3. **Final training**: Catena completa
   ```python
   'process_names': None  # Tutti i processi
   ```

### Production

- Usa sempre la catena completa per risultati finali
- Il filtraggio è principalmente per development/debugging

## File Coinvolti

- `configs/processes_config.py`: Definizione `PROCESSES` e `get_filtered_processes()`
- `configs/controller_config.py`: Campo `process_names` in `CONTROLLER_CONFIG`
- `train_controller.py`: Usa `get_filtered_processes()` per filtrare
- `test_process_filtering.py`: Test suite

## Backward Compatibility

Se `process_names` non è specificato:
- Default: `None`
- Comportamento: Usa TUTTI i processi
- ✅ Codice esistente continua a funzionare

## Best Practices

1. **Development**: Inizia con 2 processi per velocità
2. **Testing**: Incrementa gradualmente fino a 4
3. **Production**: Usa catena completa
4. **Ordine**: Mantieni sempre ordine causale logico
5. **Validation**: Lascia che il sistema validi i nomi (non bypassare)

## Limitazioni

- Non puoi filtrare per caratteristiche (e.g., "tutti i processi con output_dim=1")
- Devi specificare esplicitamente i nomi
- L'ordine deve avere senso causale (il sistema non lo valida automaticamente)

## Future Enhancements

Possibili estensioni:
- Auto-detect dell'ordine ottimale basato su dipendenze
- Filtraggio per caratteristiche (input_dim, output_dim, etc.)
- Preset comuni (e.g., 'minimal', 'standard', 'full')
- Validation automatica dell'ordine causale
