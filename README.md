# Setup Ambiente e Avvio Server MLflow

Questo documento spiega come inizializzare l’ambiente Conda del progetto e come avviare un server MLflow in locale per il tracking degli esperimenti.

---

## 1. Creazione dell’ambiente Conda

Assicurati di avere [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o [Anaconda](https://www.anaconda.com/download) installato.

Crea l’ambiente da env.yaml:

```bash
conda env create -f env.yaml
```

## 2. Attivazione dell’ambiente
Attiva l’ambiente appena creato:

```bash
conda activate esercizio-env
```

## 3. (Opzionale) Avviare MLflow UI in locale
MLflow permette di tracciare esperimenti, metriche, modelli e artifact.
Puoi avviarlo in locale scrivendo da terminale:

```bash
mlflow server --host 127.0.0.1 --port 8080 
  ````
