import argparse, os, numpy as np, pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
# ✅ deja UNA sola definición de --model. Puedes cambiar el checkpoint si quieres.
parser.add_argument("--model", type=str, default="dccuchile/bert-base-spanish-wwm-cased")
parser.add_argument("--out", type=str, default="services/nlp_emotion/out")
parser.add_argument("--csv", type=str, default="data/sample_emotion.csv")
args = parser.parse_args()

labels = ["happy","sad","anxious","lonely","neutral"]
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in enumerate(labels)}

# === Cargar datos ===
df = pd.read_csv(args.csv)
ds = Dataset.from_pandas(df)

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(args.model)

def tok(ex):
    t = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=96)
    t["labels"] = label2id[ex["label"]]
    return t

# === Split + tokenización ===
ds = ds.train_test_split(test_size=0.2, seed=42)
ds_tok_train = ds["train"].map(tok)
ds_tok_test  = ds["test"].map(tok)

# ✅ Elimina columnas crudas para evitar que el collator intente tensorizar strings
to_drop_train = [c for c in ds_tok_train.column_names if c in ("text", "label", "__index_level_0__")]
to_drop_test  = [c for c in ds_tok_test.column_names  if c in ("text", "label", "__index_level_0__")]
if to_drop_train:
    ds_tok_train = ds_tok_train.remove_columns(to_drop_train)
if to_drop_test:
    ds_tok_test = ds_tok_test.remove_columns(to_drop_test)

# === Modelo ===
model = AutoModelForSequenceClassification.from_pretrained(
    args.model,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    # use_safetensors=True,  # ← descomenta si usas un checkpoint con .safetensors (p.ej., xlm-roberta-base)
)

# === Métricas ===
def compute(p):
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, average="macro")
    acc = accuracy_score(p.label_ids, preds)
    return {"f1": f1, "acc": acc}

# === Args de entrenamiento ===
os.makedirs(args.out, exist_ok=True)
training_args = TrainingArguments(
    output_dir=args.out,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=args.epochs,
    eval_strategy="epoch",          # reemplaza evaluation_strategy
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    dataloader_pin_memory=False,    # útil en Windows/CPU
    dataloader_num_workers=0        # evita issues de multiproceso en Windows
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_tok_train,
    eval_dataset=ds_tok_test,
    tokenizer=tokenizer,            # (deprec warning OK por ahora)
    compute_metrics=compute
)

trainer.train()
trainer.save_model(args.out)
tokenizer.save_pretrained(args.out)
print("Saved to", args.out)
