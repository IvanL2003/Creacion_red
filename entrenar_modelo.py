import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os

# ============================================================
# PASO 1: Cargar y preparar el dataset CSV
# ============================================================
print("=" * 60)
print("PASO 1: Cargando dataset...")
print("=" * 60)

df = pd.read_csv("hand_dataset.csv")
print(f"  Forma del dataset: {df.shape}")
print(f"  Clases encontradas: {df['label'].nunique()}")
print(f"  Distribución de clases:\n{df['label'].value_counts().to_string()}\n")

# Separar features y labels
X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values

# Codificar labels a números (A=0, B=1, ..., Z=25)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"  Mapeo de clases: {dict(zip(label_encoder.classes_, range(num_classes)))}")

# Guardar el mapeo de labels para usarlo en Android
label_map = {int(i): str(label) for i, label in enumerate(label_encoder.classes_)}
with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)
print(f"  label_map.json guardado con {len(label_map)} clases\n")

# Convertir labels a one-hot encoding
y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes)

# ============================================================
# PASO 2: Dividir en train/validation/test
# ============================================================
print("=" * 60)
print("PASO 2: Dividiendo dataset...")
print("=" * 60)

# 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_onehot, test_size=0.30, random_state=42, stratify=y_encoded
)

# Dividir temp en val y test
y_temp_labels = np.argmax(y_temp, axis=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp_labels
)

print(f"  Train:      {X_train.shape[0]} muestras")
print(f"  Validation: {X_val.shape[0]} muestras")
print(f"  Test:       {X_test.shape[0]} muestras\n")

# ============================================================
# PASO 3: Construir el modelo
# ============================================================
print("=" * 60)
print("PASO 3: Construyendo modelo...")
print("=" * 60)

model = tf.keras.Sequential([
    # Capa de entrada
    tf.keras.layers.Input(shape=(73,)),

    # Capa 1: 256 neuronas
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),

    # Capa 2: 128 neuronas
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    # Capa 3: 64 neuronas
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    # Capa de salida: una neurona por clase
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# PASO 4: Entrenar el modelo
# ============================================================
print("\n" + "=" * 60)
print("PASO 4: Entrenando modelo...")
print("=" * 60)

# Callbacks
callbacks = [
    # Detener si no mejora en 15 épocas
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    # Reducir learning rate si se estanca
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# PASO 5: Evaluar el modelo
# ============================================================
print("\n" + "=" * 60)
print("PASO 5: Evaluando modelo...")
print("=" * 60)

# Evaluar en test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Loss:     {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)\n")

# Matriz de confusión simplificada
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report
print("  Reporte de clasificación:")
print(classification_report(
    y_true, y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

# ============================================================
# PASO 6: Guardar modelo como SavedModel
# ============================================================
print("=" * 60)
print("PASO 6: Guardando modelo...")
print("=" * 60)

saved_model_dir = "hand_gesture_saved_model"
model.export(saved_model_dir)
print(f"  SavedModel guardado en: {saved_model_dir}/\n")

# ============================================================
# PASO 7: Convertir a TensorFlow Lite (.tflite)
# ============================================================
print("=" * 60)
print("PASO 7: Convirtiendo a TensorFlow Lite...")
print("=" * 60)

# Convertir desde SavedModel (compatible con Keras 3)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Optimización: reduce tamaño ~4x con mínima pérdida de precisión
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertir
tflite_model = converter.convert()

# Guardar .tflite
tflite_path = "hand_gesture_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

tflite_size_kb = os.path.getsize(tflite_path) / 1024
print(f"  Modelo TFLite guardado: {tflite_path}")
print(f"  Tamaño: {tflite_size_kb:.1f} KB\n")

# ============================================================
# PASO 8: Verificar modelo TFLite
# ============================================================
print("=" * 60)
print("PASO 8: Verificando modelo TFLite...")
print("=" * 60)

# Cargar y probar el modelo convertido
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  Input shape:  {input_details[0]['shape']}")
print(f"  Input dtype:  {input_details[0]['dtype']}")
print(f"  Output shape: {output_details[0]['shape']}")
print(f"  Output dtype: {output_details[0]['dtype']}")

# Probar con una muestra del test set
test_sample = np.expand_dims(X_test[0], axis=0).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_sample)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])
tflite_pred = np.argmax(tflite_output)
keras_pred = np.argmax(model.predict(test_sample, verbose=0))

print(f"\n  Predicción Keras:  {label_encoder.classes_[keras_pred]}")
print(f"  Predicción TFLite: {label_encoder.classes_[tflite_pred]}")
print(f"  ¿Coinciden? {'✓ SÍ' if keras_pred == tflite_pred else '✗ NO'}")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 60)
print("RESUMEN - Archivos generados:")
print("=" * 60)
print("  1. hand_gesture_saved_model/  → SavedModel completo")
print("  2. hand_gesture_model.tflite  → Modelo para Android")
print("  3. label_map.json             → Mapeo índice → letra")
print("\n  Para Android, copia estos archivos a:")
print("  app/src/main/assets/")
print("    ├── hand_gesture_model.tflite")
print("    └── label_map.json")
print("=" * 60)
