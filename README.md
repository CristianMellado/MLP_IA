## MLP
Perceptron Multicapa entrenado con CUDA para clasificacion de digitos del dataset MNIST
- Cesar Lengua Malaga
- Cristian Mellado Baca 
- Sebastian Postigo Avalos

## Fue hecho en Google Colab y ejecutado con su GPU. Los pasos para ejecutar `mlp_gpu.cu` son:

### Verificar la GPU y preparar el compilador

```
%%bash
apt-get update -qq
apt-get install -y -qq build-essential

echo "Versión de CUDA:"
nvcc --version
```

### Carga el dataset MNIST y preprocesarlo
```python
import numpy as np
from tensorflow.keras.datasets import mnist
#descargar datos
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#normalizar y aplanar imagenes (28x28 -> 784)
X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0  #[60000, 784]
X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0  #[10000, 784]
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[y_test]

#guardar en binario para q CUDA lo lea
X_train.tofile("X_train.bin")
y_train.tofile("y_train.bin")
X_test.tofile("X_test.bin")
y_test.tofile("y_test.bin")
```
### Se crea el `mlp_gpu.cu`
```cuda
%%writefile mlp_gpu.cu

// codigo del mlp_gpu.cu

```

### Compilar y ejecutarlo
```bash
%%bash
nvcc -std=c++17 -arch=sm_75 mlp_gpu.cu -o mlp_gpu
./mlp_gpu
```
los errores por epoca se guardan en un errors.txt

### Y se crea la matriz de confusion con las predicciones
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
NUM_TEST = 10000

#carga los pesos entrenados
W1 = np.fromfile("W1.bin", dtype=np.float32).reshape((INPUT_SIZE, HIDDEN_SIZE))
W2 = np.fromfile("W2.bin", dtype=np.float32).reshape((HIDDEN_SIZE, OUTPUT_SIZE))
B1 = np.fromfile("B1.bin", dtype=np.float32)  # (128,)
B2 = np.fromfile("B2.bin", dtype=np.float32)  # (10,)

#carga datos de test
h_X_test = np.fromfile("X_test.bin", dtype=np.float32).reshape((NUM_TEST, INPUT_SIZE))

def predict_batch(X_batch, W1, W2, B1, B2):
    B1 = B1.reshape(1, -1)
    B2 = B2.reshape(1, -1)
    Z1 = np.maximum(0, np.dot(X_batch, W1) + B1)
    logits = np.dot(Z1, W2) + B2
    predictions = np.argmax(logits, axis=1)
    return predictions

h_y_test = np.fromfile("y_test.bin", dtype=np.int32)
y_pred = predict_batch(h_X_test, W1, W2, B1, B2)

print("h_y_test unique:", np.unique(h_y_test))
print("y_pred unique:", np.unique(y_pred))

cm = confusion_matrix(h_y_test, y_pred, labels=np.arange(10))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Matriz de Confusión")
plt.show()
```
