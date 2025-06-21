#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>

//hiperparametros
const int INPUT_SIZE = 784; // 28x28
const int HIDDEN_SIZE = 128; // Neuronas de la capa oculta
const int OUTPUT_SIZE = 10; //0-9
const float LEARNING_RATE = 0.01f; //la tasa de aprednizaje
const int EPOCHS = 1000; //epocas
const int BATCH_SIZE = 64;

//kernel: Forward Pass (ReLU + Logits)
__global__ void forward_pass(float* X, float* W1, float* W2, float* B1, float* B2,
                            float* Z1, float* Z2, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Capa oculta (ReLU)
    if (idx < batch_size * HIDDEN_SIZE) {
        int batch = idx / HIDDEN_SIZE;
        int neuron = idx % HIDDEN_SIZE;
        float sum = B1[neuron];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            sum += W1[neuron * INPUT_SIZE + i] * X[batch * INPUT_SIZE + i];
        }
        Z1[idx] = fmaxf(0.0f, sum);  // ReLU
    }
    // Capa de salida (Logits para Softmax)
    else if (idx < batch_size * (HIDDEN_SIZE + OUTPUT_SIZE)) {
        int batch = (idx - batch_size * HIDDEN_SIZE) / OUTPUT_SIZE;
        int neuron = (idx - batch_size * HIDDEN_SIZE) % OUTPUT_SIZE;
        float sum = B2[neuron];
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            sum += W2[neuron * HIDDEN_SIZE + i] * Z1[batch * HIDDEN_SIZE + i];
        }
        Z2[batch * OUTPUT_SIZE + neuron] = sum;
    }

}


//kernel: Backward Pass (Gradientes)
__global__ void backward_pass(float* X, float* W1, float* W2, float* B1, float* B2,
                             float* Z1, float* Z2, int* Y, float* dW1, float* dW2,
                             float* dB1, float* dB2, int batch_size) {
    extern __shared__ float shared_mem[];
    float* dZ2 = shared_mem; // gradientes capa salida
    float* dZ1 = &shared_mem[batch_size * OUTPUT_SIZE]; //gradientes capa oculta

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    //gradiente capa salida (Softmax + Cross-Entropy)
    if (idx < batch_size * OUTPUT_SIZE) {
        int batch = idx / OUTPUT_SIZE;
        int neuron = idx % OUTPUT_SIZE;
        float target = (neuron == Y[batch]) ? 1.0f : 0.0f;
        dZ2[idx] = (Z2[idx] - target) / batch_size;
    }

    //gradientes W2 y B2
    if (idx < OUTPUT_SIZE * HIDDEN_SIZE) {
        int row = idx / HIDDEN_SIZE;
        int col = idx % HIDDEN_SIZE;
        float grad = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad += dZ2[b * OUTPUT_SIZE + row] * Z1[b * HIDDEN_SIZE + col];
        }
        dW2[idx] = grad;
    }

    //gradiente B2
    if (idx < OUTPUT_SIZE) {
        float grad = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad += dZ2[b * OUTPUT_SIZE + idx];
        }
        dB2[idx] = grad;
    }

    __syncthreads();

    //gradiente capa oculta (ReLU)
    if (idx < batch_size * HIDDEN_SIZE) {
        int batch = idx / HIDDEN_SIZE;
        int neuron = idx % HIDDEN_SIZE;
        float grad = 0.0f;
        for (int k = 0; k < OUTPUT_SIZE; ++k) {
            grad += dZ2[batch * OUTPUT_SIZE + k] * W2[k * HIDDEN_SIZE + neuron];
        }
        grad *= (Z1[idx] > 0.0f) ? 1.0f : 0.0f;
        dZ1[idx] = grad;  // ∂L/∂Z1
    }

    __syncthreads();

    //gradientes W1 y B1
    if (idx < HIDDEN_SIZE * INPUT_SIZE) {
        int row = idx / INPUT_SIZE;
        int col = idx % INPUT_SIZE;
        float grad = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad += dZ1[b * HIDDEN_SIZE + row] * X[b * INPUT_SIZE + col];
        }
        dW1[idx] = grad;
    }

    //gradiente B1
    if (idx < HIDDEN_SIZE) {
        float grad = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad += dZ1[b * HIDDEN_SIZE + idx];
        }
        dB1[idx] = grad;
    }
}

//kernel: actualizacion de pesos
__global__ void update_weights(float* W1, float* W2, float* B1, float* B2,
                              float* dW1, float* dW2, float* dB1, float* dB2,
                              float lr, int size_W1, int size_W2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size_W1) {
        W1[idx] -= lr * dW1[idx];  //actualizar W1

        //float update = lr * dW1[idx];
        //update = fmaxf(fminf(update, max_update), -max_update);
        //W1[idx] -= update;
    }
    else if (idx < size_W1 + size_W2) {
        W2[idx - size_W1] -= lr * dW2[idx - size_W1];  //actualizar W2
    }
    else if (idx < size_W1 + size_W2 + HIDDEN_SIZE) {
        B1[idx - size_W1 - size_W2] -= lr * dB1[idx - size_W1 - size_W2];  //actualizar B1

        //float update = lr * dB1[idx];
        //update = fmaxf(fminf(update, max_update), -max_update);
        //B1[idx] -= update;
    }
    else if (idx < size_W1 + size_W2 + HIDDEN_SIZE + OUTPUT_SIZE) {
        B2[idx - size_W1 - size_W2 - HIDDEN_SIZE] -= lr * dB2[idx - size_W1 - size_W2 - HIDDEN_SIZE];  //actualizar B2
        //float update = lr * dB2[idx];
        //update = fmaxf(fminf(update, max_update), -max_update);
        //B2[idx] -= update;
    }
}

void save_array(const char* filename, float* array, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(array), size * sizeof(float));
    file.close();
}

void train(float* h_X_train, int* h_y_train, int num_samples) {
    //reservar memoria en GPU
    float *d_X, *d_W1, *d_W2, *d_B1, *d_B2, *d_Z1, *d_Z2;
    int *d_Y;
    float *d_dW1, *d_dW2, *d_dB1, *d_dB2;

    cudaMalloc(&d_X, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_B1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_B2, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_Z1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_Z2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_Y, BATCH_SIZE * sizeof(int));
    cudaMalloc(&d_dW1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_dW2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_dB1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_dB2, OUTPUT_SIZE * sizeof(float));

    //inicializar pesos aleatorios enntre -0.1 y 0.1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);

    float h_W1[INPUT_SIZE * HIDDEN_SIZE];
    float h_W2[HIDDEN_SIZE * OUTPUT_SIZE];
    float h_B1[HIDDEN_SIZE] = {0};
    float h_B2[OUTPUT_SIZE] = {0};

    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; ++i) h_W1[i] = dis(gen);
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; ++i) h_W2[i] = dis(gen);

    cudaMemcpy(d_W1, h_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, h_B1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, h_B2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    std::ofstream error_file("errors.txt");
    if (!error_file.is_open()) {
        std::cerr << "Error al abrir errors.txt" << std::endl;
        return;
    }

    //entrenamiento
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float total_error = 0.0f;
        for (int batch_start = 0; batch_start < num_samples; batch_start += BATCH_SIZE) {
            // Copiar batch a GPU
            cudaMemcpy(d_X, h_X_train + batch_start * INPUT_SIZE,
                      BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Y, h_y_train + batch_start,
                      BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);

            //forward Pass
            forward_pass<<<(BATCH_SIZE * (HIDDEN_SIZE + OUTPUT_SIZE) + 255) / 256, 256>>>(
                d_X, d_W1, d_W2, d_B1, d_B2, d_Z1, d_Z2, BATCH_SIZE);

            float h_Z2[BATCH_SIZE * OUTPUT_SIZE];
            cudaMemcpy(h_Z2, d_Z2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            //en el calculo del error (train()), reemplaza:
             float batch_error = 0.0f;
            for (int b = 0; b < BATCH_SIZE; ++b) {
                for (int k = 0; k < OUTPUT_SIZE; ++k) {
                    float target = (k == h_y_train[batch_start + b]) ? 1.0f : 0.0f;
                    batch_error += powf(h_Z2[b * OUTPUT_SIZE + k] - target, 2);
                }
            }
            total_error += batch_error;

            //backward Pass
            int shared_mem_size = (BATCH_SIZE * (OUTPUT_SIZE + HIDDEN_SIZE)) * sizeof(float);
            backward_pass<<<(HIDDEN_SIZE * INPUT_SIZE + 255) / 256, 256, shared_mem_size>>>(
                d_X, d_W1, d_W2, d_B1, d_B2, d_Z1, d_Z2, d_Y,
                d_dW1, d_dW2, d_dB1, d_dB2, BATCH_SIZE);

            float h_dB2[OUTPUT_SIZE];
            cudaMemcpy(h_dB2, d_dB2, OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
            printf("Época %d - Grad B2: ", epoch);
            for (int i = 0; i < OUTPUT_SIZE; ++i) printf("%.4f ", h_dB2[i]);
            printf("\n");


            //antes de update_weights en train():
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Error post-backward: %s\n", cudaGetErrorString(err));


            //actualizar pesos
            update_weights<<<(INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE + 255)/256, 256>>>(
                d_W1, d_W2, d_B1, d_B2, d_dW1, d_dW2, d_dB1, d_dB2, LEARNING_RATE,
                INPUT_SIZE * HIDDEN_SIZE, HIDDEN_SIZE * OUTPUT_SIZE);
        }
        printf("Época %d: Error = %.4f\n", epoch, total_error / (num_samples * OUTPUT_SIZE));
        //guarda el error por Epoca
        error_file << epoch << " " << total_error / (num_samples * OUTPUT_SIZE) << "\n";

        float h_final_W1[INPUT_SIZE * HIDDEN_SIZE];
        float h_final_W2[HIDDEN_SIZE * OUTPUT_SIZE];
        float h_final_B1[HIDDEN_SIZE];
        float h_final_B2[OUTPUT_SIZE];

        cudaMemcpy(h_final_W1, d_W1, INPUT_SIZE*HIDDEN_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_final_W2, d_W2, HIDDEN_SIZE*OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_final_B1, d_B1, HIDDEN_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_final_B2, d_B2, OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

        //guarda los pesos en archivos binarios
        std::ofstream w1_file("W1.bin", std::ios::binary);
        w1_file.write(reinterpret_cast<char*>(h_final_W1), INPUT_SIZE*HIDDEN_SIZE*sizeof(float));
        w1_file.close();

        std::ofstream w2_file("W2.bin", std::ios::binary);
        w2_file.write(reinterpret_cast<char*>(h_final_W2), HIDDEN_SIZE*OUTPUT_SIZE*sizeof(float));
        w2_file.close();

        std::ofstream b1_file("B1.bin", std::ios::binary);
        b1_file.write(reinterpret_cast<char*>(h_final_B1), HIDDEN_SIZE*sizeof(float));
        b1_file.close();

        std::ofstream b2_file("B2.bin", std::ios::binary);
        b2_file.write(reinterpret_cast<char*>(h_final_B2), OUTPUT_SIZE*sizeof(float));
        b2_file.close();

    }

    error_file.close();

    float* h_final_W1 = new float[INPUT_SIZE * HIDDEN_SIZE];
    float* h_final_W2 = new float[HIDDEN_SIZE * OUTPUT_SIZE];
    float* h_final_B1 = new float[HIDDEN_SIZE];
    float* h_final_B2 = new float[OUTPUT_SIZE];

    cudaMemcpy(h_final_W1, d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_W2, d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_B1, d_B1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_B2, d_B2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* w1_file = fopen("W1.bin", "wb");
    fwrite(h_final_W1, sizeof(float), INPUT_SIZE*HIDDEN_SIZE, w1_file);
    fclose(w1_file);

    FILE* w2_file = fopen("W2.bin", "wb");
    fwrite(h_final_W2, sizeof(float), HIDDEN_SIZE*OUTPUT_SIZE, w2_file);
    fclose(w2_file);

    FILE* b1_file = fopen("B1.bin", "wb");
    fwrite(h_final_B1, sizeof(float), HIDDEN_SIZE, b1_file);
    fclose(b1_file);

    FILE* b2_file = fopen("B2.bin", "wb");
    fwrite(h_final_B2, sizeof(float), OUTPUT_SIZE, b2_file);
    fclose(b2_file);

    delete[] h_final_W1;
    delete[] h_final_W2;
    delete[] h_final_B1;
    delete[] h_final_B2;

    //liberar memoria GPU
    cudaFree(d_X);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_B1);
    cudaFree(d_B2);
    cudaFree(d_Z1);
    cudaFree(d_Z2);
    cudaFree(d_Y);
    cudaFree(d_dW1);
    cudaFree(d_dW2);
    cudaFree(d_dB1);
    cudaFree(d_dB2);

    //copiar los pesos finales desde device a host
    cudaMemcpy(h_final_W1, d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_W2, d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_B1, d_B1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_B2, d_B2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    save_array("W1.npy", h_final_W1, INPUT_SIZE * HIDDEN_SIZE);
    save_array("W2.npy", h_final_W2, HIDDEN_SIZE * OUTPUT_SIZE);
    save_array("B1.npy", h_final_B1, HIDDEN_SIZE);
    save_array("B2.npy", h_final_B2, OUTPUT_SIZE);

}

int main() {


    float h_X_train[BATCH_SIZE * INPUT_SIZE] = {0};
    int h_y_train[BATCH_SIZE] = {0};
    train(h_X_train, h_y_train, BATCH_SIZE);

}