# Machine Learning for Chess Engines (From Zero to a Practical C Implementation)

This document is a **complete, self-contained guide** to what we discussed: how neural networks are trained (not by “keeping the good random weights,” but by **gradient descent + backpropagation**), how to implement a network in **C** without pointer-heavy “node graphs,” and how to adapt the exact code shape to a **chess engine evaluation** function with `Board`/bitboards.

It includes:
- The full math of forward pass, losses, gradients, and backprop
- Why dense arrays beat node graphs in practice
- A generic one-file C MLP demo (XOR) with save/load
- A chess-engine-shaped one-file C neural evaluation module (bitboards → features → eval), plus a training hook
- How to integrate into search and how to generate training targets
- A clear performance roadmap toward “professional / competitive” CPU evaluation (NNUE concepts, quantization, SIMD, batching, caching)

> **Important reality check:** A “professional performant” chess NN eval (like NNUE in modern engines) is **more engineering than theory**: feature design, incremental updates, fixed-point quantization, SIMD kernels, and careful integration with search. This guide takes you from zero to an implementation you can actually run, then outlines the precise steps to reach NNUE-grade performance.

---

## 0. Terminology and the core idea

A (feedforward) neural network for evaluation is a function

\[
f_\theta : \mathcal{S} \to \mathbb{R}
\]

where:
- \( \mathcal{S} \) is the set of chess positions (your `Board`)
- \( \theta \) are all parameters (weights and biases)
- output is typically a scalar “value” (e.g., centipawns or win-prob proxy)

In practice:
1. Convert position to a numeric **feature vector** \(x\in\mathbb{R}^{n}\).
2. Apply several layers of affine maps + nonlinearities.
3. Output a scalar \( \hat{y} \) as evaluation.

---

## 1. Why training is not “keep better random weights”

A common beginner mental model is: pick random weights, test, keep the good ones, repeat. That is **evolutionary search**.

Most neural nets train by **gradient descent**:
- Start with random weights once.
- For each training example:
  - Compute a prediction with a forward pass.
  - Compute an error via a loss function.
  - Compute gradients of the loss w.r.t. each weight using **backpropagation**.
  - Update weights a tiny step in the direction that reduces loss.

This is optimization of a differentiable objective.

---

## 2. The math of a fully connected layer

A dense (fully connected) layer maps input \(a^{(l-1)}\in\mathbb{R}^{n}\) to output \(a^{(l)}\in\mathbb{R}^{m}\) via:

### 2.1 Affine transform
\[
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
\]
- \(W^{(l)}\in\mathbb{R}^{m\times n}\)
- \(b^{(l)}\in\mathbb{R}^{m}\)
- \(z^{(l)}\in\mathbb{R}^{m}\) is **pre-activation**

### 2.2 Activation (nonlinearity)
\[
a^{(l)} = \sigma(z^{(l)})
\]
Applied elementwise:
- Sigmoid: \(\sigma(t)=\frac{1}{1+e^{-t}}\)
- ReLU: \(\sigma(t)=\max(0,t)\)
- Tanh: \(\sigma(t)=\tanh(t)\)

> Without a nonlinearity, compositions of affine maps collapse to a single affine map — you would not get a “deep” function family.

---

## 3. Loss functions: how the network knows it’s wrong

You need a scalar objective \(L(\theta)\) to minimize.

### 3.1 Mean Squared Error (MSE)
For a scalar output:
\[
L = \frac12(\hat{y} - y)^2
\]
Derivative:
\[
\frac{\partial L}{\partial \hat{y}} = \hat{y}-y
\]

MSE is simplest and is what we used for the minimal examples.

### 3.2 Chess-specific note
Modern value nets often predict:
- win probability / value in \([-1,1]\), trained with logistic/tanh targets, or
- a distribution over outcomes, or
- centipawn-like scalar

In this guide we use a **bounded output** (tanh) and then scale to centipawns.

---

## 4. Backpropagation: the chain rule in vector form

Backprop computes gradients \(\nabla_\theta L\) efficiently.

Define the **error signal** (“delta”) at layer \(l\) as:

\[
\delta^{(l)} := \frac{\partial L}{\partial z^{(l)}}
\]

### 4.1 Output layer delta

If output activation is \(a^{(L)}=\sigma(z^{(L)})\), then:

\[
\delta^{(L)} = \frac{\partial L}{\partial a^{(L)}} \odot \sigma'(z^{(L)})
\]

For MSE, \(\frac{\partial L}{\partial a^{(L)}} = a^{(L)} - y\).

So:
\[
\delta^{(L)} = (a^{(L)} - y)\odot \sigma'(z^{(L)})
\]

### 4.2 Hidden layer delta recurrence

For a hidden layer \(l\):

\[
\delta^{(l)} = \left((W^{(l+1)})^T \delta^{(l+1)}\right)\odot \sigma'(z^{(l)})
\]

This is the key identity: propagate error backward through the transpose of weights.

### 4.3 Gradients w.r.t. weights and biases

Because:
\[
z_i^{(l)} = \sum_j W_{ij}^{(l)} a_j^{(l-1)} + b_i^{(l)}
\]

We have:
\[
\frac{\partial L}{\partial W_{ij}^{(l)}} = \delta_i^{(l)} \cdot a_j^{(l-1)}
\]
\[
\frac{\partial L}{\partial b_i^{(l)}} = \delta_i^{(l)}
\]

So in code, weight gradient is outer-product-ish: `delta * prev_activation`.

---

## 5. Parameter update: gradient descent / SGD

A basic update rule (stochastic gradient descent):

\[
W \leftarrow W - \eta \frac{\partial L}{\partial W},\quad
b \leftarrow b - \eta \frac{\partial L}{\partial b}
\]

- \(\eta\) is the learning rate (lr)
- For minibatches, you typically average gradients over the batch.

---

## 6. Why you do **not** represent a NN as a node-pointer graph in C

You asked whether a node should be something like:

```c
typedef struct { weight, layer, points_to, pointed_by, ... } Node;
```

That is **not** how performant neural nets are implemented.

### 6.1 The reason
Dense neural nets are dominated by linear algebra operations:
- matrix-vector (inference)
- matrix-matrix (minibatch training)
- transposed matrix-vector (backprop)

A pointer-rich graph:
- is cache-unfriendly
- adds overhead per edge
- makes backprop more complex
- prevents easy SIMD / vectorization

### 6.2 The practical representation
Store weights as **flat arrays**:
- `W[l]` is a row-major matrix
- `b[l]` is a vector
- activations and deltas are vectors

This is exactly what the C code below does.

---

## 7. Generic one-file C MLP demo (XOR) — full code

This is the **exact one-file program** from our discussion:
- builds a general MLP with arbitrary layer sizes
- sigmoid activations
- MSE loss
- SGD updates
- save/load

> This is excellent as your “first correct backprop implementation” and sanity test.

```c
// nn_onefile.c
// Build (Linux/macOS):  gcc -O3 -march=native nn_onefile.c -lm -o nn
// Build (Windows MinGW): gcc -O3 nn_onefile.c -lm -o nn.exe
//
// Runs XOR training demo, prints loss, saves model to "model.bin".

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// ---------- tiny utilities ----------
static unsigned int g_rng = 1u;

static int seed_rng(unsigned int s) {
    if (s == 0u) s = 1u;
    g_rng = s;
    return 0;
}

// xorshift32 RNG
static unsigned int rng_u32() {
    unsigned int x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return x;
}

static float rng_f01() {
    // [0,1)
    return (rng_u32() / 4294967296.0f);
}

static float rng_f_neg1_1() {
    return 2.0f * rng_f01() - 1.0f;
}

static float sigmoid(float x) {
    // numerically okay for small demos
    return 1.0f / (1.0f + expf(-x));
}

static float dsigmoid_from_a(float a) {
    // if a = sigmoid(z), derivative wrt z is a*(1-a)
    return a * (1.0f - a);
}

// ---------- network representation ----------
// No structs: use parallel arrays.
//
// Layers: sizes[0] = input dim, sizes[L-1] = output dim
// For each layer l from 1..L-1:
//   W[l] is (sizes[l] x sizes[l-1]) stored row-major
//   b[l] is (sizes[l])
//
// Also store activations a[l] and pre-activations z[l] for forward pass.
// Gradients: dW[l], db[l]

static int alloc_network(
    int L,
    int *sizes,
    float ***W_out, float ***b_out,
    float ***a_out, float ***z_out,
    float ***dW_out, float ***db_out
) {
    int l;

    float **W = (float**)calloc((size_t)L, sizeof(float*));
    float **b = (float**)calloc((size_t)L, sizeof(float*));
    float **a = (float**)calloc((size_t)L, sizeof(float*));
    float **z = (float**)calloc((size_t)L, sizeof(float*));
    float **dW = (float**)calloc((size_t)L, sizeof(float*));
    float **db = (float**)calloc((size_t)L, sizeof(float*));

    if (!W || !b || !a || !z || !dW || !db) return 1;

    // activations exist for all layers
    for (l = 0; l < L; l++) {
        a[l] = (float*)calloc((size_t)sizes[l], sizeof(float));
        z[l] = (float*)calloc((size_t)sizes[l], sizeof(float));
        if (!a[l] || !z[l]) return 2;
    }

    // weights/biases exist for layers 1..L-1
    for (l = 1; l < L; l++) {
        int rows = sizes[l];
        int cols = sizes[l - 1];
        W[l]  = (float*)calloc((size_t)(rows * cols), sizeof(float));
        b[l]  = (float*)calloc((size_t)rows, sizeof(float));
        dW[l] = (float*)calloc((size_t)(rows * cols), sizeof(float));
        db[l] = (float*)calloc((size_t)rows, sizeof(float));
        if (!W[l] || !b[l] || !dW[l] || !db[l]) return 3;
    }

    *W_out = W; *b_out = b;
    *a_out = a; *z_out = z;
    *dW_out = dW; *db_out = db;
    return 0;
}

static int free_network(int L, float **W, float **b, float **a, float **z, float **dW, float **db) {
    int l;
    if (W)  { for (l = 0; l < L; l++) free(W[l]);  free(W); }
    if (b)  { for (l = 0; l < L; l++) free(b[l]);  free(b); }
    if (a)  { for (l = 0; l < L; l++) free(a[l]);  free(a); }
    if (z)  { for (l = 0; l < L; l++) free(z[l]);  free(z); }
    if (dW) { for (l = 0; l < L; l++) free(dW[l]); free(dW); }
    if (db) { for (l = 0; l < L; l++) free(db[l]); free(db); }
    return 0;
}

static int init_weights_xavier(int L, int *sizes, float **W, float **b) {
    // Xavier-ish uniform init: scale ~ sqrt(6/(fan_in+fan_out))
    int l;
    for (l = 1; l < L; l++) {
        int fan_in = sizes[l - 1];
        int fan_out = sizes[l];
        float limit = sqrtf(6.0f / (float)(fan_in + fan_out));

        int rows = fan_out;
        int cols = fan_in;
        int i;
        for (i = 0; i < rows * cols; i++) {
            W[l][i] = rng_f_neg1_1() * limit;
        }
        for (i = 0; i < rows; i++) {
            b[l][i] = 0.0f;
        }
    }
    return 0;
}

// Forward pass: a[0] already set to input.
// Computes z[l] = W[l] a[l-1] + b[l], then a[l] = sigmoid(z[l])
static int forward(int L, int *sizes, float **W, float **b, float **a, float **z) {
    int l;
    for (l = 1; l < L; l++) {
        int out_dim = sizes[l];
        int in_dim  = sizes[l - 1];

        int i, j;
        for (i = 0; i < out_dim; i++) {
            float sum = b[l][i];
            // row i of W times vector a[l-1]
            int row_base = i * in_dim;
            for (j = 0; j < in_dim; j++) {
                sum += W[l][row_base + j] * a[l - 1][j];
            }
            z[l][i] = sum;
            a[l][i] = sigmoid(sum);
        }
    }
    return 0;
}

// Zero gradients
static int zero_grads(int L, int *sizes, float **dW, float **db) {
    int l;
    for (l = 1; l < L; l++) {
        int rows = sizes[l];
        int cols = sizes[l - 1];
        int i;
        for (i = 0; i < rows * cols; i++) dW[l][i] = 0.0f;
        for (i = 0; i < rows; i++) db[l][i] = 0.0f;
    }
    return 0;
}

// Backprop for one sample (SGD).
// Loss: MSE = 0.5 * sum_k (a_L[k] - y[k])^2
// Output delta: (a - y) * sigmoid'(z) = (a - y) * a*(1-a)
// Hidden deltas: delta_l = (W_{l+1}^T delta_{l+1}) * sigmoid'(z_l)
//
// We accumulate gradients in dW/db arrays.
static int backward_one(
    int L, int *sizes,
    float **W,
    float **a,
    float **dW, float **db,
    float *y
) {
    // Allocate delta arrays for each layer on the fly.
    // For small projects this is fine; later you’d reuse buffers.
    int l;
    float **delta = (float**)calloc((size_t)L, sizeof(float*));
    if (!delta) return 10;

    for (l = 0; l < L; l++) {
        delta[l] = (float*)calloc((size_t)sizes[l], sizeof(float));
        if (!delta[l]) return 11;
    }

    // Output layer delta
    int last = L - 1;
    {
        int k;
        for (k = 0; k < sizes[last]; k++) {
            float ak = a[last][k];
            float diff = ak - y[k];
            delta[last][k] = diff * dsigmoid_from_a(ak);
        }
    }

    // Hidden layers deltas
    for (l = last - 1; l >= 1; l--) {
        int dim = sizes[l];
        int next_dim = sizes[l + 1];
        int prev_dim = sizes[l - 1];

        (void)prev_dim;

        int i, j;
        for (i = 0; i < dim; i++) {
            // sum_j W[l+1][j,i] * delta[l+1][j]
            float sum = 0.0f;
            for (j = 0; j < next_dim; j++) {
                // W[l+1] is row-major: index = j*dim + i (since dim == sizes[l])
                sum += W[l + 1][j * dim + i] * delta[l + 1][j];
            }
            float ai = a[l][i];
            delta[l][i] = sum * dsigmoid_from_a(ai);
        }
    }

    // Gradients:
    // dW[l][i,j] += delta[l][i] * a[l-1][j]
    // db[l][i]   += delta[l][i]
    for (l = 1; l < L; l++) {
        int out_dim = sizes[l];
        int in_dim = sizes[l - 1];

        int i, j;
        for (i = 0; i < out_dim; i++) {
            db[l][i] += delta[l][i];
            int row_base = i * in_dim;
            for (j = 0; j < in_dim; j++) {
                dW[l][row_base + j] += delta[l][i] * a[l - 1][j];
            }
        }
    }

    for (l = 0; l < L; l++) free(delta[l]);
    free(delta);
    return 0;
}

static float loss_mse(int out_dim, float *pred, float *y) {
    float s = 0.0f;
    int k;
    for (k = 0; k < out_dim; k++) {
        float d = pred[k] - y[k];
        s += 0.5f * d * d;
    }
    return s;
}

// SGD update: W -= lr * dW, b -= lr * db
// For minibatch, you’d average grads; here we do pure SGD on each sample.
static int sgd_update(int L, int *sizes, float **W, float **b, float **dW, float **db, float lr) {
    int l;
    for (l = 1; l < L; l++) {
        int rows = sizes[l];
        int cols = sizes[l - 1];
        int i;

        for (i = 0; i < rows * cols; i++) {
            W[l][i] -= lr * dW[l][i];
        }
        for (i = 0; i < rows; i++) {
            b[l][i] -= lr * db[l][i];
        }
    }
    return 0;
}

// Save/load
static int save_model(const char *path, int L, int *sizes, float **W, float **b) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;

    if (fwrite(&L, sizeof(int), 1, f) != 1) return 2;
    if (fwrite(sizes, sizeof(int), (size_t)L, f) != (size_t)L) return 3;

    int l;
    for (l = 1; l < L; l++) {
        int rows = sizes[l];
        int cols = sizes[l - 1];
        if (fwrite(W[l], sizeof(float), (size_t)(rows * cols), f) != (size_t)(rows * cols)) return 4;
        if (fwrite(b[l], sizeof(float), (size_t)rows, f) != (size_t)rows) return 5;
    }

    fclose(f);
    return 0;
}

static int load_model(const char *path, int *L_out, int **sizes_out, float ***W_out, float ***b_out,
                      float ***a_out, float ***z_out, float ***dW_out, float ***db_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return 1;

    int L;
    if (fread(&L, sizeof(int), 1, f) != 1) return 2;

    int *sizes = (int*)calloc((size_t)L, sizeof(int));
    if (!sizes) return 3;
    if (fread(sizes, sizeof(int), (size_t)L, f) != (size_t)L) return 4;

    float **W, **b, **a, **z, **dW, **db;
    if (alloc_network(L, sizes, &W, &b, &a, &z, &dW, &db) != 0) return 5;

    int l;
    for (l = 1; l < L; l++) {
        int rows = sizes[l];
        int cols = sizes[l - 1];
        if (fread(W[l], sizeof(float), (size_t)(rows * cols), f) != (size_t)(rows * cols)) return 6;
        if (fread(b[l], sizeof(float), (size_t)rows, f) != (size_t)rows) return 7;
    }

    fclose(f);

    *L_out = L;
    *sizes_out = sizes;
    *W_out = W; *b_out = b;
    *a_out = a; *z_out = z;
    *dW_out = dW; *db_out = db;
    return 0;
}

// ---------- demo: XOR dataset ----------
int main() {
    seed_rng((unsigned int)time(NULL));

    // XOR: 2 inputs -> 1 output
    // We'll use a 2-4-1 network.
    int L = 3;
    int *sizes = (int*)calloc((size_t)L, sizeof(int));
    sizes[0] = 2;
    sizes[1] = 4;
    sizes[2] = 1;

    float **W, **b, **a, **z, **dW, **db;
    if (alloc_network(L, sizes, &W, &b, &a, &z, &dW, &db) != 0) {
        printf("alloc_network failed\n");
        return 1;
    }
    init_weights_xavier(L, sizes, W, b);

    // Training data
    // X: 4 samples, each has 2 floats
    // Y: 4 samples, each has 1 float
    float train_X[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    float train_Y[4][1] = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    int epochs = 20000;
    float lr = 0.3f;

    int e;
    for (e = 1; e <= epochs; e++) {
        float total_loss = 0.0f;

        int i;
        for (i = 0; i < 4; i++) {
            // set input activation
            a[0][0] = train_X[i][0];
            a[0][1] = train_X[i][1];

            forward(L, sizes, W, b, a, z);

            total_loss += loss_mse(sizes[L - 1], a[L - 1], train_Y[i]);

            zero_grads(L, sizes, dW, db);
            backward_one(L, sizes, W, a, dW, db, train_Y[i]);
            sgd_update(L, sizes, W, b, dW, db, lr);
        }

        // print sometimes
        if ((e % 1000) == 0) {
            printf("epoch %d  loss %.6f\n", e, total_loss / 4.0f);
        }
    }

    // Show final predictions
    printf("\nFinal predictions:\n");
    int i;
    for (i = 0; i < 4; i++) {
        a[0][0] = train_X[i][0];
        a[0][1] = train_X[i][1];
        forward(L, sizes, W, b, a, z);
        printf("%.0f XOR %.0f => %.4f\n", train_X[i][0], train_X[i][1], a[L - 1][0]);
    }

    // Save model
    if (save_model("model.bin", L, sizes, W, b) == 0) {
        printf("\nSaved model to model.bin\n");
    } else {
        printf("\nFailed to save model\n");
    }

    free_network(L, W, b, a, z, dW, db);
    free(sizes);
    return 0;
}
```

---

## 8. Restructuring to a chess engine evaluation network in one file

### 8.1 Goal
Given your `Board b` (bitboards + `gamestate`), implement:

- `float nn_eval_cp(EvalNet* net, const Board* b)` → forward pass only
- `void nn_train_one(EvalNet* net, const Board* b, float target_cp, float lr)` → single training step

### 8.2 Feature encoding we used (simple baseline)
We discussed an easy encoding:

- **12 piece planes × 64 squares = 768 binary features**
  - `x[piece*64 + sq] = 1` if that piece occupies that square
- plus **side-to-move feature** `x[768] ∈ {0,1}`

So:
\[
x \in \mathbb{R}^{769}
\]

This is not the fastest possible representation, but it is conceptually clean.

### 8.3 Architecture we used for chess eval
- Input: 769
- Hidden1: 128 (ReLU)
- Hidden2: 64 (ReLU)
- Output: 1 (tanh), scaled to centipawns

Output:
\[
\hat{y} = \tanh(z_3) \in [-1,1]
\]
Then centipawns:
\[
\text{cp} = 1000\cdot \hat{y}
\]

### 8.4 Full one-file chess eval net in C (exact code)
```c
// chess_nn_eval.c
// gcc -O3 -march=native chess_nn_eval.c -lm -o chess_nn_eval
//
// This is a simple MLP for chess evaluation (not NNUE).
// - Input features: 12 piece planes x 64 squares + 1 side-to-move = 769
// - Hidden layers: 128, 64 (ReLU)
// - Output: 1 (tanh), scaled to centipawns
//
// You can train it from (position, target_eval) pairs, where target_eval
// could come from your search at depth N, or a dataset.
// This is intentionally self-contained and easy to modify.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ------------------- your engine-ish Board -------------------
typedef struct {
    uint64_t bb[12];     // bitboards for each piece type
    uint32_t gamestate;  // pack side-to-move, castling, etc.
    int ply;
} Board;

// Example piece indices (YOU MUST match your engine)
enum {
    WP=0, WN=1, WB=2, WR=3, WQ=4, WK=5,
    BP=6, BN=7, BB=8, BR=9, BQ=10, BK=11
};

// Example: assume gamestate bit0 = side to move (0=white,1=black)
// YOU MUST match your engine
static int side_to_move(const Board* b) {
    return (int)(b->gamestate & 1u);
}

// ------------------- RNG -------------------
static uint32_t g_rng = 1u;

static void seed_rng(uint32_t s) {
    if (s == 0u) s = 1u;
    g_rng = s;
}

static uint32_t rng_u32(void) {
    uint32_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return x;
}

static float rng_f01(void) {
    return (rng_u32() / 4294967296.0f);
}

static float rng_f_neg1_1(void) {
    return 2.0f * rng_f01() - 1.0f;
}

// ------------------- math helpers -------------------
static float relu(float x) { return x > 0.0f ? x : 0.0f; }
static float drelu_from_z(float z) { return z > 0.0f ? 1.0f : 0.0f; }

static float tanh_fast(float x) {
    // You can just use tanhf. Keeping wrapper for easy replacement.
    return tanhf(x);
}
static float dtanh_from_a(float a) {
    // if a = tanh(z), derivative is 1 - a^2
    return 1.0f - a*a;
}

// ------------------- network shape -------------------
#define IN_DIM   769
#define H1_DIM   128
#define H2_DIM   64
#define OUT_DIM  1

// Weight matrices are row-major: W[row][col] -> W[row*cols + col]
typedef struct {
    // Layer1: H1 x IN
    float W1[H1_DIM * IN_DIM];
    float b1[H1_DIM];

    // Layer2: H2 x H1
    float W2[H2_DIM * H1_DIM];
    float b2[H2_DIM];

    // Output: 1 x H2
    float W3[OUT_DIM * H2_DIM];
    float b3[OUT_DIM];

    // Forward buffers (store z and a for backprop)
    float z1[H1_DIM], a1[H1_DIM];
    float z2[H2_DIM], a2[H2_DIM];
    float z3[OUT_DIM], a3[OUT_DIM];

    // Gradient buffers
    float dW1[H1_DIM * IN_DIM], db1[H1_DIM];
    float dW2[H2_DIM * H1_DIM], db2[H2_DIM];
    float dW3[OUT_DIM * H2_DIM], db3[OUT_DIM];
} EvalNet;

// Xavier uniform init for a layer (fan_in, fan_out)
static void xavier_init(float* W, int rows, int cols) {
    int fan_in = cols;
    int fan_out = rows;
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < rows*cols; i++) {
        W[i] = rng_f_neg1_1() * limit;
    }
}

// Initialize all weights/biases
static void net_init(EvalNet* net) {
    xavier_init(net->W1, H1_DIM, IN_DIM);
    xavier_init(net->W2, H2_DIM, H1_DIM);
    xavier_init(net->W3, OUT_DIM, H2_DIM);
    memset(net->b1, 0, sizeof(net->b1));
    memset(net->b2, 0, sizeof(net->b2));
    memset(net->b3, 0, sizeof(net->b3));
}

// Zero gradients (for one-sample SGD; for minibatch you’d accumulate)
static void net_zero_grads(EvalNet* net) {
    memset(net->dW1, 0, sizeof(net->dW1));
    memset(net->db1, 0, sizeof(net->db1));
    memset(net->dW2, 0, sizeof(net->dW2));
    memset(net->db2, 0, sizeof(net->db2));
    memset(net->dW3, 0, sizeof(net->dW3));
    memset(net->db3, 0, sizeof(net->db3));
}

// Forward pass: input x -> output a3[0]
static float net_forward(EvalNet* net, const float* x) {
    // Layer 1: z1 = W1 x + b1, a1 = ReLU(z1)
    for (int i = 0; i < H1_DIM; i++) {
        float sum = net->b1[i];
        int row = i * IN_DIM;
        for (int j = 0; j < IN_DIM; j++) {
            sum += net->W1[row + j] * x[j];
        }
        net->z1[i] = sum;
        net->a1[i] = relu(sum);
    }

    // Layer 2: z2 = W2 a1 + b2, a2 = ReLU(z2)
    for (int i = 0; i < H2_DIM; i++) {
        float sum = net->b2[i];
        int row = i * H1_DIM;
        for (int j = 0; j < H1_DIM; j++) {
            sum += net->W2[row + j] * net->a1[j];
        }
        net->z2[i] = sum;
        net->a2[i] = relu(sum);
    }

    // Output: z3 = W3 a2 + b3, a3 = tanh(z3)
    {
        float sum = net->b3[0];
        for (int j = 0; j < H2_DIM; j++) {
            sum += net->W3[j] * net->a2[j]; // OUT_DIM=1 so row base = 0
        }
        net->z3[0] = sum;
        net->a3[0] = tanh_fast(sum);
    }

    return net->a3[0];
}

// Backprop for one sample.
// Target t is in [-1,1] (scaled). Loss = 0.5*(a3 - t)^2
static void net_backward_one(EvalNet* net, const float* x, float t) {
    // delta3 = dL/dz3 = (a3 - t) * dtanh(z3)
    float a3 = net->a3[0];
    float delta3 = (a3 - t) * dtanh_from_a(a3);

    // Grad for W3, b3:
    // dW3[j] += delta3 * a2[j]
    // db3 += delta3
    net->db3[0] += delta3;
    for (int j = 0; j < H2_DIM; j++) {
        net->dW3[j] += delta3 * net->a2[j];
    }

    // delta2 = (W3^T delta3) * dReLU(z2)
    float delta2[H2_DIM];
    for (int i = 0; i < H2_DIM; i++) {
        float back = net->W3[i] * delta3;               // since OUT_DIM=1
        delta2[i] = back * drelu_from_z(net->z2[i]);
    }

    // Grad for W2, b2:
    // dW2[i,j] += delta2[i] * a1[j]
    // db2[i] += delta2[i]
    for (int i = 0; i < H2_DIM; i++) {
        net->db2[i] += delta2[i];
        int row = i * H1_DIM;
        for (int j = 0; j < H1_DIM; j++) {
            net->dW2[row + j] += delta2[i] * net->a1[j];
        }
    }

    // delta1 = (W2^T delta2) * dReLU(z1)
    float delta1[H1_DIM];
    for (int i = 0; i < H1_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < H2_DIM; j++) {
            // W2[j,i] in row-major is W2[j*H1 + i]
            sum += net->W2[j * H1_DIM + i] * delta2[j];
        }
        delta1[i] = sum * drelu_from_z(net->z1[i]);
    }

    // Grad for W1, b1:
    // dW1[i,j] += delta1[i] * x[j]
    // db1[i] += delta1[i]
    for (int i = 0; i < H1_DIM; i++) {
        net->db1[i] += delta1[i];
        int row = i * IN_DIM;
        for (int j = 0; j < IN_DIM; j++) {
            net->dW1[row + j] += delta1[i] * x[j];
        }
    }
}

// SGD step
static void net_sgd_update(EvalNet* net, float lr) {
    for (int i = 0; i < H1_DIM*IN_DIM; i++) net->W1[i] -= lr * net->dW1[i];
    for (int i = 0; i < H1_DIM; i++)        net->b1[i] -= lr * net->db1[i];

    for (int i = 0; i < H2_DIM*H1_DIM; i++) net->W2[i] -= lr * net->dW2[i];
    for (int i = 0; i < H2_DIM; i++)        net->b2[i] -= lr * net->db2[i];

    for (int i = 0; i < H2_DIM; i++)        net->W3[i] -= lr * net->dW3[i];
    net->b3[0] -= lr * net->db3[0];
}

// Save/load (binary)
static int net_save(const EvalNet* net, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return 1;
    if (fwrite(net, sizeof(EvalNet), 1, f) != 1) { fclose(f); return 2; }
    fclose(f);
    return 0;
}
static int net_load(EvalNet* net, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 1;
    if (fread(net, sizeof(EvalNet), 1, f) != 1) { fclose(f); return 2; }
    fclose(f);
    return 0;
}

// ------------------- feature extraction: Board -> x[IN_DIM] -------------------
// We use 12*64 one-hot planes:
// x[piece*64 + sq] = 1 if that piece occupies sq else 0
// plus x[768] = side_to_move (0 or 1) (you can also do +/-1)
//
// IMPORTANT: this assumes your bitboards match the piece order enum above,
// and that bit index corresponds to squares consistently.
static void board_to_features(const Board* b, float* x) {
    memset(x, 0, sizeof(float) * IN_DIM);

    // 12 planes
    for (int p = 0; p < 12; p++) {
        uint64_t bb = b->bb[p];
        while (bb) {
            // extract least-significant 1 bit index
            uint64_t lsb = bb & (~bb + 1ull);
            int sq = (int)__builtin_ctzll(bb); // GCC/Clang; on MSVC use _tzcnt_u64 or custom
            x[p * 64 + sq] = 1.0f;
            bb ^= lsb;
        }
    }

    // side to move feature
    x[768] = (float)side_to_move(b);
}

// ------------------- eval API -------------------
// Convert tanh output in [-1,1] to centipawns by scaling.
// Pick a scale like 1000cp meaning tanh=1 ~ +1000cp.
static float nn_eval_cp(EvalNet* net, const Board* b) {
    float x[IN_DIM];
    board_to_features(b, x);
    float y = net_forward(net, x); // y in [-1,1]
    const float SCALE_CP = 1000.0f;
    return y * SCALE_CP;
}

// ------------------- training API -------------------
// Train on one example: (Board b, target_cp)
//
// We convert target_cp to target_tanh in [-1,1] by dividing by SCALE_CP
// and clamping to [-1,1]. Then do forward, backprop, SGD update.
static void nn_train_one(EvalNet* net, const Board* b, float target_cp, float lr) {
    const float SCALE_CP = 1000.0f;

    float x[IN_DIM];
    board_to_features(b, x);

    // scale and clamp target
    float t = target_cp / SCALE_CP;
    if (t > 1.0f) t = 1.0f;
    if (t < -1.0f) t = -1.0f;

    net_forward(net, x);
    net_zero_grads(net);
    net_backward_one(net, x, t);
    net_sgd_update(net, lr);
}

// ------------------- minimal demo main -------------------
int main(void) {
    seed_rng((uint32_t)time(NULL));

    EvalNet net;
    net_init(&net);

    // You would load a trained net here:
    // if (net_load(&net, "evalnet.bin") != 0) { net_init(&net); }

    // Demo: empty board stub (this does nothing meaningful)
    Board b;
    memset(&b, 0, sizeof(b));
    b.gamestate = 0; // white to move

    float cp = nn_eval_cp(&net, &b);
    printf("eval(empty) = %.2f cp\n", cp);

    // Example training call (nonsense target)
    nn_train_one(&net, &b, 50.0f, 1e-3f);

    // Save
    net_save(&net, "evalnet.bin");
    return 0;
}
```

---

## 9. Integrating into your engine (search + eval)

### 9.1 Inference integration (eval function)
Typical engine shape:

```c
// global net loaded at startup
static EvalNet g_net;

int eval_cp(const Board* b) {
    float cp = nn_eval_cp(&g_net, b);

    // Optional: blend with classical eval during transition
    // cp = 0.8f * cp + 0.2f * classical_eval_cp(b);

    return (int)cp;
}
```

### 9.2 Where training targets come from
You need pairs \((x, y)\): positions and target evaluations.

Two practical target sources you can generate yourself:

#### (A) Supervised: “train to match my search”
- Generate positions (from selfplay, random playouts, or PGNs).
- Run your search to depth D (or fixed nodes/time).
- Use the resulting score as `target_cp`.
- Train net to approximate that score.

This teaches the net to approximate your (expensive) search evaluation.

#### (B) TD learning (bootstrapping from successor states)
This is closer to RL but more complex:
- net predicts \(V(s)\)
- after move to \(s'\), update toward \(V(s')\)
- must handle terminal states (mate/stalemate)

You can do (A) first and get something working.

---

## 10. “Passive training” on your laptop
Nothing special is required. A “passive trainer” is just a long-running program:

- load model (or init)
- loop forever:
  - fetch/generate a training position
  - compute target via search or dataset
  - call `nn_train_one`
  - checkpoint every N steps

Optional: insert a sleep to reduce CPU usage.

---

## 11. Performance: why the naive MLP may be too slow in search

The naive MLP eval recomputes from scratch and costs roughly:

- Layer1: \(128 \times 769 \approx 98k\) multiply-adds
- Layer2: \(64 \times 128 = 8192\)
- Output: \(64\)

So ~100k ops per eval. In a search evaluating millions of nodes, this can dominate runtime.

This is why competitive engines use **NNUE**-style evaluation:
- sparse binary features
- incremental updates per move
- quantized int8/int16 arithmetic
- heavy SIMD optimization

---

## 12. Roadmap to NNUE-grade performance (engineering plan)

This section tells you exactly what you need beyond the baseline.

### 12.1 Feature design: sparse and incrementally updatable
NNUE uses features like:
- (king square, piece type, piece square)
so only a handful of features change on a move.

Key property: you can update hidden activations by **adding/subtracting** a small number of weight vectors when pieces move.

### 12.2 Quantization
Instead of float:
- weights in int8/int16
- accumulators in int32
- activation is clipped ReLU / piecewise

This gives big speedups on CPU and predictable evaluation cost.

### 12.3 SIMD kernels
Manually vectorize the critical dot products:
- AVX2/AVX-512 on x86
- use aligned loads, unrolled loops

### 12.4 Caching and batching (when applicable)
- Transposition table already caches eval per position hash indirectly.
- If you evaluate multiple positions at once (less common in alpha-beta), batching can help, but incremental NNUE is usually better.

---

## 13. Key correctness checks (debugging your implementation)

When implementing from scratch, you need sanity checks:

1. **Overfit a tiny dataset**:
   - If the net cannot drive training loss near zero on a tiny dataset, backprop is wrong.

2. **Finite difference gradient check**:
   For a single weight \(w\):
   \[
   \frac{\partial L}{\partial w} \approx \frac{L(w+\epsilon)-L(w-\epsilon)}{2\epsilon}
   \]
   Compare to backprop gradient.

3. **Activation saturation**:
   - If using sigmoid everywhere and weights are too large, gradients vanish.
   - ReLU hidden layers often train better.

---

## 14. Summary: what we established

- Training is typically **gradient descent + backprop**, not “keep good random weights.”
- A neural net is conceptually a weighted digraph, but implementation should be **dense matrices/vectors**, not node pointers.
- We built:
  - a generic MLP trainer in one C file (XOR)
  - a chess-engine-shaped eval net in one C file (bitboards → features → eval + train step)
- For serious chess performance, you will move toward:
  - NNUE-like sparse features
  - incremental update
  - quantization + SIMD

---

## 15. Next steps (concrete)

To adapt the chess eval code perfectly to your engine, you must pin down:

1. Square indexing convention (bit 0 = a1? h1?).
2. Exact mapping of `bb[12]` piece order to your enums.
3. Which bits in `gamestate` represent:
   - side to move
   - castling rights
   - en passant file (if any)

Then you can:
- implement `board_to_features` consistent with your engine
- integrate `nn_eval_cp` into your evaluation pipeline
- build a trainer that generates (position, search_score) pairs and trains overnight

If you want, the next iteration after this baseline should be:
- replace float features with a sparse list of active indices (because 768 one-hot features are sparse)
- compute the first layer as an accumulation of weight rows for active features (faster than full dot product)
- then proceed toward incremental update (NNUE direction)
