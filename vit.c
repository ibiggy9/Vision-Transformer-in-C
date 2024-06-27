#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper function to initialize weights
void initialize_weights(float *weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = ((float) rand() / RAND_MAX) * 2 - 1; // Random weights between -1 and 1
    }
}

// Helper function for layer normalization
void layer_norm(float *input, float *output, int size) {
    float mean = 0;
    float variance = 0;

    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;

    for (int i = 0; i < size; i++) {
        variance += (input[i] - mean) * (input[i] - mean);
    }
    variance /= size;

    float epsilon = 1e-5;
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - mean) / sqrt(variance + epsilon);
    }
}

// Patch Embedding with learned weights
void patch_embedding(float *input, float *patches, float *embedding_weights, int input_dim, int patch_size, int num_patches, int embedding_dim) {
    for (int i = 0; i < num_patches; i++) {
        for (int j = 0; j < embedding_dim; j++) {
            patches[i * embedding_dim + j] = 0;
            for (int k = 0; k < patch_size; k++) {
                patches[i * embedding_dim + j] += input[i * patch_size + k] * embedding_weights[k * embedding_dim + j];
            }
        }
    }
}

// Multi-Head Self-Attention
void multi_head_self_attention(float *patches, float *attention_output, float *attention_weights, int num_patches, int embedding_dim) {
    float *queries = (float *) malloc(num_patches * embedding_dim * sizeof(float));
    float *keys = (float *) malloc(num_patches * embedding_dim * sizeof(float));
    float *values = (float *) malloc(num_patches * embedding_dim * sizeof(float));

    // Compute queries, keys, and values
    for (int i = 0; i < num_patches; i++) {
        for (int j = 0; j < embedding_dim; j++) {
            queries[i * embedding_dim + j] = 0;
            keys[i * embedding_dim + j] = 0;
            values[i * embedding_dim + j] = 0;
            for (int k = 0; k < embedding_dim; k++) {
                queries[i * embedding_dim + j] += patches[i * embedding_dim + k] * attention_weights[k * embedding_dim + j];
                keys[i * embedding_dim + j] += patches[i * embedding_dim + k] * attention_weights[k * embedding_dim + j];
                values[i * embedding_dim + j] += patches[i * embedding_dim + k] * attention_weights[k * embedding_dim + j];
            }
        }
    }

    // Attention mechanism
    for (int i = 0; i < num_patches; i++) {
        for (int j = 0; j < embedding_dim; j++) {
            attention_output[i * embedding_dim + j] = 0;
            for (int k = 0; k < num_patches; k++) {
                float score = 0;
                for (int l = 0; l < embedding_dim; l++) {
                    score += queries[i * embedding_dim + l] * keys[k * embedding_dim + l];
                }
                float softmax_score = exp(score) / num_patches; // Simplified softmax
                attention_output[i * embedding_dim + j] += softmax_score * values[k * embedding_dim + j];
            }
        }
    }

    free(queries);
    free(keys);
    free(values);
}

// MLP Head
void mlp_head(float *attention_output, float *output, float *mlp_weights, int num_patches, int embedding_dim) {
    int hidden_dim = embedding_dim; // Simplified MLP with single layer

    for (int i = 0; i < num_patches; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            output[i * hidden_dim + j] = 0;
            for (int k = 0; k < embedding_dim; k++) {
                output[i * hidden_dim + j] += attention_output[i * embedding_dim + k] * mlp_weights[k * hidden_dim + j];
            }
        }
    }
}

// Load preprocessed data
void load_data(float **data, int *labels, int num_samples, const char *data_directory) {
    char filename[256];
    for (int i = 0; i < num_samples; i++) {
        snprintf(filename, sizeof(filename), "%s/%d.npy", data_directory, i);
        FILE *file = fopen(filename, "rb");
        if (file == NULL) {
            printf("Error opening file: %s\n", filename);
            exit(1);
        }
        fread(data[i], sizeof(float), 1025 * 94, file);
        fclose(file);

        snprintf(filename, sizeof(filename), "%s/%d_label.npy", data_directory, i);
        file = fopen(filename, "rb");
        if (file == NULL) {
            printf("Error opening label file: %s\n", filename);
            exit(1);
        }
        fread(&labels[i], sizeof(int), 1, file);
        fclose(file);
    }
}

int main() {
    int input_dim = 1025 * 94;
    int patch_size = 4;
    int num_patches = input_dim / patch_size;
    int embedding_dim = 8;
    int num_samples = 100; // Number of samples to load
    const char *data_directory = "path/to/your/data_directory"; // Directory with preprocessed data

    float **data = (float **) malloc(num_samples * sizeof(float *));
    for (int i = 0; i < num_samples; i++) {
        data[i] = (float *) malloc(input_dim * sizeof(float));
    }
    int *labels = (int *) malloc(num_samples * sizeof(int));

    // Load preprocessed data
    load_data(data, labels, num_samples, data_directory);

    float *patches = (float *) malloc(num_patches * embedding_dim * sizeof(float));
    float *attention_output = (float *) malloc(num_patches * embedding_dim * sizeof(float));
    float *output = (float *) malloc(num_patches * embedding_dim * sizeof(float));

    float *embedding_weights = (float *) malloc(patch_size * embedding_dim * sizeof(float));
    float *attention_weights = (float *) malloc(embedding_dim * embedding_dim * sizeof(float));
    float *mlp_weights = (float *) malloc(embedding_dim * embedding_dim * sizeof(float));

    initialize_weights(embedding_weights, patch_size * embedding_dim);
    initialize_weights(attention_weights, embedding_dim * embedding_dim);
    initialize_weights(mlp_weights, embedding_dim * embedding_dim);

    for (int i = 0; i < num_samples; i++) {
        patch_embedding(data[i], patches, embedding_weights, input_dim, patch_size, num_patches, embedding_dim);
        multi_head_self_attention(patches, attention_output, attention_weights, num_patches, embedding_dim);
        mlp_head(attention_output, output, mlp_weights, num_patches, embedding_dim);

        // Layer normalization
        float *normalized_output = (float *) malloc(num_patches * embedding_dim * sizeof(float));
        layer_norm(output, normalized_output, num_patches * embedding_dim);

        // Print output for the first sample
        if (i == 0) {
            for (int j = 0; j < num_patches; j++) {
                for (int k = 0; k < embedding_dim; k++) {
                    printf("%f ", normalized_output[j * embedding_dim + k]);
                }
                printf("\n");
            }
        }

        free(normalized_output);
    }

    for (int i = 0; i < num_samples; i++) {
        free(data[i]);
    }
    free(data);
    free(labels);
    free(patches);
    free(attention_output);
    free(output);
    free(embedding_weights);
    free(attention_weights);
    free(mlp_weights);

    return 0;
}
