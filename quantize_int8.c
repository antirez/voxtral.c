/*
 * quantize_int8.c - Quantize safetensors in a model directory
 *
 * Usage: ./quantize_int8 <input_dir> <output_dir>
 *
 * Converts BF16 2D weight tensors to INT8 with per-tensor scale factors.
 * Copies non-safetensor files to output directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <math.h>

/* Constants */

#define MAX_TENSORS 2048
#define MAX_FILES 256

/* Patterns to keep as BF16 (not quantize) */
static const char *KEEP_BF16_PATTERNS[] = {
    "tok_embeddings",
    ".w2.",
    "norm.weight",
    "ada_rms_norm_t_cond",
    "audio_language_projection",
    "whisper_encoder",
    NULL
};

/* Safetensors Reader */

typedef enum {
    DTYPE_F32 = 0,
    DTYPE_F16 = 1,
    DTYPE_BF16 = 2,
    DTYPE_I32 = 3,
    DTYPE_I64 = 4,
    DTYPE_BOOL = 5,
    DTYPE_I8 = 6,
    DTYPE_UNKNOWN = -1
} dtype_t;

typedef struct {
    char name[256];
    dtype_t dtype;
    int ndim;
    int64_t shape[8];
    size_t data_offset;
    size_t data_size;
} tensor_info_t;

typedef struct {
    void *data;
    size_t file_size;
    size_t header_size;
    int num_tensors;
    tensor_info_t tensors[MAX_TENSORS];
} safetensors_t;

static void skip_ws(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

static int parse_str(const char **p, char *out, size_t max) {
    skip_ws(p);
    if (**p != '"') return -1;
    (*p)++;
    size_t i = 0;
    while (**p && **p != '"' && i < max - 1) {
        if (**p == '\\') {
            (*p)++;
            if (**p == 'n') out[i++] = '\n';
            else if (**p == '"') out[i++] = '"';
            else if (**p == '\\') out[i++] = '\\';
            else out[i++] = **p;
        } else {
            out[i++] = **p;
        }
        (*p)++;
    }
    out[i] = '\0';
    if (**p != '"') return -1;
    (*p)++;
    return 0;
}

static int64_t parse_int(const char **p) {
    skip_ws(p);
    int64_t val = 0;
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

static dtype_t parse_dtype(const char *s) {
    if (strcmp(s, "F32") == 0) return DTYPE_F32;
    if (strcmp(s, "F16") == 0) return DTYPE_F16;
    if (strcmp(s, "BF16") == 0) return DTYPE_BF16;
    if (strcmp(s, "I32") == 0) return DTYPE_I32;
    if (strcmp(s, "I64") == 0) return DTYPE_I64;
    if (strcmp(s, "BOOL") == 0) return DTYPE_BOOL;
    if (strcmp(s, "I8") == 0) return DTYPE_I8;
    return DTYPE_UNKNOWN;
}

static int parse_tensor_entry(const char **p, tensor_info_t *t) {
    skip_ws(p);
    if (**p != '{') return -1;
    (*p)++;

    t->dtype = DTYPE_UNKNOWN;
    t->ndim = 0;
    t->data_offset = 0;
    t->data_size = 0;

    while (**p && **p != '}') {
        skip_ws(p);
        if (**p == ',') { (*p)++; continue; }

        char key[64];
        if (parse_str(p, key, sizeof(key)) != 0) return -1;

        skip_ws(p);
        if (**p != ':') return -1;
        (*p)++;
        skip_ws(p);

        if (strcmp(key, "dtype") == 0) {
            char dtype_str[32];
            if (parse_str(p, dtype_str, sizeof(dtype_str)) != 0) return -1;
            t->dtype = parse_dtype(dtype_str);
        } else if (strcmp(key, "shape") == 0) {
            if (**p != '[') return -1;
            (*p)++;
            t->ndim = 0;
            while (**p && **p != ']' && t->ndim < 8) {
                skip_ws(p);
                if (**p == ',') { (*p)++; continue; }
                t->shape[t->ndim++] = parse_int(p);
            }
            if (**p == ']') (*p)++;
        } else if (strcmp(key, "data_offsets") == 0) {
            if (**p != '[') return -1;
            (*p)++;
            skip_ws(p);
            size_t start = (size_t)parse_int(p);
            skip_ws(p);
            if (**p == ',') (*p)++;
            skip_ws(p);
            size_t end = (size_t)parse_int(p);
            t->data_offset = start;
            t->data_size = end - start;
            skip_ws(p);
            if (**p == ']') (*p)++;
        } else {
            /* Skip unknown value */
            if (**p == '"') {
                (*p)++;
                while (**p && **p != '"') {
                    if (**p == '\\') (*p)++;
                    if (**p) (*p)++;
                }
                if (**p == '"') (*p)++;
            } else if (**p == '[' || **p == '{') {
                char open = **p, close = (open == '[') ? ']' : '}';
                int depth = 1;
                (*p)++;
                while (**p && depth > 0) {
                    if (**p == open) depth++;
                    else if (**p == close) depth--;
                    (*p)++;
                }
            } else {
                while (**p && **p != ',' && **p != '}') (*p)++;
            }
        }
    }
    if (**p == '}') (*p)++;
    return 0;
}

static safetensors_t *st_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }

    size_t file_size = (size_t)st.st_size;
    if (file_size < 8) { close(fd); return NULL; }

    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) return NULL;

    uint64_t header_size;
    memcpy(&header_size, data, 8);

    if (header_size > file_size - 8) {
        munmap(data, file_size);
        return NULL;
    }

    char *json = malloc(header_size + 1);
    memcpy(json, (char *)data + 8, header_size);
    json[header_size] = '\0';

    safetensors_t *sf = calloc(1, sizeof(safetensors_t));
    sf->data = data;
    sf->file_size = file_size;
    sf->header_size = header_size;

    /* Parse header */
    const char *p = json;
    skip_ws(&p);
    if (*p != '{') { free(json); munmap(data, file_size); free(sf); return NULL; }
    p++;

    while (*p && *p != '}' && sf->num_tensors < MAX_TENSORS) {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        char name[256];
        if (parse_str(&p, name, sizeof(name)) != 0) break;

        skip_ws(&p);
        if (*p != ':') break;
        p++;

        if (strcmp(name, "__metadata__") == 0) {
            skip_ws(&p);
            if (*p == '{') {
                int depth = 1;
                p++;
                while (*p && depth > 0) {
                    if (*p == '{') depth++;
                    else if (*p == '}') depth--;
                    p++;
                }
            }
            continue;
        }

        tensor_info_t *t = &sf->tensors[sf->num_tensors];
        snprintf(t->name, sizeof(t->name), "%s", name);
        if (parse_tensor_entry(&p, t) != 0) break;
        sf->num_tensors++;
    }

    free(json);
    return sf;
}

static void st_close(safetensors_t *sf) {
    if (!sf) return;
    if (sf->data) munmap(sf->data, sf->file_size);
    free(sf);
}

static const void *st_data(const safetensors_t *sf, const tensor_info_t *t) {
    return (const char *)sf->data + 8 + sf->header_size + t->data_offset;
}

static int64_t tensor_numel(const tensor_info_t *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

/* BF16 conversion */

static float bf16_to_f32(uint16_t bf16) {
    uint32_t f32 = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

/* Quantization */

static int should_quantize(const char *name) {
    for (int i = 0; KEEP_BF16_PATTERNS[i]; i++) {
        if (strstr(name, KEEP_BF16_PATTERNS[i])) return 0;
    }
    return 1;
}

typedef struct {
    char name[256];
    dtype_t dtype;
    int ndim;
    int64_t shape[8];
    void *data;
    size_t data_size;
} output_tensor_t;

static output_tensor_t *out_tensors = NULL;
static int num_out_tensors = 0;

static void add_output_tensor(const char *name, dtype_t dtype, int ndim,
                              const int64_t *shape, const void *data, size_t size) {
    output_tensor_t *t = &out_tensors[num_out_tensors++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->dtype = dtype;
    t->ndim = ndim;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    t->data = malloc(size);
    memcpy(t->data, data, size);
    t->data_size = size;
}

static void quantize_bf16_to_int8(const uint16_t *bf16, int64_t numel,
                                   int8_t *out_int8, float *out_scale) {
    /* Find abs max */
    float abs_max = 0.0f;
    for (int64_t i = 0; i < numel; i++) {
        float v = bf16_to_f32(bf16[i]);
        float a = fabsf(v);
        if (a > abs_max) abs_max = a;
    }

    if (abs_max == 0.0f) {
        *out_scale = 1.0f;
        memset(out_int8, 0, numel);
        return;
    }

    float scale = abs_max / 127.0f;
    *out_scale = scale;

    for (int64_t i = 0; i < numel; i++) {
        float v = bf16_to_f32(bf16[i]);
        float scaled = v / scale;
        int32_t rounded = (int32_t)roundf(scaled);
        if (rounded < -128) rounded = -128;
        if (rounded > 127) rounded = 127;
        out_int8[i] = (int8_t)rounded;
    }
}

/* Safetensors Writer */

static const char *dtype_str(dtype_t d) {
    switch (d) {
        case DTYPE_F32: return "F32";
        case DTYPE_F16: return "F16";
        case DTYPE_BF16: return "BF16";
        case DTYPE_I32: return "I32";
        case DTYPE_I64: return "I64";
        case DTYPE_BOOL: return "BOOL";
        case DTYPE_I8: return "I8";
        default: return "UNKNOWN";
    }
}

static int write_safetensors(const char *path) {
    /* Build JSON header */
    size_t json_cap = 1024 * 1024;  /* 1MB should be enough */
    char *json = malloc(json_cap);
    size_t json_len = 0;
    size_t data_offset = 0;

    json_len += snprintf(json + json_len, json_cap - json_len, "{");

    for (int i = 0; i < num_out_tensors; i++) {
        output_tensor_t *t = &out_tensors[i];

        if (i > 0) json_len += snprintf(json + json_len, json_cap - json_len, ",");

        json_len += snprintf(json + json_len, json_cap - json_len,
                             "\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                             t->name, dtype_str(t->dtype));

        for (int j = 0; j < t->ndim; j++) {
            if (j > 0) json_len += snprintf(json + json_len, json_cap - json_len, ",");
            json_len += snprintf(json + json_len, json_cap - json_len, "%lld",
                                 (long long)t->shape[j]);
        }

        size_t end_offset = data_offset + t->data_size;
        json_len += snprintf(json + json_len, json_cap - json_len,
                             "],\"data_offsets\":[%zu,%zu]}",
                             data_offset, end_offset);
        data_offset = end_offset;
    }

    json_len += snprintf(json + json_len, json_cap - json_len, "}");

    /* Write file */
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s for writing\n", path);
        free(json);
        return -1;
    }

    /* 8-byte header size */
    uint64_t header_size = json_len;
    fwrite(&header_size, 8, 1, f);

    /* JSON header */
    fwrite(json, 1, json_len, f);

    /* Tensor data */
    for (int i = 0; i < num_out_tensors; i++) {
        fwrite(out_tensors[i].data, 1, out_tensors[i].data_size, f);
    }

    fclose(f);
    free(json);
    return 0;
}

/* File operations */

static int copy_file(const char *src, const char *dst) {
    FILE *in = fopen(src, "rb");
    if (!in) return -1;

    FILE *out = fopen(dst, "wb");
    if (!out) { fclose(in); return -1; }

    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
        fwrite(buf, 1, n, out);
    }

    fclose(in);
    fclose(out);
    return 0;
}

static int ends_with(const char *str, const char *suffix) {
    size_t slen = strlen(str);
    size_t suflen = strlen(suffix);
    if (suflen > slen) return 0;
    return strcmp(str + slen - suflen, suffix) == 0;
}

/* Main */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_dir> <output_dir>\n", argv[0]);
        return 1;
    }

    const char *input_dir = argv[1];
    const char *output_dir = argv[2];

    /* Verify input is a directory */
    struct stat st;
    if (stat(input_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "Error: %s is not a directory\n", input_dir);
        return 1;
    }

    /* Create output directory */
    mkdir(output_dir, 0755);

    /* Scan input directory */
    DIR *dir = opendir(input_dir);
    if (!dir) {
        fprintf(stderr, "Error: cannot open directory %s\n", input_dir);
        return 1;
    }

    char *st_files[MAX_FILES];
    char *other_files[MAX_FILES];
    int num_st_files = 0;
    int num_other_files = 0;

    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;

        char path[512];
        snprintf(path, sizeof(path), "%s/%s", input_dir, ent->d_name);

        struct stat fst;
        if (stat(path, &fst) != 0 || !S_ISREG(fst.st_mode)) continue;

        if (ends_with(ent->d_name, ".safetensors")) {
            st_files[num_st_files++] = strdup(ent->d_name);
        } else {
            other_files[num_other_files++] = strdup(ent->d_name);
        }
    }
    closedir(dir);

    if (num_st_files == 0) {
        fprintf(stderr, "Error: No .safetensors files found in %s\n", input_dir);
        return 1;
    }

    /* Copy non-safetensor files */
    for (int i = 0; i < num_other_files; i++) {
        char src[512], dst[512];
        snprintf(src, sizeof(src), "%s/%s", input_dir, other_files[i]);
        snprintf(dst, sizeof(dst), "%s/%s", output_dir, other_files[i]);
        if (copy_file(src, dst) == 0) {
            printf("Copied: %s\n", other_files[i]);
        }
    }

    /* Allocate output tensor array */
    out_tensors = calloc(MAX_TENSORS, sizeof(output_tensor_t));
    num_out_tensors = 0;

    /* Statistics */
    int stat_int8 = 0, stat_bf16 = 0, stat_f32 = 0;
    size_t bytes_original = 0, bytes_quantized = 0;

    /* Process safetensor files */
    for (int fi = 0; fi < num_st_files; fi++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", input_dir, st_files[fi]);
        printf("Processing: %s\n", st_files[fi]);

        safetensors_t *sf = st_open(path);
        if (!sf) {
            fprintf(stderr, "Error: cannot open %s\n", path);
            continue;
        }

        for (int ti = 0; ti < sf->num_tensors; ti++) {
            tensor_info_t *t = &sf->tensors[ti];
            const void *data = st_data(sf, t);
            int64_t numel = tensor_numel(t);

            size_t elem_size = (t->dtype == DTYPE_F32) ? 4 :
                               (t->dtype == DTYPE_BF16 || t->dtype == DTYPE_F16) ? 2 :
                               (t->dtype == DTYPE_I8) ? 1 : 4;
            bytes_original += numel * elem_size;

            /* Quantize 2D BF16 tensors that match criteria */
            if (t->ndim == 2 && t->dtype == DTYPE_BF16 && should_quantize(t->name)) {
                int8_t *int8_data = malloc(numel);
                float scale;

                quantize_bf16_to_int8((const uint16_t *)data, numel, int8_data, &scale);

                /* Add quantized tensor */
                add_output_tensor(t->name, DTYPE_I8, t->ndim, t->shape,
                                  int8_data, numel);

                /* Add scale tensor */
                int64_t scale_shape[1] = {1};
                add_output_tensor(t->name, DTYPE_F32, 1, scale_shape, &scale, 4);
                /* Rename to add _scale suffix */
                snprintf(out_tensors[num_out_tensors - 1].name,
                         sizeof(out_tensors[num_out_tensors - 1].name),
                         "%s_scale", t->name);

                bytes_quantized += numel + 4;  /* INT8 + F32 scale */
                stat_int8++;
                free(int8_data);
            } else {
                /* Keep original */
                add_output_tensor(t->name, t->dtype, t->ndim, t->shape,
                                  data, t->data_size);
                bytes_quantized += t->data_size;

                if (t->dtype == DTYPE_BF16) stat_bf16++;
                else stat_f32++;
            }
        }

        st_close(sf);
    }

    /* Write output safetensors */
    char out_path[512];
    snprintf(out_path, sizeof(out_path), "%s/consolidated.safetensors", output_dir);

    if (write_safetensors(out_path) != 0) {
        fprintf(stderr, "Error: failed to write output\n");
        return 1;
    }

    /* Print summary */
    double original_mb = bytes_original / (1024.0 * 1024.0);
    double quantized_mb = bytes_quantized / (1024.0 * 1024.0);

    printf("\n=== Quantization Summary ===\n");
    printf("INT8 tensors:  %d\n", stat_int8);
    printf("BF16 tensors:  %d\n", stat_bf16);
    printf("F32 tensors:   %d\n", stat_f32);
    printf("Original size: %.1f MB\n", original_mb);
    printf("Quantized:     %.1f MB\n", quantized_mb);
    printf("Compression:   %.2fx\n", original_mb / quantized_mb);
    printf("Output:        %s\n", out_path);
    printf("Other files:   %d copied\n", num_other_files);

    /* Cleanup */
    for (int i = 0; i < num_out_tensors; i++) {
        free(out_tensors[i].data);
    }
    free(out_tensors);
    for (int i = 0; i < num_st_files; i++) free(st_files[i]);
    for (int i = 0; i < num_other_files; i++) free(other_files[i]);

    return 0;
}
