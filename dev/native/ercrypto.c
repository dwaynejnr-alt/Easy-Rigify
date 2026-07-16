/*
 * ercrypto.c - native model-crypto core for Easy Rigify (abi3 / Py_LIMITED_API).
 *
 * Exposes ONE function to Python:
 *     _ercrypto.transform(data: bytes) -> bytes
 * which XORs `data` with a SHA-256 counter-mode keystream derived from a key
 * that lives ONLY inside this compiled binary. The transform is symmetric, so
 * the same call both encrypts and decrypts a model body.
 *
 * The key is never a plaintext literal: it is rebuilt at runtime from masked
 * byte tables (so `strings` on the binary reveals nothing) and then hashed.
 * This is deliberately the LIMITED API (works on any CPython >= 3.7 of the
 * matching platform), so one binary per OS/arch loads in every Blender.
 *
 * NOT shipped as source: this file lives under dev/ (build-excluded). Only the
 * compiled binaries in native/ ship.
 */
#define Py_LIMITED_API 0x03070000
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ SHA-256 */
typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    size_t   buflen;
} sha256_ctx;

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define ROTR(x,n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha256_transform(sha256_ctx *c, const uint8_t *p) {
    uint32_t w[64], a,b,cc,d,e,f,g,h,t1,t2;
    int i;
    for (i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[i*4]<<24)|((uint32_t)p[i*4+1]<<16)|((uint32_t)p[i*4+2]<<8)|((uint32_t)p[i*4+3]);
    for (i = 16; i < 64; i++) {
        uint32_t s0 = ROTR(w[i-15],7) ^ ROTR(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROTR(w[i-2],17) ^ ROTR(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a=c->state[0];b=c->state[1];cc=c->state[2];d=c->state[3];
    e=c->state[4];f=c->state[5];g=c->state[6];h=c->state[7];
    for (i = 0; i < 64; i++) {
        uint32_t S1 = ROTR(e,6) ^ ROTR(e,11) ^ ROTR(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        t1 = h + S1 + ch + K256[i] + w[i];
        uint32_t S0 = ROTR(a,2) ^ ROTR(a,13) ^ ROTR(a,22);
        uint32_t maj = (a & b) ^ (a & cc) ^ (b & cc);
        t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=cc; cc=b; b=a; a=t1+t2;
    }
    c->state[0]+=a;c->state[1]+=b;c->state[2]+=cc;c->state[3]+=d;
    c->state[4]+=e;c->state[5]+=f;c->state[6]+=g;c->state[7]+=h;
}

static void sha256_init(sha256_ctx *c) {
    c->state[0]=0x6a09e667;c->state[1]=0xbb67ae85;c->state[2]=0x3c6ef372;c->state[3]=0xa54ff53a;
    c->state[4]=0x510e527f;c->state[5]=0x9b05688c;c->state[6]=0x1f83d9ab;c->state[7]=0x5be0cd19;
    c->bitlen=0; c->buflen=0;
}

static void sha256_update(sha256_ctx *c, const uint8_t *data, size_t len) {
    size_t i;
    for (i = 0; i < len; i++) {
        c->buf[c->buflen++] = data[i];
        if (c->buflen == 64) {
            sha256_transform(c, c->buf);
            c->bitlen += 512;
            c->buflen = 0;
        }
    }
}

static void sha256_final(sha256_ctx *c, uint8_t out[32]) {
    uint64_t bits = c->bitlen + (uint64_t)c->buflen * 8;
    size_t i = c->buflen;
    c->buf[i++] = 0x80;
    if (i > 56) {
        while (i < 64) c->buf[i++] = 0;
        sha256_transform(c, c->buf);
        i = 0;
    }
    while (i < 56) c->buf[i++] = 0;
    for (int j = 7; j >= 0; j--) c->buf[i++] = (uint8_t)(bits >> (j*8));
    sha256_transform(c, c->buf);
    for (int j = 0; j < 8; j++) {
        out[j*4]   = (uint8_t)(c->state[j] >> 24);
        out[j*4+1] = (uint8_t)(c->state[j] >> 16);
        out[j*4+2] = (uint8_t)(c->state[j] >> 8);
        out[j*4+3] = (uint8_t)(c->state[j]);
    }
}

/* --------------------------------------------------------------- key material
 * The secret is stored masked so it is not a readable string in the binary.
 * Each byte was produced as  secret[i] ^ MASK(i)  where MASK(i) = 0x5B + 7*i
 * (mod 256). Reconstructed and hashed at runtime; the raw 32-byte key never
 * appears as a constant. (This is obfuscation, not a substitute for the fact
 * that a determined reverser with a debugger can still recover it - see the
 * design notes. Its job is to force binary reversing over trivial reuse.)
 */
static const uint8_t SECRET_MASKED[] = {
    /* masked with 0x5B+7*i; regenerate via dev/native/_refgen.py --emit-c */
    0x1E,0x03,0x1A,0x09,0x25,0x17,0xE2,0xE5,0xF5,0xE3,0x9B,0x92,0xCA,0xC4,0xDE,0xB6,
    0xB2,0xA2,0xAD,0x8F,0xDD,0xD4,0x83,0xCE,0x39,0x30,0x7C,0x77,0x7B,0x43,0x41,0x19,
    0x50,0x27,0x30,0x6A,0x6D,0x6C,0x55,0x5E,0x45
};

static void derive_key(uint8_t key[32]) {
    size_t n = sizeof(SECRET_MASKED);
    uint8_t secret[64];
    for (size_t i = 0; i < n; i++)
        secret[i] = (uint8_t)(SECRET_MASKED[i] ^ (uint8_t)(0x5B + 7 * i));
    sha256_ctx c;
    sha256_init(&c);
    sha256_update(&c, secret, n);
    sha256_final(&c, key);
    memset(secret, 0, sizeof(secret));
}

/* -------------------------------------------------------------- keystream XOR
 * Matches the reference algorithm: for block index i, the 32-byte keystream
 * block is sha256(key || le64(i)); the plaintext/ciphertext is XORed with the
 * concatenated blocks.
 */
static void apply_stream(uint8_t *data, size_t n, const uint8_t key[32]) {
    uint8_t block[32];
    uint64_t counter = 0;
    size_t off = 0;
    while (off < n) {
        sha256_ctx c;
        sha256_init(&c);
        sha256_update(&c, key, 32);
        uint8_t le[8];
        for (int j = 0; j < 8; j++) le[j] = (uint8_t)(counter >> (j*8));  /* little-endian */
        sha256_update(&c, le, 8);
        sha256_final(&c, block);
        size_t take = n - off;
        if (take > 32) take = 32;
        for (size_t j = 0; j < take; j++) data[off + j] ^= block[j];
        off += take;
        counter++;
    }
    memset(block, 0, sizeof(block));
}

/* --------------------------------------------------------------- Python glue */
static PyObject *py_transform(PyObject *self, PyObject *args) {
    const char *in;
    Py_ssize_t n;
    if (!PyArg_ParseTuple(args, "y#", &in, &n))
        return NULL;
    PyObject *out = PyBytes_FromStringAndSize(NULL, n);
    if (out == NULL)
        return NULL;
    char *buf = PyBytes_AsString(out);
    if (buf == NULL) { Py_DECREF(out); return NULL; }
    if (n > 0) memcpy(buf, in, (size_t)n);
    uint8_t key[32];
    derive_key(key);
    apply_stream((uint8_t *)buf, (size_t)n, key);
    memset(key, 0, sizeof(key));
    return out;
}

static PyMethodDef methods[] = {
    {"transform", py_transform, METH_VARARGS,
     "transform(data: bytes) -> bytes : XOR data with the model keystream (symmetric)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_ercrypto", NULL, -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__ercrypto(void) {
    return PyModule_Create(&moduledef);
}
