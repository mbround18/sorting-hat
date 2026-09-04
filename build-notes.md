# Build notes (this machine)

GPU build:

    cargo build --release --features cuda

`.cargo/config.toml` pins the three things that otherwise break the build here:
libclang for bindgen, GCC 12 as the CUDA host compiler (CUDA 12.0 rejects 13+),
and `sm_86` so only the RTX 3090 Ti's architecture is compiled.

CPU-only llama build: `cargo build --release --features llama`.
No model at all: the default build, which uses the heuristic backend.
