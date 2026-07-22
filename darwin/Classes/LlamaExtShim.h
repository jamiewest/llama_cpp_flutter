#ifndef LlamaExtShim_h
#define LlamaExtShim_h

#include <stdbool.h>
#include <stdint.h>

// C-callable wrappers over llama.cpp's *staging* API (src/llama-ext.h).
//
// The staging header is C++ (it pulls in <map>), so its symbols carry C++
// linkage and the header itself cannot be vendored into the Swift-importable
// llama.framework module. These wrappers re-export the two functions the MTP
// (multi-token-prediction) speculative-decoding path needs with C linkage;
// LlamaExtShim.cpp declares the mangled originals and forwards to them.
//
// Staging means exactly that: signatures are pinned to the LLAMA_CPP_TAG in
// tool/versions.env (see CLAUDE.md § Staging ABI) and are re-checked against
// src/llama-ext.h by tool/check_llama_ext_abi.sh whenever that pin moves.
//
// `ctx` is a `struct llama_context *` passed as `void *` so this header does
// not redeclare types owned by the llama module.

#ifdef __cplusplus
extern "C" {
#endif

// Whether decodes on `ctx` also output "nextn" embeddings — the hidden state
// before the final output norm, which the MTP drafter consumes as input.
// `masked` limits output to tokens with `batch.logits != 0` (and sizes the
// output buffer like the logits buffer, instead of by full batch width).
void llama_ext_set_embeddings_nextn(void *ctx, bool value, bool masked);

// The nextn embedding row for the i-th batch token of the last decode
// (negative i indexes from the end, mirroring llama_get_logits_ith). Returns
// NULL on invalid indices. Valid until the next decode on `ctx`.
const float *llama_ext_get_embeddings_nextn_ith(void *ctx, int32_t i);

#ifdef __cplusplus
}
#endif

#endif /* LlamaExtShim_h */
