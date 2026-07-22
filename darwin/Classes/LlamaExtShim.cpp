#include "LlamaExtShim.h"

// Prototypes copied from llama.cpp src/llama-ext.h at the pinned LLAMA_REF.
// They are deliberately re-declared here (instead of including llama-ext.h,
// which is not shipped in the xcframework headers) with the same C++ linkage,
// so the references below mangle to the symbols exported by llama.framework.
struct llama_context;

void llama_set_embeddings_nextn(struct llama_context *ctx, bool value, bool masked);
float *llama_get_embeddings_nextn_ith(struct llama_context *ctx, int32_t i);

extern "C" void llama_ext_set_embeddings_nextn(void *ctx, bool value, bool masked) {
  llama_set_embeddings_nextn(static_cast<struct llama_context *>(ctx), value, masked);
}

extern "C" const float *llama_ext_get_embeddings_nextn_ith(void *ctx, int32_t i) {
  return llama_get_embeddings_nextn_ith(static_cast<struct llama_context *>(ctx), i);
}
