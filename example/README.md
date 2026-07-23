# llama_cpp_flutter_example

Demo chat app for `llama_cpp_flutter`: a
[flutter_ai_toolkit](https://pub.dev/packages/flutter_ai_toolkit) chat UI
backed by a local GGUF model through the
[`agents`](https://pub.dev/packages/agents) framework, with in-app screens
for configuring the **model** (preset or custom GGUF URL, context size, GPU
offload, thinking) and the **agent** (persona/system prompt, sampling,
demo tools). The same Dart code runs natively on macOS/iOS and in the
browser via wllama (Wasm).

## Run it

Native (Metal):

```sh
flutter run -d macos
```

Web:

```sh
flutter run -d chrome \
  --web-header=Cross-Origin-Opener-Policy=same-origin \
  --web-header=Cross-Origin-Embedder-Policy=require-corp
```

The COOP/COEP headers make the page cross-origin isolated, which wllama
needs for multi-threaded inference; without them the app still works but
falls back to a single thread (the app shows a banner when that happens).
When deploying a web build, configure the same two headers on your server.
For hosts that can't set headers (GitHub Pages), `web/index.html` loads the
vendored [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker)
shim, which injects them via a service worker and reloads the page once;
on servers that already send the headers it does nothing.

`.github/workflows/deploy-pages.yml` builds this app and deploys it to
GitHub Pages (`https://<owner>.github.io/<repo>/`) on every push to main.
One-time setup: repo Settings → Pages → Source → "GitHub Actions".

`web/index.html` imports `@wllama/wllama`'s ESM bundle from jsDelivr and
exposes `globalThis.Wllama`, which the plugin's web runtime requires. Keep
that JS version in sync with the `wllama.wasm` asset bundled by the plugin.

Models download on first load and are cached (native: application-support
directory; web: browser cache / OPFS). Tool calling works with the
tool-trained presets (Qwen3, Llama 3.2, LFM2) — try "what time is it?" or
"roll a 20-sided die".

## CI link smoke test

CI builds this app on macOS so every PR exercises compile + link of the
plugin's Swift/C++ against the pinned vendored `llama.xcframework`.

Run a real model end-to-end (loads a GGUF and streams a short generation):

```sh
flutter test integration_test/model_smoke_test.dart -d macos \
  --dart-define=MODEL_PATH=/absolute/path/to/model.gguf
```

Any small GGUF works, e.g.
[stories260K.gguf](https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf)
(~1 MB). The macOS app sandbox is disabled in this example's entitlements so
the test can read the model from an arbitrary path.
