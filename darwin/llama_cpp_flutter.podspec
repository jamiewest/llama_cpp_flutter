#
# Shared iOS + macOS podspec (sharedDarwinSource).
# The vendored llama.xcframework comes from ggml-org/llama.cpp GitHub releases
# via scripts/fetch_llama_xcframework.sh (pin: tool/versions.env) and is NOT
# checked into source control. It is fetched below at `pod install` time so a
# fresh checkout builds without manual steps; the fetch is a no-op once the
# installed framework matches the pin. Set LLAMA_CPP_FLUTTER_SKIP_FETCH=1 to opt
# out (offline/lint environments) and run the script yourself.
#
unless ENV['LLAMA_CPP_FLUTTER_SKIP_FETCH'] == '1'
  fetch_script = File.expand_path('../scripts/fetch_llama_xcframework.sh', __dir__)
  unless system('bash', fetch_script)
    raise 'llama.xcframework fetch failed — run ' \
          'scripts/fetch_llama_xcframework.sh manually ' \
          '(or set LLAMA_CPP_FLUTTER_SKIP_FETCH=1 to skip).'
  end
end

Pod::Spec.new do |s|
  s.name             = 'llama_cpp_flutter'
  s.version          = '0.1.0'
  s.summary          = 'On-device llama.cpp inference for iOS and macOS.'
  s.description      = <<-DESC
  Flutter plugin wrapping llama.cpp via a vendored xcframework, exposed through
  a Pigeon bridge with a streaming token EventChannel.
                       DESC
  s.homepage         = 'https://github.com/jamiewest/llama_cpp_flutter'
  s.license          = { :type => 'MIT', :text => 'See LICENSE in repo root.' }
  s.author           = { 'Jamie West' => 'jamiewst@gmail.com' }
  s.source           = { :path => '.' }

  # Build the plugin as a static framework so the dynamic llama.framework it
  # vendors is linked + embedded by the final app target (which carries
  # `-framework llama`), rather than by this pod's own dynamic-framework link.
  s.static_framework    = true

  s.source_files        = 'Classes/**/*'
  s.vendored_frameworks = 'Frameworks/llama.xcframework'
  s.preserve_paths      = 'Frameworks/llama.xcframework'
  s.frameworks          = 'Metal', 'MetalKit', 'Accelerate', 'Foundation'

  s.ios.dependency 'Flutter'
  s.osx.dependency 'FlutterMacOS'
  s.ios.deployment_target = '16.4'
  s.osx.deployment_target = '13.3'

  s.swift_version = '5.0'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
  }
end
