cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug -GNinja -Bbuild
cmake --build build

glslc -fshader-stage=compute shader.glsl -o shader.spv
copy .\shader.spv .\build\shader.spv