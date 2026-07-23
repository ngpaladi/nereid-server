// Example nereid C++ subprocess model: output = input + 1.
//
// The server compiles this to a `model` executable on startup and runs it per
// request. It speaks nereid's language-agnostic tensor contract:
//   - input:  raw little-endian float32 bytes on stdin; the (batch-normalized)
//             shape is in $NEREID_INPUT_SHAPE (e.g. "1,4").
//   - output: a framed tensor written to $NEREID_OUTPUT_PATH — a header line
//             "float32 d0,d1,...\n" followed by the raw little-endian bytes.
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main() {
    const char* out_path = std::getenv("NEREID_OUTPUT_PATH");
    if (!out_path) {
        std::cerr << "NEREID_OUTPUT_PATH not set\n";
        return 1;
    }
    const char* shape_env = std::getenv("NEREID_INPUT_SHAPE");
    const std::string shape = shape_env ? shape_env : "";

    // Read the whole input tensor (raw little-endian float32) from stdin.
    std::vector<char> raw((std::istreambuf_iterator<char>(std::cin)),
                          std::istreambuf_iterator<char>());
    const size_t n = raw.size() / sizeof(float);
    std::vector<float> vals(n);
    std::memcpy(vals.data(), raw.data(), n * sizeof(float));

    // The "inference": add one.
    for (float& v : vals) {
        v += 1.0f;
    }

    // Write the framed output tensor. The output shape equals the input shape.
    std::ofstream out(out_path, std::ios::binary);
    out << "float32 " << shape << "\n";
    out.write(reinterpret_cast<const char*>(vals.data()),
              static_cast<std::streamsize>(n * sizeof(float)));
    return 0;
}
