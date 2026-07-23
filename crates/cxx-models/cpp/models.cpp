#include "cxx-models/cpp/models.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nereid {
namespace {

// Example compiled-in model: output = input + 1 (float32).
class AddOne : public Model {
public:
    void run(rust::Str dtype,
             rust::Slice<const int64_t> shape,
             rust::Slice<const uint8_t> data,
             rust::Vec<int64_t>& out_shape,
             rust::String& out_dtype,
             rust::Vec<uint8_t>& out_data) const override {
        if (std::string(dtype.data(), dtype.size()) != "float32") {
            throw std::runtime_error("cxx AddOne model only supports float32");
        }

        const size_t n = data.size() / sizeof(float);
        std::vector<float> vals(n);
        std::memcpy(vals.data(), data.data(), n * sizeof(float));
        for (float& v : vals) {
            v += 1.0f;
        }

        for (int64_t d : shape) {
            out_shape.push_back(d);
        }
        out_dtype = rust::String("float32");
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(vals.data());
        out_data.reserve(n * sizeof(float));
        for (size_t i = 0; i < n * sizeof(float); ++i) {
            out_data.push_back(bytes[i]);
        }
    }
};

}  // namespace

bool model_exists(rust::Str name) {
    return std::string(name.data(), name.size()) == "cxxadd";
}

std::unique_ptr<Model> create_model(rust::Str name) {
    if (std::string(name.data(), name.size()) == "cxxadd") {
        return std::make_unique<AddOne>();
    }
    return nullptr;
}

}  // namespace nereid
