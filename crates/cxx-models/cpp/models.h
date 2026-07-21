#pragma once
#include "rust/cxx.h"

#include <cstdint>
#include <memory>

namespace nereid {

// The interface every compiled-in C++ model implements. Bridged to Rust as the
// opaque `Model` type. Output is written to the out-parameters (cxx-owned
// containers), which keeps this header free of any cxx-generated types.
class Model {
public:
    virtual ~Model() = default;

    virtual void run(rust::Str dtype,
                     rust::Slice<const int64_t> shape,
                     rust::Slice<const uint8_t> data,
                     rust::Vec<int64_t>& out_shape,
                     rust::String& out_dtype,
                     rust::Vec<uint8_t>& out_data) const = 0;
};

// Model registry — add a model by registering its name here.
bool model_exists(rust::Str name);
std::unique_ptr<Model> create_model(rust::Str name);

}  // namespace nereid
