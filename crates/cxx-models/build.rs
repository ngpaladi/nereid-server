fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("cpp/models.cpp")
        .std("c++17")
        .compile("cxx-models");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cpp/models.cpp");
    println!("cargo:rerun-if-changed=cpp/models.h");
}
