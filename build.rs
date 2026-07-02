fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    let mut prost_config = prost_build::Config::new();
    prost_config.protoc_executable(protoc);

    // Both protos declare `package inference;`, so prost merges them into a
    // single generated module included via `tonic::include_proto!("inference")`.
    // grpc_service.proto is the Triton-compatible KServe v2 surface.
    tonic_prost_build::configure().compile_with_config(
        prost_config,
        &["proto/inference.proto", "proto/grpc_service.proto"],
        &["proto"],
    )?;

    Ok(())
}
