use tonic::{transport::Server, Request, Response, Status};

use hello_world::greeter_server::{Greeter, GreeterServer};
use hello_world::{AddTensorsRequest, AddTensorsResponse, HelloReply, HelloRequest, Tensor};

pub mod hello_world {
    tonic::include_proto!("helloworld");
}

#[derive(Debug, Default)]
pub struct MyGreeter {}

#[tonic::async_trait]
impl Greeter for MyGreeter {
    async fn say_hello(
        &self,
        request: Request<HelloRequest>,
    ) -> Result<Response<HelloReply>, Status> {
        println!("Got a request: {:?}", request);

        let reply = HelloReply {
            message: format!("Hello {}!", request.into_inner().name),
        };

        Ok(Response::new(reply))
    }

    async fn add_tensors(
        &self,
        request: Request<AddTensorsRequest>,) -> Result<Response<AddTensorsResponse>, Status> {
        let AddTensorsRequest { a, b } = request.into_inner();

        let a = a.ok_or_else(|| Status::invalid_argument("missing tensor a"))?;
        let b = b.ok_or_else(|| Status::invalid_argument("missing tensor b"))?;

        if a.shape != b.shape {
            return Err(Status::invalid_argument(
                "tensor shapes must match for addition",
            ));
        }

        if a.values.len() != b.values.len() {
            return Err(Status::invalid_argument(
                "tensor value lengths must match for addition",
            ));
        }

        let result_values = a
            .values
            .iter()
            .zip(&b.values)
            .map(|(x, y)| x + y)
            .collect();

        let response = AddTensorsResponse {
            result: Some(Tensor {
                values: result_values,
                shape: a.shape,
            }),
        };

        Ok(Response::new(response))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let greeter = MyGreeter::default();

    Server::builder()
        .add_service(GreeterServer::new(greeter))
        .serve(addr)
        .await?;

    Ok(())
}
