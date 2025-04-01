use std::io::{Read};
use std::fs::{File};
use protobuf::{Message};
use onnx_protobuf::{ModelProto,TensorProto};

use onnx_rusty_inference_engine::inference_engine::model_inference::inference;
use onnx_rusty_inference_engine::inference_engine::utils::u8_to_f32;

fn main() {
  //MNIST-8
  /*let mut onnx_file = String::from("models/mnist-8.onnx");
  let input_path = "mnist_data_0.pb"; //image gathered from the mnist repo
  let output_path = "mnist_output_0.pb"; //output gathered from the mnist repo
  let input_tensor_name = vec!["Input3", "Parameter193"];*/

  //SQUEEZENET1.0-8
  let onnx_file = String::from("models/squeezenet1.0-8.onnx");
  let input_path = "squeezenet_data_0.pb"; //image gathered from the squeezenet repo
  let output_path = "squeezenet_output_0.pb"; //output gathered from the squeezenet repo
  let input_tensor_name = vec!["data_0"];

  read_and_make_inference(onnx_file, input_path, output_path, input_tensor_name);
  //read_and_write(onnx_file, input_path, output_path, input_tensor_name);
  //read_modify_write(onnx_file, input_path, output_path, input_tensor_name);
}

fn read_and_make_inference(onnx_file: String, input_path: &str, output_path: &str, input_tensor_name: Vec<&str>) {
  /*Library parsing call*/
  let onnx_bytes = std::fs::read(onnx_file).expect("Failed to read file");
  let model = ModelProto::parse_from_bytes(&*onnx_bytes).expect("Failed to convert the file");

  /*Custom parsing call*/
  //let mut model = generate_onnx_model(&onnx_file, "models/onnx.proto");
  //println!("{:?}", model);

  let input_data = read_input_data(input_path).unwrap();
  let output_data = read_input_data(output_path).unwrap();

  inference(model, input_data, input_tensor_name);

  println!("Expected Data: {:?}", output_data);
}

fn read_input_data(input_path: &str) -> Option<Vec<f32>> {
  let mut file = File::open(input_path).expect("Cannot open input file");

  let mut buffer = Vec::new();
  file.read_to_end(&mut buffer).expect("Error while reading file");

  let parsed_message = TensorProto::parse_from_bytes(&buffer).expect("Error while deserializing the message");

  Some(parsed_message.raw_data.clone().chunks_exact(4).map(|chunk| u8_to_f32(chunk)).collect())
}


