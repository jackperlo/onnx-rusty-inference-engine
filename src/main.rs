pub mod onnx_structure;

use std::io::{Read};
use std::fs::{File};
use protobuf::{Message};

mod read_proto;
mod read_onnx;
mod write_onnx;
mod convolution_op;
mod relu_op;
mod max_pool_op;
mod dropout_op;
mod global_average_pool_op;
mod softmax;
mod model_inference;
mod reshape_op;

use crate::read_onnx::generate_onnx_model;
use crate::model_inference::inference;
use crate::onnx_structure::{ModelProto, NodeProto, TensorProto};
use crate::write_onnx::generate_onnx_file;

fn main() {
  //MNIST-8
  /*let mut onnx_file = String::from("models/mnist-8.onnx");
  let input_path = "mnist_data_0.pb"; //image gathered from the mnist repo
  let output_path = "mnist_output_0.pb"; //output gathered from the mnist repo
  let input_tensor_name = vec!["Input3", "Parameter193"];*/

  //SQUEEZENET1.0-8
  let mut onnx_file = String::from("models/squeezenet1.0-8.onnx");
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
  let mut model = ModelProto::parse_from_bytes(&*onnx_bytes).expect("Failed to convert the file");

  /*Custom parsing call*/
  //let mut model = generate_onnx_model(&onnx_file, "models/onnx.proto");
  //println!("{:?}", model);

  let input_data = read_input_data(input_path).unwrap();
  let output_data = read_input_data(output_path).unwrap();

  inference(model, input_data, input_tensor_name);

  println!("Expected Data: {:?}", output_data);
}

fn read_and_write(mut onnx_file: String, input_path: &str, output_path: &str, input_tensor_name: Vec<&str>) {
  /*Library parsing call*/
  let onnx_bytes = std::fs::read(onnx_file.clone()).expect("Failed to read file");
  let mut model = ModelProto::parse_from_bytes(&*onnx_bytes).expect("Failed to convert the file");

  /*Custom parsing call*/
  //let mut model = generate_onnx_model(&onnx_file, "models/onnx.proto");
  //println!("{:?}", model);

  let onnx_generated_file: Vec<&str> = onnx_file.split(".onnx").collect();
  onnx_file = String::from(onnx_generated_file[0]);
  onnx_file.push_str("_generated.onnx");
  generate_onnx_file(&onnx_file, &mut model);
}

fn read_modify_write(mut onnx_file: String, input_path: &str, output_path: &str, input_tensor_name: Vec<&str>) {
  /*Library parsing call*/
  let onnx_bytes = std::fs::read(onnx_file.clone()).expect("Failed to read file");
  let mut model = ModelProto::parse_from_bytes(&*onnx_bytes).expect("Failed to convert the file");

  /*Custom parsing call*/
  //let mut model = generate_onnx_model(&onnx_file, "models/onnx.proto");
  //println!("{:?}", model);

  /* Model edit: producer name*/
  model.set_producer_name("Jack&Fabri".to_string());

  /* Model edit: adding a new node */
  let mut new_node_proto = model.graph.node[0].clone();
  new_node_proto.set_name("Node Jack&Fabri".to_string());
  model.graph.as_mut().expect("Graph Not Found").node.insert(0, new_node_proto);

  /* Model edit: model name*/
  model.graph.as_mut().expect("Graph Not Found").set_name("Jack&Fabri_Graph".to_string());

  let onnx_generated_file: Vec<&str> = onnx_file.split(".onnx").collect();
  onnx_file = String::from(onnx_generated_file[0]);
  onnx_file.push_str("_generated.onnx");
  generate_onnx_file(&onnx_file, &mut model);
}

fn read_input_data(input_path: &str) -> Option<Vec<f32>> {
  let mut file = File::open(input_path).expect("Cannot open input file");

  let mut buffer = Vec::new();
  file.read_to_end(&mut buffer).expect("Error while reading file");

  let parsed_message = TensorProto::parse_from_bytes(&buffer).expect("Error while deserializing the message");

  Some(parsed_message.raw_data.clone().unwrap().chunks_exact(4).map(|chunk| u8_to_f32(chunk)).collect())
}

fn u8_to_f32(bytes: &[u8]) -> f32 {
  assert_eq!(bytes.len(), 4);
  let mut array: [u8; 4] = Default::default();
  array.copy_from_slice(&bytes);
  f32::from_le_bytes(array)
}
