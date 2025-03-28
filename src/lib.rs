pub mod onnx_structure;
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

use std::fs::File;
use std::io::Read;
use pyo3::prelude::*;

use crate::onnx_structure::{ModelProto, TensorProto};
use crate::read_onnx::generate_onnx_model;
use crate::model_inference::inference;
use protobuf::Message;

/// A Python module implemented in Rust.
#[pymodule]
fn Group17(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(onnx_make_inference, m)?)?;
  Ok(())
}

#[pyfunction]
fn onnx_make_inference (onnx_file: String, input_path: &str, output_path: &str, input_tensor_name: Vec<&str>) {
  /* LIBRARY PARSING */
  let onnx_bytes = std::fs::read(onnx_file.clone()).expect("Failed to read file");
  let mut model = ModelProto::parse_from_bytes(&*onnx_bytes).expect("Failed to convert the file");

  /* CUSTOM PARSING */
  //let mut model = generate_onnx_model(&onnx_file, "models/onnx.proto");
  //println!("{:?}", model);

  let input_data = read_input_data(input_path).unwrap();
  let output_data = read_input_data(output_path).unwrap();

  inference(model, input_data, input_tensor_name);

  println!("Expected Data: {:?}", output_data);
}

fn read_input_data(input_path: &str) -> Option<Vec<f32>>{
  let mut res: Option<Vec<f32>> = None;

  let mut file = File::open(input_path).expect("Cannot open input file");

  let mut buffer = Vec::new();
  file.read_to_end(&mut buffer).expect("Error while reading file");

  let parsed_message = TensorProto::parse_from_bytes(&buffer).expect("Error while deserializing the message");

  res = Some(parsed_message.raw_data.clone().unwrap().chunks_exact(4).map(|chunk| u8_to_f32(chunk)).collect());

  res
}

fn u8_to_f32(bytes: &[u8]) -> f32 {
  assert_eq!(bytes.len(), 4);
  let mut array: [u8; 4] = Default::default();
  array.copy_from_slice(&bytes);
  f32::from_le_bytes(array)
}