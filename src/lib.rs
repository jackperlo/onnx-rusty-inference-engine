pub mod inference_engine;
pub mod inference_fp32_ops;

/*
use std::fs::File;
use std::io::Read;
use pyo3::prelude::*;
use onnx_protobuf::{ModelProto, TensorProto};
use protobuf::Message;

use crate::inference_engine::model_inference::inference;

/// A Python module implemented in Rust.
#[pymodule]
fn group17(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(onnx_make_inference, m)?)?;
  Ok(())
}

#[pyfunction]
fn onnx_make_inference (onnx_file: String, input_path: &str, output_path: &str, input_tensor_name: Vec<&str>) {
  let onnx_bytes = std::fs::read(onnx_file.clone()).expect("Failed to read file");
  let model = ModelProto::parse_from_bytes(&*onnx_bytes).expect("Failed to convert the file");

  let input_data = read_input_data(input_path).unwrap();
  let output_data = read_input_data(output_path).unwrap();

  inference(model, input_data, input_tensor_name);

  println!("Expected Data: {:?}", output_data);
}

fn read_input_data(input_path: &str) -> Option<Vec<f32>>{
  let mut _res: Option<Vec<f32>> = None;

  let mut file = File::open(input_path).expect("Cannot open input file");

  let mut buffer = Vec::new();
  file.read_to_end(&mut buffer).expect("Error while reading file");

  let parsed_message = TensorProto::parse_from_bytes(&buffer).expect("Error while deserializing the message");

  _res = Some(parsed_message.raw_data.clone().chunks_exact(4).map(|chunk| u8_to_f32(chunk)).collect());

  _res
}

fn u8_to_f32(bytes: &[u8]) -> f32 {
  assert_eq!(bytes.len(), 4);
  let mut array: [u8; 4] = Default::default();
  array.copy_from_slice(&bytes);
  f32::from_le_bytes(array)
}
*/