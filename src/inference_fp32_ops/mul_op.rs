use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use ndarray::{Array2, Array4};
use onnx_protobuf::NodeProto;

/// This function manages the mul operation
/// # Arguments
/// * output_container: contains the partial results computed by inference operations
/// * node: node on which mul has to be performed
pub fn mul(output_container: &Arc<Mutex<HashMap<String,
  (Option<Array2<f32>>, Option<Array4<f32>>)>>>,
          node: &NodeProto) {
  let map = output_container.lock().unwrap();

  let input_1 = Array2::from(
    map.get(node.input[0].as_str()).unwrap().0.clone().unwrap());
  let input_2 = Array2::from(
    map.get(node.input[1].as_str()).unwrap().0.clone().unwrap());

  drop(map);

  let output_layer: Array2<f32> = input_1.dot(&input_2);
  #[cfg(feature = "debug_prints")]{
    dbg!("MatMul: {:?}", output_layer.clone());
  }
  #[cfg(feature = "operations_prints")]{
    println!("Mul, done! by {}", thread::current().name().unwrap_or("MAIN THREAD"));
  }
  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (Some(output_layer), None));
}