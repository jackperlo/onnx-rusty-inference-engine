use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use ndarray::{Array2, Array4, Axis, concatenate};
use onnx_protobuf::NodeProto;

/// This function manages the concatenation operation
/// # Arguments
/// * output_container: contains the partial results computed by inference operations
/// * node: node on which the concatenation has to be performed
pub fn concatenation(output_container: &Arc<Mutex<HashMap<String,
  (Option<Array2<f32>>, Option<Array4<f32>>)>>>,
                  node: &NodeProto) {
  let map = output_container.lock().unwrap();
  let input_1 = Array4::from(
    map.get(node.input[0].as_str()).unwrap().1.clone().unwrap());
  let input_2 = Array4::from(
    map.get(node.input[1].as_str()).unwrap().1.clone().unwrap().clone());

  drop(map);

  let mut axis = 1;

  for attr in &node.attribute {
    axis = match attr.name.as_ref() {
      "axis" => attr.i,
      _ => panic!("ATTRIBUTE NAME FOR CONCATENATE NOT FOUND, {}",
                  <String as AsRef<str>>::as_ref(&attr.name))
    };
  }
  let output_layer: Array4<f32> = concatenate(Axis(axis as usize),
                                              &[input_1.view(), input_2.view()]).unwrap();
  #[cfg(feature = "debug_prints")]{
    dbg!("Concatenate: {:?}", output_layer);
  }
  #[cfg(feature = "operations_prints")]{
    println!("Concatenation, done! by {}", thread::current().name().unwrap_or("MAIN THREAD"));
  }
  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (None, Some(output_layer)));
}