use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use ndarray::{Array, Array2, Array4};
use onnx_protobuf::NodeProto;

/// This function manages the relu operation
/// # Arguments
/// * output_container: contains the partial results computed by inference operations
/// * node: node on which relu has to be performed
pub fn relu(output_container: &Arc<Mutex<HashMap<String,
  (Option<Array2<f32>>, Option<Array4<f32>>)>>>,
           node: &NodeProto) {
  let map = output_container.lock().unwrap();
  let input = Array4::from(map.get(node.input[0].as_str())
    .unwrap().1.clone().unwrap());

  drop(map);

  let output_layer: Array4<f32> = relu_wrapper(&input);
  #[cfg(feature = "debug_prints")]{
    dbg!("Relu: {:?}", output_layer.clone());
  }
  #[cfg(feature = "operations_prints")]{
    println!("Relu, done! by {}", thread::current().name().unwrap_or("MAIN THREAD"));
  }
  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (None, Some(output_layer)));
}

fn relu_wrapper(x: &Array4<f32>) -> Array4<f32> {
  x.map(|&val| val.max(0.0))
}

#[allow(dead_code)]
fn test_relu(){
  let input = Array::from_shape_vec(
    (1, 1, 7, 5),
    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, -17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0]
  )
    .unwrap();

  let output = Array::from_shape_vec(
    (1, 1, 7, 5),
    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 0.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0]
  )
    .unwrap();

  println!("{:?}", relu_wrapper(&input));
  println!("{:?}", output);
}