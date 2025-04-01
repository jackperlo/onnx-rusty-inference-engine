use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use ndarray::{Array, Axis};
use ndarray::prelude::*;
use num_traits::Float;
use onnx_protobuf::NodeProto;

/// This function manages the softmax operation
/// # Arguments
/// * output_container: contains the partial results computed by inference operations
/// * node: node on which softmax has to be performed
pub fn softmax(output_container: &Arc<Mutex<HashMap<String,
  (Option<Array2<f32>>, Option<Array4<f32>>)>>>,
              node: &NodeProto) {
  let map = output_container.lock().unwrap();
  let input = Array4::from(
    map.get(node.input[0].as_str()).unwrap().1.clone().unwrap());

  drop(map);

  let result = softmax_wrapper(input, None);
  #[cfg(feature = "debug_prints")]{
    dbg!(result.clone());
  }
  #[cfg(feature = "operations_prints")]{
    println!("Softmax, done! by {}", thread::current().name().unwrap_or("MAIN THREAD"));
  }
  // TODO: manage output layer to get TOP1 predicted class
  let mut i = 0;
  let mut best_class_index = 0;
  let mut best_class_percentage = f32::min_value();
  while i < result.len_of(Axis(1)){
    if result[[0, i]] > best_class_percentage {
      best_class_percentage = result[[0, i]];
      best_class_index = i+1;
    }
    i += 1;
  }
  // TODO: get model name to print it
  println!("\nSqueezenet1.0-8 Inference results: Class {}-nth predicted.\nActual Data: {:?}", best_class_index, result.clone());
}

//OPSET VERSION = 8
fn softmax_wrapper(input: Array4<f32>, axis: Option<usize>) -> Array2<f32> {
  let batch_size = input.len_of(Axis(0));
  let other_size = input.len_of(Axis(1))*input.len_of(Axis(2))*input.len_of(Axis(3));
  let mut x: Array2<f32> = input.into_shape((batch_size, other_size)).unwrap();
  //let mut x_64 = x.map(|x| *x as f64);

  let axis = axis.unwrap_or(1);
  let max_val = x.fold_axis(Axis(axis), f32::NEG_INFINITY, |&max, &el| el.max(max));
  x -= &max_val.insert_axis(Axis(axis));
  let exp_x = x.mapv(f32::exp);
  let sum_exp_x = exp_x.sum_axis(Axis(axis));
  exp_x / &sum_exp_x.insert_axis(Axis(axis))
}

#[allow(dead_code)]
fn test_softmax() {
  // Esempio di utilizzo
  let x = Array::from_shape_vec((1, 1, 2,4), vec![118.85734,5640.1426,2.,3.,1000.,1001.,1002.,1003.]).unwrap();
  let _x_2 = Array::from_shape_vec((1, 1, 1,3), vec![-1.,0.,1.]).unwrap();
  println!("input: \n{:?}", x);
  let result = softmax_wrapper(x, None);
  println!("output: \n{:?}", result);
}
