use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use ndarray::Array;
use ndarray::prelude::*;
use onnx_protobuf::{NodeProto, TensorProto, ValueInfoProto};
use rand::Rng;
use crate::inference_engine::utils::{already_into_initializer, get_stored_tensor};

/// This function manages the reshape operation
/// # Arguments
/// * output_container: contains the partial results computed by inference operations
/// * node: node on which reshape has to be performed
/// * model_inputs: inputs of the onnx model
/// * model_initializers: initializers of the onnx model
pub fn reshape(output_container: &Arc<Mutex<HashMap<String,
  (Option<Array2<f32>>, Option<Array4<f32>>)>>>,
              node: &NodeProto,
              model_inputs: &Vec<ValueInfoProto>,
              model_initializers: &Vec<TensorProto>) {
  let mut _data: Array4<f32> = Default::default();
  if already_into_initializer(model_initializers, node.input[0].as_str()) {
    let (arr4, _, _, _, _) = get_stored_tensor(0,
                                               node,
                                               model_inputs,
                                               model_initializers);
    _data = arr4.unwrap();
  } else {
    let map = output_container.lock().unwrap();
    _data = Array4::from(map.get(node.input[0].as_str()).unwrap().1.clone().unwrap());
    drop(map);
  }

  let mut _shape: Array1<i64> = Default::default();
  if already_into_initializer(model_initializers, node.input[1].as_str()) {
    let (_, _, _, _, arr1) = get_stored_tensor(1,
                                               node,
                                               model_inputs,
                                               model_initializers);
    _shape = arr1.unwrap();
  } else {
    panic!("Unable to retrieve Shape for Reshape operation");
  }

  let output_layer: Array2<f32> = reshape_wrapper(_data, _shape, None);

  #[cfg(feature = "debug_prints")]{
    dbg!("Reshape: {:?}", output_layer.clone());
  }
  #[cfg(feature = "operations_prints")]{
    println!("Reshape, done! by {}", thread::current().name().unwrap_or("MAIN THREAD"));
  }
  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (Some(output_layer), None));
}

//OPSET VERSION = 8
fn reshape_wrapper(data: Array4<f32>, shape: Array1<i64>, allowzero: Option<usize>) -> Array2<f32> {
  return if allowzero.is_none() {
    reshape_implementation(data, shape, 0)   /* Default value of allowzero = 0  */
  } else {
    reshape_implementation(data, shape, allowzero.unwrap())
  }
}

fn reshape_implementation(data: Array4<f32>, shape: Array1<i64>, allowzero: usize) -> Array2<f32> {
  let mut new_shape = shape.clone().to_vec();

  if allowzero == 0 {
    /* When any value in the ‘shape’ input is equal to zero the corresponding dimension value is copied
    from the input tensor dynamically. */
    let zeros_index: Vec<usize> = shape
      .iter()
      .enumerate()
      .filter(|&(_, &value)| value == 0)
      .map(|(index, _)| index)
      .collect();

    let data_shape = data.shape().to_vec();
    for &index in &zeros_index {
      new_shape[index] = data_shape[index] as i64;
    }
  }
  /* allowzero=1 indicates that if any value in the ‘shape’ input is set to zero,
  the zero value is honored */

  let (d1, d2) = (new_shape[0] as usize, new_shape[1] as usize);

  let aus = data.as_slice_memory_order().unwrap();
  let arr: Array2<f32> = Array2::from_shape_vec((d1, d2), aus.to_vec()).unwrap();
  arr
}

#[allow(dead_code)]
fn test_reshape() {
  // Esempio di utilizzo
  let mut rng = rand::thread_rng();

  let original_shape: Vec<i64> = vec![16, 3];
  let orig: Array1<i64> = original_shape.into();

  let data = Array::from_shape_vec((4, 2, 2, 3), (0..48).map(|_| rng.gen::<f32>()).collect()).unwrap();

  let reshaped = reshape_wrapper(data.into(), orig, Some(0));

  println!("reshaped: \n{:?}", reshaped);
}
