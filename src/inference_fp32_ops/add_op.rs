use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use ndarray::{Array2, Array3, Array4, Axis};
use onnx_protobuf::{NodeProto, TensorProto, ValueInfoProto};
use num_traits::Float;

use crate::inference_engine::utils::{already_into_initializer, get_stored_tensor};

/// This function manages the add operation
/// # Arguments
/// * output_container: contains the partial results computed by inference operations
/// * node: node on which add has to be performed
/// * model_inputs: inputs of the onnx model
/// * model_initializers: initializers of the onnx model
pub fn add(output_container: &Arc<Mutex<HashMap<String,
  (Option<Array2<f32>>, Option<Array4<f32>>)>>>,
          node: &NodeProto,
          model_inputs: &Vec<ValueInfoProto>,
          model_initializers: &Vec<TensorProto>) {
  let mut input_1_arr_4: Array4<f32> = Default::default();
  let mut input_1_arr_2: Array2<f32> = Default::default();
  let mut input_2_arr_3: Array3<f32> = Default::default();
  let mut input_2_arr_2: Array2<f32> = Default::default();

  let map = output_container.lock().unwrap();

  if already_into_initializer(model_initializers, node.input[0].as_str()) {
    let (arr4, _, arr2, _, _) =
      get_stored_tensor(0, node, model_inputs, model_initializers);
    match arr4 {
      Some(arr4) => input_1_arr_4 = arr4,
      None => {
        match arr2 {
          Some(arr2) => input_1_arr_2 = arr2,
          None => panic!("Cannot retrieve input 1 for Add operation from initializers")
        }
      }
    };
  } else {
    match map.get(node.input[0].as_str()).unwrap().0.clone() {
      Some(arr2) => input_1_arr_2 = arr2,
      None => {
        match map.get(node.input[0].as_str()).unwrap().1.clone() {
          Some(arr4) => input_1_arr_4 = arr4,
          None => panic!("Cannot retrieve input 1 for Add operation from hashmap input/output")
        }
      }
    }
  }

  drop(map);

  if already_into_initializer(model_initializers, node.input[1].as_str()) {
    let (_, arr3, arr2, _, _) =
      get_stored_tensor(1, node, model_inputs, model_initializers);
    match arr3 {
      Some(arr3) => input_2_arr_3 = arr3,
      None => {
        match arr2 {
          Some(arr2) => input_2_arr_2 = arr2,
          None => panic!("Cannot retrieve input 2 for Add operation from initializes")
        }
      }
    }
  } else {
    panic!("Cannot retrieve input 2 for Add operation");
  }

  let mut map_mut = output_container.lock().unwrap();

  let mut output_layer_2: Array2<f32> = Default::default();
  let mut output_layer_4: Array4<f32> = Default::default();
  if input_1_arr_4.len() > 0{
    output_layer_4 = input_1_arr_4+input_2_arr_3;
    #[cfg(feature = "debug_prints")]{
      dbg!(output_layer_4.clone());
    }
    #[cfg(feature = "operations_prints")]{
      println!("Add, done! by {}", thread::current().name().unwrap_or("MAIN THREAD"));
    }
    map_mut.insert(node.output[0].clone(), (None, Some(output_layer_4)));
  }else{
    output_layer_2 = input_1_arr_2+input_2_arr_2;
    #[cfg(feature = "debug_prints")]{
      dbg!("Add: {:?}", output_layer_2.clone());
    }
    #[cfg(feature = "operations_prints")]{
      println!("Add, done! by {}", thread::current().name().unwrap_or("MAIN THREAD"));
    }
    map_mut.insert(node.output[0].clone(), (Some(output_layer_2.clone()), None));

    // TODO: manage the add operation properly so that it does not compute here the TOP1 predicted class, but does it in the output layer
    let mut i = 0;
    let mut best_class_index = 0;
    let mut best_class_percentage = f32::min_value();
    while i < output_layer_2.clone().len_of(Axis(1)){
      if output_layer_2[[0, i]] > best_class_percentage {
        best_class_percentage = output_layer_2[[0, i]];
        best_class_index = i+1;
      }
      i += 1;
    }
    println!("\nMNist-8 Inference results: Class {}-nth predicted.\nActual Data: {:?}",
             best_class_index, output_layer_2);
  }
}