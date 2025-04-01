use std::collections::HashMap;
use std::{io, thread};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use ndarray::{Array2, Array4};
use onnx_protobuf::{ModelProto, NodeProto};

use crate::inference_engine::utils::manage_input_data;
use crate::inference_engine::multithreading::utils::{possibile_wating_for_previous_results,
                                                     search_node_without_previous_dependencies};
use crate::inference_engine::multithreading::multithreading::check_parallel_nodes_and_start_threads;

use crate::inference_fp32_ops::max_pool_op::max_pool;
use crate::inference_fp32_ops::add_op::add;
use crate::inference_fp32_ops::concatenate_op::concatenation;
use crate::inference_fp32_ops::convolution_op::convolution;
use crate::inference_fp32_ops::dropout_op::drop_out;
use crate::inference_fp32_ops::global_average_pool_op::global_average_pool;
use crate::inference_fp32_ops::mul_op::mul;
use crate::inference_fp32_ops::relu_op::relu;
use crate::inference_fp32_ops::reshape_op::reshape;
use crate::inference_fp32_ops::softmax_op::softmax;

/// This function make inference on the pre-trained model received as input
/// # Arguments
/// * model: Protobuf ModelProto struct which contains the onnx model
/// * input_data: this is the input vector of the model
/// * input_tensor_name: name(s) of the model's input(s)
pub fn inference(model: ModelProto, input_data: Vec<f32>, input_tensor_name: Vec<&str>) {
  let hashmap_outputs_to_inputs: Arc<Mutex<HashMap<String,
    (Option<Array2<f32>>, Option<Array4<f32>>)>>> =
    Arc::new(Mutex::new(HashMap::new()));
  let arc_model= Arc::new(model);

  // Used by main thread to wait for the considered node computed data
  let condition_var: Arc<(Mutex<Vec<String>>, Condvar)> =
    Arc::new((Mutex::new(Vec::new()), Condvar::new()));

  let mut position = 0;
  // Positions of nodes which have already been executed by other threads rather than the main one
  let mut position_to_skip: Vec<i32> = Vec::new();

  let mut found_independent_nodes = false;

  manage_input_data(&hashmap_outputs_to_inputs, &arc_model, input_data, input_tensor_name);

  let map = hashmap_outputs_to_inputs.lock().unwrap();
  let result = search_node_without_previous_dependencies(&arc_model,
                                                         map.keys().collect());
  // Unlock the lock on the main results' container (i.e. the hashmap)
  drop(map);

  let mut threads: Vec<io::Result<JoinHandle<()>>> = Vec::new();

  if result.is_some() {
    // Launch independent threads
    found_independent_nodes = true;

    position_to_skip = result.clone().unwrap().1;
    let independent_nodes = result.unwrap().0;

    let mut n_t = 0;
    for node in independent_nodes {
      let t_map = hashmap_outputs_to_inputs.clone();
      let t_model = arc_model.clone();
      let t_condvar = condition_var.clone();

      threads.push(thread::Builder::new()
        .name(format!("{}{}", "Thread", n_t))
        .spawn(move || {
          node_inference(&node, &t_map, &t_model);

          // Notify to main thread that some nodes(/threads) are being executed
          let (l, cvar) = &*t_condvar;
          let mut new_value_added = l.lock().unwrap();
          new_value_added.extend(node.output.clone());
          cvar.notify_all();
        }));

      n_t += 1;
    }
  }

  for node in &arc_model.graph.node {
#[cfg(feature = "debug_prints")]{
  print!("TRY INFERENCE ON {:?} ON {} OPERATION by MAIN THREAD", node.input);
}
    if found_independent_nodes {
      check_parallel_nodes_and_start_threads(&arc_model,
                                             position,
                                             node,
                                             &mut position_to_skip,
                                             &hashmap_outputs_to_inputs,
                                             &condition_var,
                                             &mut threads);
      found_independent_nodes = false;
    }

    if !position_to_skip.contains(&position) {
      possibile_wating_for_previous_results(node,
                                            &hashmap_outputs_to_inputs,
                                            &condition_var,
                                            &arc_model);

      check_parallel_nodes_and_start_threads(&arc_model,
                                             position,
                                             node,
                                             &mut position_to_skip,
                                             &hashmap_outputs_to_inputs,
                                             &condition_var,
                                             &mut threads);
    }

    position += 1;
  }

  for t in threads {
    t.expect("PROBLEM JOINING").join().expect("ERROR");
  }
}


/// This function executes the inference operation of the given node
/// # Arguments
/// * node: node to perform inference for
/// * hashmap_outputs_to_inputs: contains the partial results calculated by inferences operations
/// * model: smart pointer which contains the onnx model
pub fn node_inference(node: &NodeProto,
                      hashmap_outputs_to_inputs: &Arc<Mutex<HashMap<String,
                        (Option<Array2<f32>>, Option<Array4<f32>>)>>>,
                      model: &Arc<ModelProto>) {
  println!("INFERENCE ON INPUT(s) {:?} ON {} OPERATION done by {}",
           node.input,
           node.op_type.clone(),
           thread::current().name().unwrap_or("MAIN PROCESS"));

  let operation = &node.op_type;
  match operation.as_str() {
    "Conv" => convolution(hashmap_outputs_to_inputs,
                             node,
                             &model.graph.input,
                             &model.graph.initializer),
    "Relu" => relu(hashmap_outputs_to_inputs, node),
    "MaxPool" => max_pool(hashmap_outputs_to_inputs, node,
                             &model.graph.input,
                             &model.graph.initializer),
    "Concat" => concatenation(hashmap_outputs_to_inputs, node),
    "Dropout" => drop_out(hashmap_outputs_to_inputs, node),
    "GlobalAveragePool" => global_average_pool(hashmap_outputs_to_inputs, node),
    "Softmax" => softmax(hashmap_outputs_to_inputs, node),
    "Reshape" => reshape(hashmap_outputs_to_inputs, node,
                            &model.graph.input,
                            &model.graph.initializer),
    "Add" => add(hashmap_outputs_to_inputs, node,
                    &model.graph.input,
                    &model.graph.initializer),
    "MatMul" => mul(hashmap_outputs_to_inputs, node),
    _ => { panic!("INFERENCE OPERATION '{}' NOT FOUND FOR NODE {}",
                  operation.as_str(),
                  &<String as AsRef<str>>::as_ref(&node.name)) }
  }
}