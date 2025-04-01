use onnx_protobuf::{ModelProto, NodeProto};
use std::sync::{Arc, Condvar, Mutex};
use std::collections::HashMap;
use ndarray::{Array2, Array4};

use crate::inference_engine::utils::already_into_initializer;
use crate::inference_engine::model_inference::node_inference;

/// This function pauses a given thread if its needed inputs are not already available
/// # Arguments
/// * node: a considered node in the model to check for input availability
/// * hashmap_outputs_to_inputs: contains the partial results calculated by inference operations
/// * condition_var: the variable used for waiting in case the inputs are not present
/// * arc_model: smart pointer which contains the onnx struct
pub fn possibile_wating_for_previous_results(node: &NodeProto,
                                         hashmap_outputs_to_inputs: &Arc<Mutex<HashMap<String,
                                           (Option<Array2<f32>>, Option<Array4<f32>>)>>>,
                                         condition_var: &Arc<(Mutex<Vec<String>>, Condvar)>,
                                         arc_model: &Arc<ModelProto>) {
  let mut inputs_are_present = false;

  while !inputs_are_present {
    inputs_are_present = true;

    for input in &node.input {
      let map = hashmap_outputs_to_inputs.lock().unwrap();
      if !map.contains_key(input) {
        if !already_into_initializer(&arc_model.graph.initializer,
                                     input.as_str()) {
          inputs_are_present = false;
          break;
        }
      }
    }

    if inputs_are_present {
      node_inference(&node, &hashmap_outputs_to_inputs, &arc_model);
    } else {
      println!("MAIN THREAD WAITING FOR CHILDREN THREADS' RESULTS...");
      let (l, cvar) = &**condition_var;
      let mut new_values_added = l.lock().unwrap();

      // avoid to fall asleep when results are already available (and spurious notification as well)
      while new_values_added.len() == 0 {
        new_values_added = cvar.wait(new_values_added).unwrap();
      }
      #[cfg(feature = "debug_prints")]{
        println!("Values obtained {:?}", new_values_added);
      }
      *new_values_added = Vec::new();
    }
  }
}

/// This function looks for nodes which could be executed in parallel (share the same input(s))
/// # Arguments
/// * nodes: slice of nodes to check
/// * input_to_check: name of node's output to check if it's shared among different nodes
/// # Returns
/// It returns the independents nodes and their position in the onnx model
pub fn search_node_who_shares_input(nodes: &[NodeProto], input_to_check: &String)
                                -> Option<(Vec<Vec<NodeProto>>, Vec<i32>)> {
  let mut node_shares_input = 0;
  let mut position = 0;
  let mut hash_shares: HashMap<i32, NodeProto> = HashMap::new();

  for node in nodes {
    if node.input.contains(input_to_check) {
      node_shares_input += 1;
      hash_shares.insert(position, node.clone());
    }

    position += 1;
  }

  if node_shares_input >= 2 {
    let mut pos_to_skip: Vec<i32> = hash_shares.keys().cloned().collect();

    let mut vec_sequence: Vec<Vec<NodeProto>> = Vec::new();

    for el in &hash_shares {
      vec_sequence.push(Vec::new());
      vec_sequence.last_mut().unwrap().push(el.1.clone());
      let output_to_find = &el.1.output[0];

      let counter = el.0 + 1;

      for node in &nodes[*el.0 as usize + 1..nodes.len()] {
        if node.input.contains(output_to_find) {
          vec_sequence.last_mut().unwrap().push(node.clone());
          pos_to_skip.push(counter);
        } else {
          break;
          /* Sequence interrupted */
        }
      }
    }

    Some((vec_sequence, pos_to_skip))
  } else {
    None
  }
}

/// This function searches if in the model there are node without previous dependencies
/// i.e. they can be executed in a separated threads
/// # Arguments
/// * model: smart pointer which contains the onnx model
/// * previous_outputs: vector which contains the names of the operations
///   which have already been done
/// # Returns
/// It returns the independents nodes and their position in the onnx model
pub fn search_node_without_previous_dependencies(model: &Arc<ModelProto>,
                                             previous_outputs: Vec<&String>)
                                             -> Option<(Vec<NodeProto>, Vec<i32>)> {
  let mut position = 0;
  let mut pos_to_skip: Vec<i32> = Vec::new();
  let mut independent_nodes: Vec<NodeProto> = Vec::new();

  for node in &model.graph.node {
    let mut is_contained = true;
    for input in &node.input {
      if !previous_outputs.contains(&input) {
        if !already_into_initializer(&model.graph.initializer,
                                     input.as_str()) {
          is_contained = false;
          break;
        }
      }
    }

    if is_contained {
      pos_to_skip.push(position);
      independent_nodes.push(node.clone());
    }

    position += 1;
  }

  if pos_to_skip.len() < 2 {
    None
  } else {
    Some((independent_nodes, pos_to_skip))
  }
}