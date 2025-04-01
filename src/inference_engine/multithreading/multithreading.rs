use std::collections::HashMap;
use std::{io, thread};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use ndarray::{Array2, Array4};
use onnx_protobuf::{ModelProto, NodeProto};

use crate::inference_engine::model_inference::node_inference;
use crate::inference_engine::multithreading::utils::search_node_who_shares_input;

/// This function starts threads associated to nodes which can be executed in parallel
/// # Arguments
/// * arc_model: smart pointer which contains the onnx model
/// * position: position of the node in the onnx model
/// * node: considered node
/// * position_to_skip: positions of the nodes which have already been executed
/// * hashmap_outputs_to_inputs: contains the partial results calculated by inference operations
/// * condition_var: the variable used for notifying that result(s) is/are ready
/// * threads: vector that contains all the generated threads
pub fn check_parallel_nodes_and_start_threads(arc_model: &Arc<ModelProto>,
                                          position: i32,
                                          node: &NodeProto,
                                          position_to_skip: &mut Vec<i32>,
                                          hashmap_outputs_to_inputs: &Arc<Mutex<HashMap<String,
                                            (Option<Array2<f32>>, Option<Array4<f32>>)>>>,
                                          condition_var: &Arc<(Mutex<Vec<String>>, Condvar)>,
                                          threads: &mut Vec<io::Result<JoinHandle<()>>>) {
  let result =
    search_node_who_shares_input(
      &arc_model.graph.node[position as usize + 1..arc_model.graph.node.len()],
      &node.output[0]);
  if result.is_some() {
    let mut vec_to_add = result.clone().unwrap().1;
    let parallel_nodes = result.unwrap().0;

    for el in vec_to_add.iter_mut() {
      *el += position + 1;
    }
    position_to_skip.extend(vec_to_add);

    let mut n_t = 0;
    for group in parallel_nodes {
      let t_map = hashmap_outputs_to_inputs.clone();
      let t_model = arc_model.clone();
      let t_condvar = condition_var.clone();

      threads.push(thread::Builder::new()
        .name(format!("{}{}", "Thread", n_t))
        .spawn(move || {
          for n in  group {
            node_inference(&n, &t_map, &t_model);

            let (l, cvar) = &*t_condvar;
            let mut new_value_added = l.lock().unwrap();
            new_value_added.push(n.output[0].clone());
            cvar.notify_all();
          }
        }));

      n_t += 1;
    }
  }
}