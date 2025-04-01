use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use ndarray::{Array, Array2, Array4, Axis};
use onnx_protobuf::NodeProto;

/// This function manages the global average pool operation
/// # Arguments
/// * output_container: contains the partial results computed by inferences operations
/// * node: node on which global average pool has to be performed
pub fn global_average_pool(output_container: &Arc<Mutex<HashMap<String,
  (Option<Array2<f32>>, Option<Array4<f32>>)>>>,
                          node: &NodeProto) {
  let map = output_container.lock().unwrap();
  let input = Array4::from(
    map.get(node.input[0].as_str()).unwrap().1.clone().unwrap());

  drop(map);

  let output_layer = global_average_pool_wrapper(input);
  #[cfg(feature = "debug_prints")]{
    dbg!(output_layer);
  }
  #[cfg(feature = "operations_prints")]{
    println!("Global Average Pool, done! by {}",
             thread::current().name().unwrap_or("MAIN THREAD"));
  }
  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (None, Some(output_layer)));
}

//OPSET VERSION. N channel out
fn global_average_pool_wrapper(x: Array4<f32>) -> Array4<f32> {
  let mut output: Array4<f32> = Array::zeros((
    x.len_of(Axis(0)),
    x.len_of(Axis(1)),
    1,
    1,
  ));

  let ch = x.len_of(Axis(1)); //foreach channel

  for c in 0..ch {
    let channel_slice = x.index_axis(Axis(1), c);
    let mut sum: f32 = channel_slice.iter().sum();
    let counter = channel_slice.len();
    sum = sum / counter as f32;
    output[[0, c, 0, 0]] = sum;
  }

  output
}

#[allow(dead_code)]
fn test_global_average_pool() {
  //(batch size, channels out, height, width)
  let input = Array::from_shape_vec(
    (1, 2, 4, 4),
    vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.]
  )
    .unwrap();
  println!("input: {:?}", input);
  let output = global_average_pool_wrapper(input);
  println!("output: {:?}", output);
}
