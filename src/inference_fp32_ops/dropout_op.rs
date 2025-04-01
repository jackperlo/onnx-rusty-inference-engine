use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use ndarray::{Array4, Array, Array2};
use onnx_protobuf::NodeProto;
use rand::{Rng, SeedableRng};

/// This function manages the dropout operation
/// # Arguments
/// * output_container: contains the partial results computed by inference operations
/// * node: node on which dropout has to be performed
pub fn drop_out(output_container: &Arc<Mutex<HashMap<String,
  (Option<Array2<f32>>, Option<Array4<f32>>)>>>,
               node: &NodeProto) {
  let map = output_container.lock().unwrap();
  let input = Array4::from(
    map.get(node.input[0].as_str()).unwrap().1.clone().unwrap());

  drop(map);

  let mut ratio: Option<f32> = None;
  for attr in &node.attribute {
    ratio = match attr.name.as_ref() {
      "ratio" => Some(attr.f),
      _ => panic!("ATTRIBUTE NAME FOR DROP OUT NOT FOUND, {}",
                  <String as AsRef<str>>::as_ref(&attr.name))
    };
  }
  let output_layer = dropout_wrapper(input,
                                         ratio,
                                         None,
                                         false,
                                         false);

  /*
   TODO: manage mask
   if output_layer.1.is_some() {
     println!("Mask: {:?}", output_layer.1.unwrap());
     hashmap_outputs_to_inputs.insert(outputs[1].clone(), output_layer.1.unwrap());
   }
  */
  #[cfg(feature = "debug_prints")]{
    dbg!("Dropout: {:?}", output_layer.0.clone());
  }
  #[cfg(feature = "operations_prints")]{
    println!("Dropout, done! by {}", thread::current().name().unwrap_or("MAIN THREAD"));
  }
  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (None, Some(output_layer.0)));
}

///VERSION 7 of Dropout Operation
fn dropout_wrapper(
  x: Array4<f32>,
  ratio: Option<f32>,
  seed: Option<u64>,
  training_mode: bool,
  return_mask: bool,
) -> (Array4<f32>, Option<Array4<bool>>) {
  let mut drop_probability = 0.5;
  match ratio {
    Some(drop) => drop_probability = drop,
    None => {}
  };

  if drop_probability == 0.0 || !training_mode {
    if return_mask {
      return (x.clone(), Some(Array4::from_elem(x.raw_dim(), true)))
    }
    return (x, None);
  }

  let mut rng = match seed {
    Some(seed_value) => rand::rngs::SmallRng::seed_from_u64(seed_value),
    None => rand::rngs::SmallRng::from_entropy(),
  };

  let mask = Array4::from_shape_fn(x.raw_dim(), |_| rng.gen::<f32>() >= drop_probability);
  let scale = 1.0 / (1.0 - drop_probability);

  let mask_num = mask.map(|&x| return if x == true { 1. } else { 0. });
  let masked_x = &x * &mask_num * scale;

  if return_mask {
    (masked_x, Some(mask))
  } else {
    (masked_x, None)
  }
}

#[allow(dead_code)]
fn test_dropout(){
// Input has shape (batch_size, channels, height, width)
  let input = Array::from_shape_vec(
    (1, 1, 4, 4),
    vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.]
  )
    .unwrap();

  println!("Input{:?}", input);

  let _ratio = 0.5; //default ratio
  let output = dropout_wrapper(input, None, None, false, false);

  println!("Output: {:?}", output);
}