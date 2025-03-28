use ndarray::{Array4, Array};
use rand::{Rng, SeedableRng};

///VERSION 7 of Dropout Operation
pub fn dropout(
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
pub fn test_dropout(){
// Input has shape (batch_size, channels, height, width)
  let input = Array::from_shape_vec(
    (1, 1, 4, 4),
    vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.]
  )
    .unwrap();

  println!("Input{:?}", input);

  let _ratio = 0.5; //default ratio
  let output = dropout(input, None, None, false, false);

  println!("Output: {:?}", output);
}