use ndarray::{Array, Array4, Axis};

//OPSET VERSION. N channel out
pub(crate) fn global_average_pool(x: Array4<f32>) -> Array4<f32> {
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
pub fn test_global_average_pool() {
  //(batch size, channels out, height, width)
  let input = Array::from_shape_vec(
    (1, 2, 4, 4),
    vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.]
  )
    .unwrap();
  println!("input: {:?}", input);
  let output = global_average_pool(input);
  println!("output: {:?}", output);
}
