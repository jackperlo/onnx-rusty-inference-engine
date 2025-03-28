use ndarray::{Array, Axis};
use ndarray::prelude::*;

//OPSET VERSION = 8
pub fn softmax(input: Array4<f32>, axis: Option<usize>) -> Array2<f32> {
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
pub fn test_softmax() {
  // Esempio di utilizzo
  let x = Array::from_shape_vec((1, 1, 2,4), vec![118.85734,5640.1426,2.,3.,1000.,1001.,1002.,1003.]).unwrap();
  let _x_2 = Array::from_shape_vec((1, 1, 1,3), vec![-1.,0.,1.]).unwrap();
  println!("input: \n{:?}", x);
  let result = softmax(x, None);
  println!("output: \n{:?}", result);
}
