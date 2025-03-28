use ndarray::Array;
use ndarray::prelude::*;
use rand::Rng;

//OPSET VERSION = 8
pub fn reshape(data: Array4<f32>, shape: Array1<i64>, allowzero: Option<usize>) -> Array2<f32> {
  return if allowzero.is_none() {
    reshape_implementation(data, shape, 0)   /* Default value of allowzero = 0  */
  } else {
    reshape_implementation(data, shape, allowzero.unwrap())
  }
}

pub fn reshape_implementation(data: Array4<f32>, shape: Array1<i64>, allowzero: usize) -> Array2<f32> {
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
pub fn test_reshape() {
  // Esempio di utilizzo
  let mut rng = rand::thread_rng();

  let original_shape: Vec<i64> = vec![16, 3];
  let orig: Array1<i64> = original_shape.into();

  let data = Array::from_shape_vec((4, 2, 2, 3), (0..48).map(|_| rng.gen::<f32>()).collect()).unwrap();

  let reshaped = reshape(data.into(), orig, Some(0));

  println!("reshaped: \n{:?}", reshaped);
}
