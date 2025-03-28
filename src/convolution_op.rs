use ndarray::*;
use num_traits::Float;

pub type DataRepresentation<F> = Array4<F>;

// Padding (specific way of adding zeros to the input matrix) kind used in the convolution.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Padding {
  // explicit padding (specified in "pads" parameter)
  NotSet,
  // output has same shape as input; if odd padding number, extra-padding added at bottom
  SameUpper,
  // output has same shape as input; if odd padding number, extra-padding added at top
  SameLower,
  // no padding
  Valid,
}

// Rust implementation of a convolutional layer.
// The weight matrix (aka kernel) shall have dimension (in that order)
// channels/groups(input channels), feature maps(output channels), kernel width, kernel height,
pub struct ConvolutionLayer<F: Float> {
  pub(in crate) kernel: Array4<F>,
  pub(in crate) bias: Option<Array1<F>>,
  pub(in crate) auto_pad: Padding,
  pub(in crate) dilations: Option<Array2<i32>>,
  pub(in crate) group: Option<i32>,
  pub(in crate) pads: Array1<F>,
  pub(in crate) strides: Array1<F>,
}

impl<F: 'static + Float + std::ops::AddAssign + std::default::Default + std::convert::From<F> + std::ops::AddAssign<f32>> ConvolutionLayer<F> where f32: From<F> {
  // Creates new convolution layer.
  pub(crate) fn new(
    kernel: Array4<F>,
    bias: Option<Array1<F>>,
    auto_pad: Padding,
    dilations: Option<Array2<i32>>,
    group: Option<i32>,
    pads: Array1<F>,
    strides: Array1<F>,
  ) -> ConvolutionLayer<F> {
    ConvolutionLayer { kernel, bias, auto_pad, dilations, group, pads, strides }
  }

  /// Creates new convolution layer. The weights are given in ONNX Tensorflow layout:
  /// feature maps(output channels), channels/groups(input channels), kernel height, kernel width
  /// converted into:
  /// channels/groups(input channels), feature maps(output channels), kernel width, kernel height,
  pub fn new_onnx_tensor_flow(
    kernel: Array4<F>,
    bias: Option<Array1<F>>,
    auto_pad: Padding,
    dilations: Option<Array2<i32>>,
    group: Option<i32>,
    pads: Array1<F>,
    strides: Array1<F>,
  ) -> ConvolutionLayer<F> {
    let permuted_view = kernel.view().permuted_axes([1, 0, 3, 2]);
    // Hack to fix the memory layout, permuted axes makes a
    // col major array / non-contiguous array from kernel
    let permuted_array: Array4<F> = Array::from_shape_vec(permuted_view.dim(), permuted_view.iter().copied().collect()).unwrap();
    ConvolutionLayer::new(permuted_array, bias, auto_pad, dilations, group, pads, strides)
  }

  /// Analog to conv2d.
  pub fn convolve(&self, image: &DataRepresentation<F>) -> DataRepresentation<F> {
    conv2d(
      &self.kernel,
      image,
      self.bias.as_ref(),
      self.auto_pad,
      self.dilations.as_ref(),
      self.group,
      &self.pads,
      &self.strides,
    )
  }
}

/// OPSET VERSION: 8
/// Performs a convolution on the given image data using this layers parameters.
/// We always convolve on flattened images and expect the input array in im2col
/// style format.
///
/// Read more here:
/// - <https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster>
///
/// Input:
///
///  - im2d(batch size, channels, height, width): Array4.
///  - kernel_weights(F=#Filters(channels/groups), C=#ChannelsOut(feature maps), width, height): Array4. (Feature Maps->#output volume)
///  - bias: Array1. (Bias, is added to each channel (after having adding each Hadamard Product))
///  - auto_pad: ["NOTSET"->pads has meant not to be None (manually specified padding),
///               "SAME_UPPER"->padding equally split between axis(if odd number, extra padding added to bottom),
///               "SAME_LOWER"->padding equally split between axis(if odd number, extra padding added to top,
///               "VALID"->no padding
///               ]
///  - dilations: Array1. (Dilation over kernel a.k.a. w filter)
///  - group: i32. Number of groups
///  - kernel_shape: Array1. If not None means the shape of kernel a.k.a. w filter. Since it's not required from the standard, it's inferred from kernel_weights
///  - pads: Array1. Manual padding specified accordingly to auto_pad
///  - strides: Array1. Moving offset over each x input axis.
/// Returns:
/// -----------------------------------------------
/// - out: Output data, of shape (B, F, H', W')
#[allow(unused_assignments)]
#[allow(unused_variables)]
pub fn conv2d<'a, T, V, F: 'static + Float + std::ops::AddAssign + std::default::Default + std::ops::AddAssign<f32>>(
  kernel_weights: T,
  im2d: T,
  bias: Option<&Array1<F>>,
  auto_pad: Padding,
  dilations: Option<&Array2<i32>>,
  group: Option<i32>,
  pads: V, //Option<&Array1<F>>
  strides: V, //Option<&Array1<F>>
) -> DataRepresentation<F>
  where
  // This trait bound ensures that kernel and im2d can be passed as owned array or view.
  // AsArray just ensures that im2d can be converted to an array view via ".into()".
  // Read more here: https://docs.rs/ndarray/0.12.1/ndarray/trait.AsArray.html
    T: AsArray<'a, F, Ix4>,
    V: AsArray<'a, F, Ix1>, f32: From<F>
{
  // Initialisations
  let im2d_arr: ArrayView4<F> = im2d.into();
  let kernel_weights_arr: ArrayView4<F> = kernel_weights.into();
  let strides_arr: ArrayView1<F> = strides.into();
  let pads_arr: ArrayView1<F> = pads.into();
  let im_col: Array2<F>; // output of fn: im2col_ref()
  let ker_col: Array2<F>;
  let new_im_height: usize;
  let new_im_width: usize;
  let weight_shape = kernel_weights_arr.shape();

  assert!(im2d_arr.shape()[1] == (weight_shape[0] * group.unwrap() as usize) && weight_shape[1] % group.unwrap() as usize == 0);

  let mut num_filters = weight_shape[0];
  match group {
    Some(g) => num_filters = num_filters / g as usize,
    None => {}
  }
  let num_channels_out = weight_shape[1];
  let kernel_height = weight_shape[3];
  let kernel_width = weight_shape[2];
  let mut pads_height_start: usize = 0;
  let mut pads_height_end: usize = 0;
  let mut pads_width_start: usize = 0;
  let mut pads_width_end: usize = 0;
  if auto_pad == Padding::NotSet {
    let pads_height_start_as_f = *pads_arr.get(0).unwrap();
    let pads_height_start_as_f32: f32 = pads_height_start_as_f.into();
    pads_height_start = pads_height_start_as_f32 as usize;
    let pads_height_end_as_f = *pads_arr.get(2).unwrap();
    let pads_height_end_as_f32: f32 = pads_height_end_as_f.into();
    pads_height_end = pads_height_end_as_f32 as usize;
    let pads_width_start_as_f = *pads_arr.get(1).unwrap();
    let pads_width_start_as_f32: f32 = pads_width_start_as_f.into();
    pads_width_start = pads_width_start_as_f32 as usize;
    let pads_width_end_as_f = *pads_arr.get(3).unwrap();
    let pads_width_end_as_f32: f32 = pads_width_end_as_f.into();
    pads_width_end = pads_width_end_as_f32 as usize;
  }

  let im_batch_size = im2d_arr.len_of(Axis(0));
  let im_channel = im2d_arr.len_of(Axis(1));
  let im_height = im2d_arr.len_of(Axis(2));
  let im_width = im2d_arr.len_of(Axis(3));
  let im_height_stride_as_f = *strides_arr.get(0).unwrap();
  let im_height_stride_as_f32: f32 = im_height_stride_as_f.into();
  let im_height_stride = im_height_stride_as_f32 as usize;
  let im_width_stride_as_f = *strides_arr.get(1).unwrap();
  let im_width_stride_as_f32: f32 = im_width_stride_as_f.into();
  let im_width_stride = im_width_stride_as_f32 as usize;

  // Calculate output shapes H', W' for two types of Padding
  match auto_pad {
    Padding::SameLower => {
      // H' = (H / stride).ceil()
      // W' = (W / stride).ceil()
      let new_im_height_float = (im_height as f32 / im_height_stride as f32).ceil();
      let new_im_width_float = (im_width as f32 / im_width_stride as f32).ceil();

      new_im_height = new_im_height_float as usize;
      new_im_width = new_im_width_float as usize;
    }
    Padding::SameUpper => {
      // H' = (H / stride).ceil()
      // W' = (W / stride).ceil()
      let new_im_height_float = (im_height as f32 / im_height_stride as f32).ceil();
      let new_im_width_float = (im_width as f32 / im_width_stride as f32).ceil();

      new_im_height = new_im_height_float as usize;
      new_im_width = new_im_width_float as usize;
    }
    Padding::NotSet => {
      // H' = {[H - HH + (2*padding)] / stride}+ 1
      // W' = {[W - WW + (2*padding)] / stride} + 1
      new_im_height = ((im_height - kernel_height + (pads_height_start + pads_height_end)) / im_height_stride) + 1;
      new_im_width = ((im_width - kernel_width + (pads_width_start + pads_width_end)) / im_width_stride) + 1;
    }
    Padding::Valid => {
      // H' =  ((H - HH) / stride_height) + 1
      // W' =  ((W - WW) / stride_width) + 1
      new_im_height = ((im_height - kernel_height) / im_height_stride) + 1;
      new_im_width = ((im_width - kernel_width) / im_width_stride) + 1;
    }
  };

  ker_col = ker2col_ref(
    kernel_weights_arr,
    kernel_height,
    kernel_width,
    num_channels_out,
    num_filters
  );

  if auto_pad != Padding::Valid {
    let mut pad_num_h = 0;
    let mut pad_num_w = 0;
    let mut pad_top = 0;
    let mut pad_bottom = 0;
    let mut pad_left = 0;
    let mut pad_right = 0;
    if auto_pad == Padding::SameUpper || auto_pad == Padding::SameLower {
      (pad_num_h, pad_num_w, pad_top, pad_bottom, pad_left, pad_right) = get_padding_size(im_height, im_width, im_height_stride, im_width_stride, kernel_height, kernel_width);
    } else if auto_pad == Padding::NotSet {
      pad_top = pads_height_start;
      pad_bottom = pads_height_end;
      pad_left = pads_width_start;
      pad_right = pads_width_end;
      pad_num_h = pads_height_start + pads_height_end;
      pad_num_w = pads_width_start + pads_width_end;
    }
    let mut im2d_arr_pad: Array4<F> = Array4::zeros((
      im_batch_size,
      im_channel,
      im_height + pad_num_h,
      im_width + pad_num_w,
    ));
    let pad_bottom_int = (im_height + pad_num_h) - pad_bottom;
    let pad_right_int = (im_width + pad_num_w) - pad_right;
    // https://github.com/rust-ndarray/ndarray/issues/823
    im2d_arr_pad
      .slice_mut(s![.., .., pad_top..pad_bottom_int, pad_left..pad_right_int])
      .assign(&im2d_arr);

    let im_height_pad = im2d_arr_pad.len_of(Axis(2));
    let im_width_pad = im2d_arr_pad.len_of(Axis(3));

    im_col = im2col_ref(
      im2d_arr_pad.view(),
      kernel_height,
      kernel_width,
      im_height_pad,
      im_width_pad,
      im_channel,
      im_height_stride,
      im_width_stride,
      dilations,
    );
  } else {
    im_col = im2col_ref(
      im2d_arr,
      kernel_height,
      kernel_width,
      im_height,
      im_width,
      im_channel,
      im_height_stride,
      im_width_stride,
      dilations,
    );
  }

  /*
  println!("ker col:");
  for row in ker_col.rows() {
    for &elem in row.iter() {
      print!("{:?}, ", f32::from(elem));
    }
    println!("");
  }*/

  let mut output = Array4::zeros((im_batch_size, num_channels_out, new_im_height, new_im_width));

  let mut image_height = 0usize;
  let mut image_width = 0usize;
  let mut displacement = 0;
  //println!("num_channels_out: {}, num_channels_in: {}, image_as_row[{}, {}], kernel_as_row: [{}, {}]", num_channels_out, num_filters, im_col.len_of(Axis(0)), im_col.len_of(Axis(1)), ker_col.len_of(Axis(0)), ker_col.len_of(Axis(1)));
  for num_channel_output in 0..num_channels_out{
    image_height = 0usize;
    image_width = 0usize;
    let image_start = 0usize;
    let mut image_end = im_col.len_of(Axis(0));
    let kernel_start = num_channel_output;
    let kernel_end= num_channel_output + 1;
    let image_idx_start = 0usize;
    let mut image_idx_end = 1usize;
    if im_channel > 1{
      image_end = 1;
      image_idx_end = (im_col.len_of(Axis(0))/im_channel)+image_idx_start;
    }
    //println!("image_start: {}, image_end: {}, kernel_start: {}, kernel_end: {}, image_idx_start: {}, image_idx_end: {}", image_start, image_end, kernel_start, kernel_end, image_idx_start, image_idx_end);

    for aus_img_idx in image_idx_start..image_idx_end{
      for channel_in in 0..im_channel {
        for row in image_start..image_end {
          for n_filter_input in kernel_start..kernel_end {
            let mut im_col_idx = row;
            let mut im_ker_idx = n_filter_input;
            if im_channel > 1 {
              im_col_idx = ((channel_in * im_col.len_of(Axis(0))) / im_channel)+aus_img_idx;
              im_ker_idx = channel_in + (im_channel * num_channel_output);
              if im_col_idx >= im_col.len_of(Axis(0)){
                im_col_idx = im_col_idx - im_col.len_of(Axis(0));
              }
              //println!("im_col_idx:{}", im_col_idx);
            }

            let im_row = im_col.slice(s![im_col_idx, ..]);
            let ker_row = ker_col.slice(s![im_ker_idx, ..]);

            /*
            if num_channel_output == 1 && image_height == 0 && image_width == 0 {
              println!("image row:");
              for row in im_row.rows() {
                for &elem in row.iter() {
                  print!("{:?}, ", f32::from(elem));
                }
                println!("");
              }
              println!("ker row:");
              for row in ker_row.rows() {
                for &elem in row.iter() {
                  print!("{:?}, ", f32::from(elem));
                }
                println!("");
              }
            }
            */

            assert_eq!(im_row.len_of(Axis(0)), ker_row.len_of(Axis(0)));

            let mut row_mul = Array1::zeros(im_row.len());
            for idx in 0..im_row.len() {
              row_mul[[idx]] = f32::from(im_row[[idx]]) * f32::from(ker_row[[idx]]);
            }

            /*
            if num_channel_output == 1 && image_height == 0 && image_width == 0 {
              println!("row mul:");
              for row in row_mul.rows() {
                for &elem in row.iter() {
                  print!("{:?}, ", elem);
                }
                println!("");
              }

              println!("output[{},{},{},{}] = {}", 0, num_channel_output, image_height, image_width, row_mul.sum());
            }
            */

            output[[0, num_channel_output, image_height, image_width]] += row_mul.sum();

          }
          if im_channel <= 1 {
            if image_width + 1 < new_im_width {
              image_width += 1;
            } else {
              image_height += 1;
              image_width = 0;
            }
          }
        }
      }
      if im_channel > 1 {
        if image_width + 1 < new_im_width {
          image_width += 1;
        } else {
          image_height += 1;
          image_width = 0;
        }
      }
    }

    displacement+=1;
  }

  /*
  println!("output:");
  for row in output.rows() {
    for &elem in row.iter() {
      print!("{:?}, ", f32::from(elem));
    }
    println!("");
  }
  */

  add_bias(&output, bias)
}

pub(in crate) fn get_padding_size(
  input_h: usize,
  input_w: usize,
  stride_h: usize,
  stride_w: usize,
  kernel_h: usize,
  kernel_w: usize,
) -> (usize, usize, usize, usize, usize, usize) {
  let pad_along_height: usize;
  let pad_along_width: usize;
  let idx_0: usize = 0;

  if input_h % stride_h == idx_0 {
    pad_along_height = (kernel_h - stride_h).max(idx_0);
  } else {
    pad_along_height = (kernel_h - (input_h % stride_h)).max(idx_0);
  };
  if input_w % stride_w == idx_0 {
    pad_along_width = (kernel_w - stride_w).max(idx_0);
  } else {
    pad_along_width = (kernel_w - (input_w % stride_w)).max(idx_0);
  };

  let pad_top = pad_along_height / 2;
  let pad_bottom = pad_along_height - pad_top;
  let pad_left = pad_along_width / 2;
  let pad_right = pad_along_width - pad_left;

  // yes top/bottom and right/left are swapped. No, I don't know
  // why this change makes it conform to the pytorch implementation.
  (
    pad_along_height,
    pad_along_width,
    pad_bottom,
    pad_top,
    pad_right,
    pad_left,
  )
}

#[allow(unused_assignments)]
pub(in crate) fn im2col_ref<'a, T, F: 'a + Float + std::default::Default>(
  im_arr: T,
  ker_height: usize,
  ker_width: usize,
  im_height: usize,
  im_width: usize,
  im_channel: usize,
  stride_h: usize,
  stride_w: usize,
  dilations: Option<&Array2<i32>>,
) -> Array2<F>
  where
  // Args:
  //   im_arr: image matrix to be translated into columns, (C,H,W)
  //   ker_height: filter height (hh)
  //   ker_width: filter width (ww)
  //   im_height: image height
  //   im_width: image width
  //
  // Returns:
  //   col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
  //         new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    T: AsArray<'a, F, Ix4>
{
  let mut cols_img: Array2<F> = Default::default();
  let mut cont = 0_usize;
  match dilations {
    Some(dilations) => {
      let dilation_h = dilations[[0, 0]] as usize;
      let dilation_w = dilations[[0, 1]] as usize;
      if dilation_h > 1 || dilation_w > 1 {
        let im2d_arr: ArrayView4<F> = im_arr.into();
        let new_h = ((im_height - dilation_h * (ker_height - 1) - 1) / stride_h) + 1;
        let new_w = ((im_width - dilation_w * (ker_width - 1) - 1) / stride_w) + 1;
        cols_img = Array2::zeros((new_h * new_w, im_channel * ker_height * ker_width));
        for i in 1..new_h + 1 {
          for j in 1..new_w + 1 {
            let h_start = (i - 1) * stride_h;
            let h_end = ((((i - 1) * stride_h + ker_height) - (i - 1) * stride_h) * dilation_h) + (i - 1) * stride_h;
            let w_start = (j - 1) * stride_w;
            let w_end = ((((j - 1) * stride_w + ker_width) - (j - 1) * stride_w) * dilation_w) + (j - 1) * stride_w;
            let patch = im2d_arr.slice(s![
                  ..,
                  ..,
                  h_start..h_end; dilation_h,
                  w_start..w_end; dilation_w
              ]);
            let patchrow_unwrap: Array1<F> = Array::from_iter(patch.map(|a| *a));

            cols_img.row_mut(cont).assign(&patchrow_unwrap);
            cont += 1;
          }
        }
      } else {
        let im2d_arr: ArrayView4<F> = im_arr.into();
        let new_h = ((im_height - ker_height) / stride_h) + 1;
        let new_w = ((im_width - ker_width) / stride_w) + 1;
        cols_img = Array2::zeros((new_h * new_w * im_channel, ker_height * ker_width));

        for k in 0..im_channel {
          for i in 1..new_h + 1 {
            for j in 1..new_w + 1 {
              let patch = im2d_arr.slice(s![
                  0,
                  k,
                  (i - 1) * stride_h..((i - 1) * stride_h + ker_height),
                  (j - 1) * stride_w..((j - 1) * stride_w + ker_width),
              ]);
              let patchrow_unwrap: Array1<F> = Array::from_iter(patch.map(|a| *a));

              cols_img.row_mut(cont).assign(&patchrow_unwrap);
              cont += 1;
            }
          }
        }
      }
    }
    None => {
      let im2d_arr: ArrayView4<F> = im_arr.into();
      let new_h = ((im_height - ker_height) / stride_h) + 1;
      let new_w = ((im_width - ker_width) / stride_w) + 1;
      cols_img = Array2::zeros((new_h * new_w * im_channel, ker_height * ker_width));

      for k in 0..im_channel {
        for i in 1..new_h + 1 {
          for j in 1..new_w + 1 {
            let patch = im2d_arr.slice(s![
                  0,
                  k,
                  (i - 1) * stride_h..((i - 1) * stride_h + ker_height),
                  (j - 1) * stride_w..((j - 1) * stride_w + ker_width),
              ]);
            let patchrow_unwrap: Array1<F> = Array::from_iter(patch.map(|a| *a));

            cols_img.row_mut(cont).assign(&patchrow_unwrap);
            cont += 1;
          }
        }
      }
    }
  };

  cols_img
}

#[allow(unused_assignments)]
pub(in crate) fn ker2col_ref<'a, T, F: 'a + Float + std::default::Default>(
  im_arr: T,
  ker_height: usize,
  ker_width: usize,
  num_channels: usize,
  num_filters: usize) -> Array2<F> where T: AsArray<'a, F, Ix4>, f32: From<F>
{
  let mut cols_img: Array2<F> = Default::default();
  let im2d_arr: ArrayView4<F> = im_arr.into();
  cols_img = Array2::zeros((num_channels * num_filters, ker_height * ker_width));
  let mut cont = 0usize;

  for k in 0..num_channels {
    for w in 0..num_filters{
      let mut channel_kernel = Array1::zeros(ker_height * ker_width);
      let mut kernel_arr: Vec<F> = vec![];
      for i in 1..ker_height + 1 {
        for j in 1..ker_width + 1 {
          let patch = im2d_arr.slice(s![
                w,
                k,
                j - 1,
                i - 1
            ]);
          for p in patch.iter() {
            kernel_arr.push(*p);
          }
        }
      }
      let patchrow_unwrap = Array::from_shape_vec(ker_height * ker_width, kernel_arr).unwrap();
      channel_kernel.assign(&patchrow_unwrap);
      cols_img.row_mut(cont).assign(&channel_kernel);
      cont += 1;
    }
  }

  cols_img
}

pub(in crate) fn add_bias<F>(x: &Array4<F>, bias: Option<&Array1<F>>) -> Array4<F>
  where
    F: 'static + Float + std::ops::AddAssign,
{
  if let Some(bias_array) = bias {
    assert_eq!(bias_array.shape()[0], x.shape()[1], "Bias array has the wrong shape {:?} for vec of shape {:?}", bias_array.shape(), x.shape());
    // Yes this is really necessary. Broadcasting with ndarray-rust
    // starts at the right side of the shape, so we have to add
    // the axes by hand (else it thinks that it should compare the
    // output width and the bias channels).
    (x + &bias_array
      .clone()
      .insert_axis(Axis(1))
      .insert_axis(Axis(2))
      .broadcast(x.shape())
      .unwrap())
      .into_dimensionality()
      .unwrap()
  } else {
    x.clone()
  }
}

#[allow(dead_code)]
fn test_convolution_1_channels_out_1_channels_in() {
  // Input has shape (batch_size, channels, height, width)
  let input = Array::from_shape_vec(
    (1, 1, 5, 6),
     vec![1., 2., 3., 4., 5., 6.,
         7., 8., 9., 10., 11., 12.,
         13., 14., 15., 16., 17., 18.,
         19., 20., 21., 22., 23., 24.,
         25., 26., 27., 28., 29., 30.]
  )
    .unwrap();

  // Kernel has shape (channels in, channels out, height, width)
  let kernel: Array4<f32> = Array::from_shape_vec(
    (1, 1, 5, 2),
    vec![1., 2.,
         3., 4.,
         5., 6.,
         7., 8.,
         9., 10.]
  )
    .unwrap();

  let strides: Array1<f32> = array![1., 1.];
  let pads: Array1<f32> = array![0., 0., 0., 0.];

  let conv_layer =
    ConvolutionLayer::new_onnx_tensor_flow(kernel.clone(), None, Padding::NotSet, None, Some(1), pads, strides);
  let output_layer: Array4<f32> = conv_layer.convolve(&input);

  println!("test_convolution_1_channels_out_1_channels_in: {:?}", output_layer);
}

#[allow(dead_code)]
fn test_convolution_2_channels_out_2_channels_in() {
  // Input has shape (batch_size, channels, height, width)
  let input = Array::from_shape_vec(
    (1, 2, 5, 6),
    vec![1., 2., 3., 4., 5., 6.,
        7., 8., 9., 10., 11., 12.,
        13., 14., 15., 16., 17., 18.,
        19., 20., 21., 22., 23., 24.,
        25., 26., 27., 28., 29., 30.,
        31., 32., 33., 34., 35., 36.,
        37., 38., 39., 40., 41., 42.,
        43., 44., 45., 46., 47., 48.,
        49., 50., 51., 52., 53., 54.,
        55., 56., 57., 58., 59., 60.]
  )
    .unwrap();

  // Kernel has shape (channels in, channels out, height, width)
  let kernel: Array4<f32> = Array::from_shape_vec(
    (2, 2, 3, 4),
    vec![1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        2., 2., 2., 2.,
        2., 2., 2., 2.,
        2., 2., 2., 2.,
        3., 3., 3., 3.,
        3., 3., 3., 3.,
        3., 3., 3., 3.,
        4., 4., 4., 4.,
        4., 4., 4., 4.,
        4., 4., 4., 4.]
  )
    .unwrap();

  let strides: Array1<f32> = array![1., 1.];
  let pads: Array1<f32> = array![0., 0., 0., 0.];

  let conv_layer =
    ConvolutionLayer::new_onnx_tensor_flow(kernel.clone(), None, Padding::NotSet, None, Some(1), pads, strides);
  let output_layer: Array4<f32> = conv_layer.convolve(&input);

  println!("test_convolution_2_channels_out_2_channels_in: {:?}", output_layer);
}

#[allow(dead_code)]
fn test_convolution_1_channel_out_2_channel_in(){
  // Input has shape (batch_size, channels, height, width)
  let input = Array::from_shape_vec(
    (1, 1, 7, 5),
    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0],
  )
    .unwrap();

  // Kernel has shape (channels in, channels out, height, width)
  let kernel: Array4<f32> = Array::from_shape_vec(
    (2, 1, 3, 4),
    vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.],
  )
    .unwrap();

  let strides: Array1<f32> = array![1., 1.];
  let pads: Array1<f32> = array![0., 0., 0., 0.];

  let conv_layer =
    ConvolutionLayer::new_onnx_tensor_flow(kernel.clone(), None, Padding::NotSet, None, Some(1), pads, strides);
  let output_layer: Array4<f32> = conv_layer.convolve(&input);

  println!("test_convolution_1_channel_out_2_channel_in: {:?}", output_layer);
}

#[allow(dead_code)]
pub fn test_convolution(){
  test_convolution_1_channels_out_1_channels_in();
  println!("\n\n");
  test_convolution_1_channel_out_2_channel_in();
  println!("\n\n");
  test_convolution_2_channels_out_2_channels_in();
}