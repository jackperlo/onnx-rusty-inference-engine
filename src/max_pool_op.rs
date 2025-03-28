use ndarray::*;
use num_traits::Float;

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
// specified by kernel_size
pub struct ConvolutionLayer<F: Float> {
  pub(in crate) auto_pad: Padding,
  pub(in crate) pads: Array1<F>,
  pub(in crate) kernel_size: Array2<i32>,
  pub(in crate) storage_order: Option<i32>,
  pub(in crate) strides: Array1<F>,
}

impl<F: 'static + Float + std::ops::AddAssign> ConvolutionLayer<F> where f32: From<F> {
  // Creates new convolution layer.
  pub fn new(
    auto_pad: Padding,
    pads: Array1<F>,
    kernel_size: Array2<i32>,
    storage_order: Option<i32>,
    strides: Array1<F>,
  ) -> ConvolutionLayer<F> {
    ConvolutionLayer { auto_pad, pads, kernel_size, storage_order, strides }
  }

  /// Analog to max_pool2d
  pub fn max_pool(&self, image: &Array4<F>) -> Array4<F> {
    max_pool2d(
      image,
      self.auto_pad,
      &self.pads,
      &self.kernel_size,
      self.storage_order.as_ref(),
      &self.strides,
    )
  }
}

/// OPSET VERSION: 8
/// Performs a max pooling on the given image data using this layers parameters.
/// We always max pool on flattened images and expect the input array in im2col
/// style format.
///
/// Read more here:
/// - <https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster>
///
/// Input:
///
///  2D images legend:
///  - im2d(batch size, channels, height, width): Array4.
///  - auto_pad: ["NOTSET"->pads has meant not to be None (manually specified padding),
///               "SAME_UPPER"->padding equally split between axis(if odd number, extra padding added to bottom),
///               "SAME_LOWER"->padding equally split between axis(if odd number, extra padding added to top,
///               "VALID"->no padding
///               ]
///  - pads: Array1. Manual padding specified accordingly to auto_pad
///  - kernel_size: Array2. It specifies the height and width size of kernel
///  - storage_order: Option<i32>. 0-> default is row major, 1-> is col major
///  - strides: Array1. Moving offset over each x input axis.
/// Returns:
/// -----------------------------------------------
/// - out: Output data, of shape (B, F, H', W')
#[allow(unused_variables)]
#[allow(unused_assignments)]
pub fn max_pool2d<'a, T, V, F: 'static + Float + std::ops::AddAssign>(
  im2d: T,
  auto_pad: Padding,
  pads: V,
  kernel_size: &Array2<i32>,
  _storage_order: Option<&i32>,
  strides: V,
) -> Array4<F>
  where
  // This trait bound ensures that kernel and im2d can be passed as owned array or view.
  // AsArray just ensures that im2d can be converted to an array view via ".into()".
  // Read more here: https://docs.rs/ndarray/0.12.1/ndarray/trait.AsArray.html
    T: AsArray<'a, F, Ix4>,
    V: AsArray<'a, F, Ix1>, f32: From<F>
{
  // Initialisations
  let im2d_arr: ArrayView4<F> = im2d.into();
  let kernel_size_arr: ArrayView2<i32> = kernel_size.into();
  let strides_arr: ArrayView1<F> = strides.into();
  let pads_arr: ArrayView1<F> = pads.into();
  let im_col: Array2<F>; // output of fn: im2col_ref()
  let new_im_height: usize;
  let new_im_width: usize;

  let kernel_height = kernel_size_arr.len_of(Axis(0));
  let kernel_width = kernel_size_arr.len_of(Axis(1));

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
      // H' =  (H - HH) / (stride_height + 1)
      // W' =  (W - WW) / (stride_width + 1)
      new_im_height = ((im_height - kernel_height) / im_height_stride) + 1;
      new_im_width = ((im_width - kernel_width) / im_width_stride) + 1;
    }
  };

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
    let mut im2d_arr_pad: Array4<F> = Array::zeros((
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
    );
  }


  let mut output: Array4<F> = Array4::zeros((im_batch_size, im_channel, new_im_height, new_im_width));
  let mut image_height = 0usize;
  let mut image_width = 0usize;
  let mut displacement = 0;
  //println!("num_channels_out: {}, num_channels_in: {}, image_as_row[{}, {}]", num_channels_out, num_filters, im_col.len_of(Axis(0)), im_col.len_of(Axis(1)));
  for num_channel_output in 0..im_channel {
    image_height = 0usize;
    image_width = 0usize;
    let mut image_start = 0usize;
    let mut image_end = im_col.len_of(Axis(0));
    if im_channel > 1 {
      image_start = (displacement * im_col.len_of(Axis(0))) / im_channel;
      if image_start >= im_col.len_of(Axis(0)) {
        image_start = 0;
        displacement = 0;
      }
      image_end = (im_col.len_of(Axis(0)) / im_channel) + image_start;
    }
    //println!("image_start: {}, image_end: {}", image_start, image_end);
    for row in image_start..image_end {
      let im_row = im_col.slice(s![row, ..]);
      /*
        println!("image row:");
        for row in im_row.rows() {
          for &elem in row.iter() {
            print!("{:?}, ", f32::from(elem));
          }
          println!("");
        }
       */
      assert_eq!(im_row.len_of(Axis(0)), kernel_height * kernel_width);

      output[[0, num_channel_output, image_height, image_width]] = im_row.iter().fold(F::min_value(), |max: F, &x| max.max(x.into()));

      if image_width + 1 < new_im_width {
        image_width += 1;
      } else {
        image_height += 1;
        image_width = 0;
      }
    }
    displacement += 1;
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

  output
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

pub(in crate) fn im2col_ref<'a, T, F: 'a + Float>(
  im_arr: T,
  ker_height: usize,
  ker_width: usize,
  im_height: usize,
  im_width: usize,
  im_channel: usize,
  stride_h: usize,
  stride_w: usize,
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
    T: AsArray<'a, F, Ix4>,
{
  let im2d_arr: ArrayView4<F> = im_arr.into();
  let new_h = ((im_height - ker_height) / stride_h) + 1;
  let new_w = ((im_width - ker_width) / stride_w) + 1;
  let mut cols_img: Array2<F> = Array::zeros((new_h * new_w * im_channel, ker_height * ker_width));
  let mut cont = 0_usize;

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
  cols_img
}

#[allow(dead_code)]
pub fn test_max_pool() {
  // Input has shape (batch_size, channels, height, width)
  let input = Array::from_shape_vec(
    (1, 1, 4, 4),
    vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.],
  )
    .unwrap();

  println!("{:?}", input);
  // Kernel has shape (channels in, channels out, height, width)
  let kernel_size: Array2<i32> = Array::zeros((3, 3));

  let strides: Array1<f32> = array![1., 1.];
  let pads: Array1<f32> = array![0., 0., 0., 0.];

  let conv_layer =
    ConvolutionLayer::new(Padding::Valid, pads, kernel_size, Some(0), strides);
  let output_layer: Array4<f32> = conv_layer.max_pool(&input);

  println!("Layer: {:?}", output_layer);
}
