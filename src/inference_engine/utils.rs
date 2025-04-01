use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use ndarray::{Array1, Array2, Array3, Array4};
use onnx_protobuf::{ModelProto, NodeProto, TensorProto, ValueInfoProto};
use onnx_protobuf::type_proto::Value;
use onnx_protobuf::tensor_shape_proto::dimension::Value::{DimParam, DimValue};

/// This function searches if a node's input is already in the model's tensor initializers
/// # Arguments
/// * model_initializers: model's tensor initializers
/// * input: input to look for
/// # Returns
/// It returns true if it's present, false otherwise
pub fn already_into_initializer(model_initializers: &Vec<TensorProto>, input_name: &str) -> bool {
  for init in model_initializers {
    if <String as AsRef<str>>::as_ref(&init.name) == input_name {
      return true;
    }
  }
  false
}

/// This function inserts into initializers the input(s) data of the onnx model
/// # Arguments
/// * hashmap_outputs_to_inputs: contains the partial results calculated by inferences operations
/// * model: smart pointer which contains the onnx model
/// * input_data: model inputs(s)
/// * input_tensor_names: names of the model input tensors
pub fn manage_input_data(hashmap_outputs_to_inputs: &Arc<Mutex<HashMap<String,
  (Option<Array2<f32>>, Option<Array4<f32>>)>>>,
                         model: &Arc<ModelProto>,
                         input_data: Vec<f32>,
                         input_tensor_name: Vec<&str>) {
  for input_name in input_tensor_name {
    if !already_into_initializer(&model.graph.initializer, input_name) {
      let dims: Vec<&i64> = get_input_data_shape(&model.graph.input, input_name);
      let array = Array4::from_shape_vec(
        (*dims[0] as usize, *dims[1] as usize, *dims[2] as usize, *dims[3] as usize),
        input_data.clone())
        .unwrap();
      let mut map = hashmap_outputs_to_inputs.lock().unwrap();
      map.insert(input_name.to_string(), (None, Some(array)));
    }
  }
}

/// This function searches the node's input shape into the onnx model.
/// # Arguments
/// * model_inputs: list of the inputs
/// * input_name: input to search for
/// # Returns
/// the list of shapes
pub fn get_input_data_shape<'a>(model_inputs: &'a Vec<ValueInfoProto>,
                            input_name: &str)
                            -> Vec<&'a i64> {
  let mut shape = vec![];
  for inp in model_inputs {
    if <String as AsRef<str>>::as_ref(&inp.name) == input_name {
      if inp.type_.value.is_some() {
        match &inp.type_.value.as_ref().unwrap() {
          Value::TensorType(t) => {
            if t.shape.is_some() {
              for el in &t.shape.as_ref().unwrap().dim {
                if el.value.is_some() {
                  match el.value.as_ref().unwrap() {
                    DimValue(v) => { shape.push(v) }
                    DimParam(_) => {panic!("DIM PARAM NOT YET IMPLEMENTED FOR NODE {}'S INPUT",
                                           input_name)} // TODO: check for this case
                    _ => {panic!("SHAPE DIMS UNRECOGNIZED FOR NODE {}'S INPUT", input_name)}
                  };
                }else {
                  panic!("UNABLE TO RETRIEVE DIMS VALUES OF NODE {}'S INPUT", input_name)
                }
              }
            } else {
              panic!("UNABLE TO RETRIEVE SHAPE VALUES OF NODE {}'S INPUT", input_name)
            }
          }
          Value::SequenceType(_) => {panic!("SEQUENCE TYPE NOT YET IMPLEMENTED FOR NODE {}'S INPUT",
                                            input_name)} // TODO: check for this case
          Value::MapType(_) => {panic!("MAP TYPE NOT YET IMPLEMENTED FOR NODE {}'S INPUT",
                                       input_name)} // TODO: check for this case
          Value::OptionalType(_) => {panic!("OPTIONAL TYPE NOT YET IMPLEMENTED FOR NODE {}'S INPUT",
                                            input_name)} // TODO: check for this case
          Value::SparseTensorType(_) => {panic!("SPARSE TENSOR TYPE NOT YET IMPLEMENTED \
                                                FOR NODE {}'S INPUT",
                                                input_name)} // TODO: check for this case
          _ => {panic!("UNRECOGNIZED TYPE FOR NODE {}'S INPUT", input_name)}
        };
      } else {
        panic!("UNABLE TO RETRIEVE TYPE OF NODE {}", input_name)
      }
      break;
    }
  }
  shape
}

/// This function retrieve the input tensors needed by a certain operation
/// # Arguments
/// * input_index: position of the input in the node's inputs
/// * node: node on which the operation has to be performed
/// * model_inputs: inputs of the onnx model
/// * model_initializers: initializers of the onnx model
/// # Returns
/// It returns a 5-tuple which contains the input tensor. Based on the position of the tuple
/// which is returned, a certain type of tensor is returned:
/// * (Array4<f32>,-,-,-,-)
/// * (-,Array3<f32>,-,-,-)
/// * (-,-,Array2<f32>,-,-)
/// * (-,-,-,Array1<f32>,-)
/// * (-,-,-,-,Array1<i64>)
pub fn get_stored_tensor(input_index: usize,
                                     node: &NodeProto,
                                     model_inputs: &Vec<ValueInfoProto>,
                                     model_initializers: &Vec<TensorProto>)
                                     -> (Option<Array4<f32>>,
                                         Option<Array3<f32>>,
                                         Option<Array2<f32>>,
                                         Option<Array1<f32>>,
                                         Option<Array1<i64>>){
  let shape = get_input_data_shape(model_inputs, &node.input[input_index]);

  let mut raw_data: Vec<f32> = vec![];
  let mut raw_data_i64: Vec<i64> = vec![];
  for init in model_initializers {
    if <String as AsRef<str>>::as_ref(&init.name) == &node.input[input_index] {
      if init.raw_data.len() > 0 {
        for chunk in <Vec<u8> as AsRef<[u8]>>::as_ref(&init.raw_data)
          .chunks(4) {
          let float32 = u8_to_f32(chunk);
          raw_data.push(float32);
        }
      } else if init.float_data.len() > 0 {
        for float_value in &init.float_data {
          raw_data.push(*float_value);
        }
      } else if init.int64_data.len() > 0 {
        for int_value in &init.int64_data {
          raw_data_i64.push(*int_value);
        }
      }
    }
  }

  if shape.len() == 4 {
    (Some(
      Array4::from_shape_vec(
        (*shape[0] as usize, *shape[1] as usize, *shape[2] as usize, *shape[3] as usize),
        raw_data).unwrap()),
     None,
     None,
     None,
     None)
  } else if shape.len() == 3 {
    (None,
     Some(Array3::from_shape_vec(
       (*shape[0] as usize, *shape[1] as usize, *shape[2] as usize),
       raw_data).unwrap()),
     None,
     None,
     None)
  } else if shape.len() == 2 {
    (None,
     None,
     Some(Array2::from_shape_vec(
       (*shape[0] as usize, *shape[1] as usize), raw_data).unwrap()),
     None,
     None)
  } else {
    if raw_data.len() > 0 {
      (None,
       None,
       None,
       Some(Array1::from_shape_vec(*shape[0] as usize, raw_data).unwrap()),
       None)
    }else {
      (None,
       None,
       None,
       None,
       Some(Array1::from_shape_vec(*shape[0] as usize, raw_data_i64).unwrap()))
    }
  }
}

/// This function converts u8 to f32
/// # Arguments
/// * bytes: number to convert as &[u8]
/// # Returns
//// the f32 converted number
pub fn u8_to_f32(bytes: &[u8]) -> f32 {
  assert_eq!(bytes.len(), 4);
  let mut array: [u8; 4] = Default::default();
  array.copy_from_slice(&bytes);
  f32::from_le_bytes(array)
}