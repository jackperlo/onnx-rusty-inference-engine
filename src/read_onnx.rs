use std::collections::HashMap;
use crate::onnx_structure::ModelProto;
use crate::read_proto::create_struct_from_proto_file;
use crate::read_proto::proto_structure::{KindOf, Proto};

/*
This function allows the program to read a .onnx file byte per byte and, thanks to the proto_structure previously read, to generate a onnx runtime model.
  - It takes two parameters:
    ~ onnx_file_path, specifies the onnx file path
    ~ proto_file_path, specifies the proto file path
  - It returns: ModelProto, a runtime model proto
*/
pub fn generate_onnx_model(onnx_file_path: &str, proto_file_path: &str) -> ModelProto {
  let proto_structure = match create_struct_from_proto_file(proto_file_path) {
    Ok(proto) => proto,
    Err(err) => panic!("{}", err)
  };

  let onnx_bytes = std::fs::read(onnx_file_path).expect("Failed to read file");
  let mut counter = 0; //counter of read bytes

  //onnx fields
  let mut wire_type: String;
  let mut field_number: i32;
  let mut field_name: String;
  let mut field_type: String;
  let mut number_of_concatenated_bytes: i32; //this variable counts the number of concatenated bytes (two bytes are concatenated if the first has msb equal to 1)
  let mut value: i32;
  let mut length_object_or_enum_field_numer: i32;
  let mut last_data_type = 0;

  //these lifo stacks contain respectively the structure read from the onnx and its length. E.g. ModelProto -> length 1024 bytes, GraphProto -> length 1000 bytes
  let mut lifo_stack_length: Vec<i32> = Vec::new();
  lifo_stack_length.push(onnx_bytes.len() as i32);
  let mut lifo_stack_struct: Vec<String> = Vec::new();
  lifo_stack_struct.push("modelproto".to_string());
  let mut lifo_stack_named_struct: Vec<String> = Vec::new();
  lifo_stack_named_struct.push("model".to_string());

  let mut model_proto: ModelProto = ModelProto::new();
  model_proto.special_fields.cached_size().set(onnx_bytes.len() as  u32);

  while counter < onnx_bytes.len() {
    //converts from byte base[10] to binary
    let mut binary_string = format!("{:b}", onnx_bytes[counter]);
    number_of_concatenated_bytes = 0;

    //it means that the binary number has msb equal to 1. Dependant information contained between the following bytes
    if binary_string.len() >= 8 {
      binary_string = concat_bytes(binary_string, &mut counter, &onnx_bytes, &mut number_of_concatenated_bytes)
    }

    //since the binary strings could have a length [0, 8] it's necessary to know the position of the last 3 bits (which identify the wire type)
    let partition_index = binary_string.len().saturating_sub(3);
    let (first_part, last_three_digits) = binary_string.split_at(partition_index);
    //getting the the last three bits (the wire type) and the other bits (the field number)
    wire_type = get_wire_type(last_three_digits);
    field_number = u64::from_str_radix(first_part, 2).unwrap() as i32; //dropping msb 0s

    //retrieving the field type and name merging the field number (extracted from the .onnx) with the current structure and the info provided from the proto_structure
    match get_field(&lifo_stack_struct.last().unwrap(), field_number, &proto_structure) {
      Some((f_n, f_t)) => {
        field_name = f_n;
        field_type = f_t;
      }
      None => panic!("ONNX SYNTAX ERROR")
    }

    if !is_simple_type(&field_type) {
      //since the field has a complex type(another Message, OneOf, Enum), such as a NodeProto, its length needs to be retrieved
      counter += 1; //push forward the onnx_bytes counter to read the structure length
      let mut length_binary_or_enum_filed_number = format!("{:b}", onnx_bytes[counter]);
      if length_binary_or_enum_filed_number.len() >= 8 {
        length_binary_or_enum_filed_number = concat_bytes(length_binary_or_enum_filed_number, &mut counter, &onnx_bytes, &mut number_of_concatenated_bytes);
      }

      /*
      the possibilities at this point are two: 1. the complex type is a Message/OneOf. 2. the complex type is a Enum.
      if the type represents a enum, the just read bytes are not its length (since the enum doesn't represent real type but enumerations), they in fact represent
      the number to search for in the enum.
      e.g. 7 -> Enum{ 1=int, 3=double, 7=float}. We search for 7 obtaining the type Float.
      */
      length_object_or_enum_field_numer = u64::from_str_radix(&*length_binary_or_enum_filed_number, 2).unwrap() as i32; //dropping msb 0s
      //so if the current field_type which represents the enum name exists into the proto_structure, its type is retrieved. Otherwise we now know that the field_type is not a enum (an empty string is returned)
      let is_enum_with_type = search_enum_in_proto_structure(&proto_structure, &field_type, length_object_or_enum_field_numer);
      if is_enum_with_type.is_empty() { //new structure read from the onnx. Specifically a Message not a Enum
        //decreasing the length of the current onnx structure by 1 Byte (wire type) + 1 Byte (field number) + x Bytes for the concatenated bytes needed to specify the structure length in the .onnx
        decrement_length(&mut lifo_stack_length, &mut lifo_stack_struct, &mut lifo_stack_named_struct, 2 + number_of_concatenated_bytes);

        lifo_stack_struct.push(field_type.clone());
        lifo_stack_length.push(length_object_or_enum_field_numer);
        lifo_stack_named_struct.push(field_name.clone());

        model_proto.dispatch(&lifo_stack_named_struct, &lifo_stack_struct[1..], field_name.as_str(), length_object_or_enum_field_numer, 0.00, String::default(),true);

        //println!("Adding Sub-Message: ({}) In {}/{} -> {}, {} ({})", field_number, lifo_stack_struct.get(lifo_stack_struct.len() - 2).unwrap(), lifo_stack_struct.last().unwrap(), field_name, length_object_or_enum_field_numer, wire_type);
      } else { //enum case. As explained above it doesn't need to be added to the runtime structures
        let mut aus_struct = lifo_stack_struct.clone();
        let mut aus_named_struct = lifo_stack_named_struct.clone();
        aus_struct.push(field_type.clone());
        aus_named_struct.push(field_name.clone());

        model_proto.dispatch(&aus_named_struct, &aus_struct[1..], field_name.as_str(), length_object_or_enum_field_numer, 0.00, String::default(),false);

        //decreasing the length of the current onnx structure by 1 Byte (wire type) + 1 Byte (field number) + x Bytes for the concatenated bytes needed to specify the structure length in the .onnx
        decrement_length(&mut lifo_stack_length, &mut lifo_stack_struct, &mut lifo_stack_named_struct, 2 + number_of_concatenated_bytes);

        //println!("Enum Case: ({}) In {}/{} -> {} = {} ({})", field_number, lifo_stack_struct.get(lifo_stack_struct.len() - 2).unwrap(), lifo_stack_struct.last().unwrap(), field_name, is_enum_with_type, wire_type);
      }
    } else if wire_type == "LEN" { //simple type cases
      //getting the length of a string or some raw_data
      counter += 1;
      binary_string = format!("{:b}", onnx_bytes[counter]);
      if binary_string.len() >= 8 {
        binary_string = concat_bytes(binary_string, &mut counter, &onnx_bytes, &mut number_of_concatenated_bytes)
      }
      value = u64::from_str_radix(&*binary_string, 2).unwrap() as i32;

      let mut string_result = String::new();

      if field_name == "int64_data"{
        let mut j = 0;
        let mut aus: i64;
        let mut phantom_counter = counter+1;
        let mut phantom_number_of_concatenated_bytes = number_of_concatenated_bytes;
        let mut old_number_of_concatenated_bytes = number_of_concatenated_bytes;
        //println!("{} {} {} {}", onnx_bytes[phantom_counter], onnx_bytes[phantom_counter+1], onnx_bytes[phantom_counter+2], onnx_bytes[phantom_counter+3]);
        while j < value {
          //println!("{}", onnx_bytes[phantom_counter]);
          binary_string = format!("{:b}", onnx_bytes[phantom_counter]);
          if binary_string.len() >= 8 {
            binary_string = concat_bytes(binary_string, &mut phantom_counter, &onnx_bytes, &mut phantom_number_of_concatenated_bytes)
          }
          aus = i64::from_str_radix(&*binary_string, 2).unwrap();
          string_result.push_str(&aus.to_string());
          string_result.push_str(", ");
          j+=(phantom_number_of_concatenated_bytes-old_number_of_concatenated_bytes)+1;
          old_number_of_concatenated_bytes = phantom_number_of_concatenated_bytes;
          phantom_counter+=1;
        }
      } else if field_name == "raw_data" || field_name == "float_data"{ //raw_data store the input tensors and initializers data
        let mut i = 1;
        while i <= value { //getting all the raw_data (each data is a f32/i64 so needs 4/8 bytes to be completely read)
          match last_data_type {
            1 => {
              string_result.push_str(&f32::from_le_bytes([onnx_bytes[counter + i as usize], onnx_bytes[counter + (i as usize + 1usize)], onnx_bytes[counter + (i as usize + 2usize)], onnx_bytes[counter + (i as usize + 3usize)]]).to_string());
              i = i + 4;
            },
            7 => {
              string_result.push_str(&i64::from_le_bytes([onnx_bytes[counter + i as usize], onnx_bytes[counter + (i as usize + 1usize)], onnx_bytes[counter + (i as usize + 2usize)], onnx_bytes[counter + (i as usize + 3usize)], onnx_bytes[counter + (i as usize + 4usize)], onnx_bytes[counter + (i as usize + 5usize)], onnx_bytes[counter + (i as usize + 6usize)], onnx_bytes[counter + (i as usize + 7usize)]]).to_string());
              i = i + 8;
            },
            _ => panic!("Data Type Not Managed into read_onnx->raw_data/float_data: {}", last_data_type)
          }
          string_result.push_str(", ");
        }
      } else { //reading a string, parsing bytes to ascii
        for i in 1..=value {
          match binary_string_to_ascii(format!("{:b}", onnx_bytes[counter + i as usize])) {
            Some(ascii_char) => string_result.push(ascii_char),
            None => println!("Parsing bytes to ASCII format failed"),
          }
        }
      }

      model_proto.dispatch(&lifo_stack_named_struct, &lifo_stack_struct[1..], field_name.as_str(), 0, 0.00, string_result.clone(), false);

      //println!("String/Raw Data Case: ({}) In {} => {} = {} ({})", field_number, lifo_stack_struct.last().unwrap(), field_name, string_result, wire_type);

      //decreasing the length of the current onnx structure by 1 Byte (wire type) + 1 Byte (field number) + x Bytes for the concatenated bytes needed to specify the structure length in the .onnx + y Bytes (value) for the string/raw data just read
      decrement_length(&mut lifo_stack_length, &mut lifo_stack_struct, &mut lifo_stack_named_struct, value + 2 + number_of_concatenated_bytes);
      counter += value as usize;
    } else {
      if field_type == "float" { //parsing float type numbers
        let mut concat_part: String = String::new();
        counter += 4;
        for n_byte in 0..4 {
          let formatted = format!("{:02X}", onnx_bytes[counter - n_byte]);
          concat_part = format!("{}{}", concat_part, formatted);
        }
        //converting Hex to u32
        let int_value = u32::from_str_radix(&*concat_part, 16).unwrap();
        //converting u32 to array of u8
        let bytes: [u8; 4] = int_value.to_le_bytes();
        //transmuting the array into a float
        let float_value: f32 = unsafe { std::mem::transmute(bytes) };

        model_proto.dispatch(&lifo_stack_named_struct, &lifo_stack_struct[1..], field_name.as_str(), 0, float_value, String::default(), false);

        //println!("Float Case: ({}) In {} => {} = {} ({})", field_number, lifo_stack_struct.last().unwrap(), field_name, float_value, field_type);

        //decreasing the length of the current onnx structure by 1 Byte (wire type) + 1 Byte (field number) + x Bytes for the concatenated bytes needed to specify the structure length in the .onnx + 4 Bytes (value) for the float representation
        decrement_length(&mut lifo_stack_length, &mut lifo_stack_struct, &mut lifo_stack_named_struct, 5 + number_of_concatenated_bytes);
      } else { //all the other cases
        counter += 1;
        binary_string = format!("{:b}", onnx_bytes[counter]);
        if binary_string.len() >= 8 {
          binary_string = concat_bytes(binary_string, &mut counter, &onnx_bytes, &mut number_of_concatenated_bytes)
        }
        value = u64::from_str_radix(&*binary_string, 2).unwrap() as i32;
        if field_name == "data_type"{
          last_data_type = value;
        }

        model_proto.dispatch(&lifo_stack_named_struct, &lifo_stack_struct[1..], field_name.as_str(), value, 0.00, String::default(), false);

        //println!("Other cases: ({}) In {} => {} = {} ({})", field_number, lifo_stack_struct.last().unwrap(), field_name, value, wire_type);

        decrement_length(&mut lifo_stack_length, &mut lifo_stack_struct, &mut lifo_stack_named_struct, 2 + number_of_concatenated_bytes);
      }
    }
    counter += 1;
  }
  model_proto
}

/*
This function allows to concatenate all the bytes read from the .onnx files (they can be distinguished because they have the msb bit equal to 1).
  -It takes 4 parameters:
    ~ start_string: contains the first byte just read
    ~ counter: this counts the already read bytes into the onnx_bytes
    ~ onnx_bytes: this is the vector containing all the bytes read from the onnx
    ~ number_bytes: number of concatenated bytes
  -It returns a string with all the bits concatenated following the protobuf standard
*/
fn concat_bytes(start_string: String, counter: &mut usize, onnx_bytes: &Vec<u8>, number_bytes: &mut i32) -> String {
  let mut count_parts = 0;
  let mut concat_part: String = String::new();

  let mut binary_string = start_string.clone();
  //following byte is related to previous byte
  while binary_string.len() >= 8 {
    count_parts += 1;
    *counter += 1;
    *number_bytes += 1;
    binary_string = format!("{:b}", onnx_bytes[*counter]);
  }
  count_parts += 1;

  /*the following code drops the MSB, concatenates in Little Endian the bytes and drops the exceeding msb zeros*/
  for i in 0..count_parts {
    let mut part = format!("{:b}", onnx_bytes[*counter + 1 - (count_parts - i)]);
    //drops of the first bit which value is 1 (except for the last byte to concatenate, which value is 0)
    if i < count_parts - 1 {
      part = format!("{}", &part[1..]);
    }

    //little endian concatenation of the inner bytes
    if i != 0 {
      concat_part = format!("{}{}", part, concat_part);
    } else {
      concat_part = part;
    }
    binary_string = concat_part.clone();
  }
  if binary_string.len() % 8 != 0 { //if the resulting bytes are not multiple of 8 (8bit=1byte), then padding 0s are added at the head of the string
    let mut padding_zeros = (binary_string.len() as f64 / 8f64).ceil(); //round to upper integer
    padding_zeros = (padding_zeros * 8.0) - binary_string.len() as f64;
    for _j in 0..padding_zeros as usize {
      binary_string = format!("0{}", binary_string);
    }
  }

  binary_string
}

/*
This function allows to decrement the amount of bytes which remain to be read for the current structure.
  -It takes 3 parameters:
    ~ vec_length: the vector which contains the length of the structures
    ~ vec_struct: the vector which contains the structures and their hierarchy read from the onnx (i.e. modelproto->graphproto->nodeproto)
    ~ vec__named_struct: the vector which contains the structures and their hierarchy read from the onnx (i.e. model->graph->input)
    ~ value_to_decrement: how much the length needs to be decreased
*/
fn decrement_length(vec_length: &mut Vec<i32>, vec_struct: &mut Vec<String>, vec_named_struct: &mut Vec<String>, value_to_decrement: i32) {
  if vec_length.len() > 0 {
    for num in vec_length.iter_mut() { //decrease the correct structure length
      *num -= value_to_decrement;
    }

    let mut zero_count = 0;
    for &num in vec_length.iter().rev() { //check for 0 length structures
      if num != 0 {
        break;
      }
      zero_count += 1;
    }

    if zero_count > 0 { //truncating the 0 length structures
      vec_length.truncate(vec_length.len() - zero_count);
      vec_struct.truncate(vec_struct.len() - zero_count);
      vec_named_struct.truncate(vec_named_struct.len() - zero_count);
    }
  }
}

/*
This function check is a string is of a simple type or not.
  - It takes one parameter:
    ~ value_type: a string representing the type to check
  - It return true if the string is simple type, false otherwise
*/
fn is_simple_type(value_type: &String) -> bool {
  ["string", "int64", "float", "bytes", "int32"].iter().any(|&s| s == value_type)
}

/*
This function converts a binary string into ascii format.
  - It takes one parameter, the binary string to parse into ascii
  - It returns a option containing a ascii char
*/
fn binary_string_to_ascii(binary_string: String) -> Option<char> {
  if let Ok(binary_num) = u8::from_str_radix(&binary_string, 2) {
    if let Some(ascii_char) = char::from_u32(binary_num as u32) {
      return Some(ascii_char);
    }
  }
  None
}

/*
This function search for the field number in the current structure inside the relative proto_structure.
  - It takes 3 parameters:
    ~ current_struct: represents the struct in which the reading is, and so in which structure of the proto_structure go searching
    ~ field_number: the number to search inside the proto_structure
    ~ proto_structure: the proto structure retrieved before
  - It returns a tuple containing the field type, field name if the search worked as expected
*/
fn get_field(current_struct: &String, field_number: i32, proto_structure: &HashMap<String, Proto>) -> Option<(String, String)> {
  for el in proto_structure {
    if el.0 == current_struct {
      return match el.1.attributes.get(&field_number) {
        Some(ap) => Some((ap.attribute_name.clone(), ap.attribute_type.clone())),
        None => {
          let mut found_one_of = false;
          let mut ret_value = None;
          for inner_el in &el.1.contents {
            match inner_el.1.kind_of {
              KindOf::Message => continue,
              KindOf::Enum => continue,
              KindOf::OneOf => {
                match inner_el.1.attributes.get(&field_number) {
                  Some(ap) => {
                    found_one_of = true;
                    ret_value = Some((ap.attribute_name.clone(), ap.attribute_type.clone()));
                    break;
                  }
                  None => continue
                }
              }
            }
          }
          if found_one_of {
            ret_value
          } else {
            None
          }
        }
      };
    } else if el.1.contents.len() != 0 {
      match get_field(current_struct, field_number, &el.1.contents) {
        None => continue,
        Some(found) => return Some(found)
      }
    }
  }

  None
}

/*
This function converts the binary string passed as parameter into the corresponding wire type, as defined by protobuf standard.
  - It takes one parameter containing the binary number
  - It returns a string of the corresponding wire type
*/
fn get_wire_type(binary_number: &str) -> String {
  let decimal_number = u64::from_str_radix(binary_number, 2).unwrap();
  match decimal_number {
    0 => "VARINT".to_string(),
    1 => "I64".to_string(),
    2 => "LEN".to_string(),
    3 => "SGROUP".to_string(),
    4 => "EGROUP".to_string(),
    5 => "I32".to_string(),
    _ => "not found".to_string()
  }
}

/*
This function searches recursively inside the proto structure for a certain enum name. If the enum is found, the attribute with value equal to tag_value is returned
  - It takes 3 parameters:
    ~ map: the hashmap in which to search
    ~ enum_name: the enum name to search for
    ~ tag_value: the value to search for in the enum (if found)
  - It returns a string containing the value inside the searched enum.
    e.g. enum_name=my_enum, tag_value=5. Enum my_enum{int=1, double=3, float=5} -> the function returns float
*/
fn search_enum_in_proto_structure(map: &HashMap<String, Proto>, enum_name: &String, tag_value: i32) -> String {
  if map.is_empty() {
    return String::default();
  }
  return match map.get(enum_name) {
    Some(proto) => {
      match proto.kind_of {
        KindOf::Enum => {
          return String::from(&proto.attributes.get(&tag_value).unwrap().attribute_type);
        }
        _ => String::default()
      }
    }
    None => {
      let mut ret_value = String::default();
      for (_el_name, el_content) in map {
        ret_value.push_str(&search_enum_in_proto_structure(&el_content.contents, enum_name, tag_value));
      }
      return ret_value;
    }
  };
}