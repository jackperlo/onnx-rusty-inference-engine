use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::SplitWhitespace;

use crate::protobuf_parser::proto_structure::{Proto,
                                              KindOf,
                                              ProtoAnnotation,
                                              ProtoAttribute,
                                              ProtoVersion};
use crate::protobuf_parser::utils::{path_trimming,
                                    get_correct_level_hashmap,
                                    search_proto_in_hashmap};

/// This function create a runtime structure which maps a .proto file content.
/// This structure (i.e. a hashmap) will be then used for parsing .onnx file.
/// # Arguments
/// * `proto_file_path`: the .proto file path
/// # Returns
/// * It returns a Result containing a hashmap representing the .proto file, or a string of error.
pub fn create_struct_from_proto_file(proto_file_path: &str)
  -> Result<HashMap<String, Proto>, String> {
  // opening file in read mode
  let file = File::open(proto_file_path).expect("Failed to open and read file .proto");
  let reader = BufReader::new(file);

  /*See proto_structure.rs -> Struct Proto explanation*/
  // this data structure maintains all the message/oneof structures
  // contained in the .proto file (i.e. Proto).
  let mut proto_map: HashMap<String, Proto> = HashMap::new();
  // contains the path to the considered proto (always remember that a proto is a Message/OneOf)
  let mut current_proto_name = String::new();
  // since version 2 and 3 are valid, this application needs to work properly with both versions
  let mut proto_version: ProtoVersion = Default::default();
  let mut proto_level = 0; //level of nesting
  // this variable maintains whether the reader is inside of a message or a oneof
  let mut parent_type = KindOf::default();

  for cur_line in reader.lines() {
    let mut line = cur_line.expect("Failed to read line from .proto file");
    // trimming line content from initial whitespaces
    line = line.to_lowercase().trim_start().parse().unwrap();

    // skipping commented lines or empty lines
    if line.starts_with("//") || line == "" {
      continue;
    }

    // skipping the enum version
    if line.starts_with("enum version"){
      proto_level += 1;
      continue;
    }

    // a line starts with this word when an enum/message/oneof closes itself.
    // The nesting level must be decreased and the path must be
    // trimmed by the last level (if it has more than 1 level)
    // (e.g. current proto path: model/tensor/sparseTensor -> model/tensor)
    if line.starts_with("}") {
      proto_level -= 1;
      let proto_name_path: Vec<&str> = current_proto_name.split('/').collect();

      if proto_name_path.len() > 1 {
        current_proto_name = path_trimming(proto_name_path);
      }
    }

    // line starts with "syntax" word (not case sensitive)
    if line.starts_with("syntax") {
      // i.e. syntax = "proto3"; the 3 needs to be extracted and saved
      let trimmed_string = line.trim();
      let mut words = trimmed_string.split_whitespace();
      if let Some(version) = words.nth(2) {
        let _proto_version: i32 = (&version[6..7]).parse().unwrap();
        proto_version = match (&version[6..7]).parse(){
          Ok(proto_version) => proto_version,
          Err(err) => { return Err(err.to_string()); }
        };
      } else {
        return Err("Cannot get the specified index of the trimmed line!".to_string());
      }
      continue;
    }

    // line starts with "message"/"oneof"/"enum" word
    if line.starts_with("message") ||
       line.starts_with("oneof") ||
       line.starts_with("enum"){
      // saving in which structure will be added the next read lines
      if line.starts_with("message") {
        parent_type = KindOf::Message;
      } else if line.starts_with("oneof"){
        parent_type = KindOf::OneOf;
      } else{
        parent_type = KindOf::Enum;
      }

      proto_level += 1; // increasing nesting level
      let trimmed_string = line.trim();
      let mut words = trimmed_string.split_whitespace();
      if let Some(proto_name) = words.nth(1) { // retrieving the proto name
        if proto_level == 1 { // the proto is at the top level of nesting (e.g. message person{...})
          current_proto_name = String::from(proto_name);
        } else {
          // nested proto. Saving the path it's way more convenient for searching it,
          // than scan all the hashmap nested levels
          // (i.e. message person{ message address {}}) -> /person/address
          let mut aus_str = String::from("/");
          aus_str.push_str(proto_name);
          current_proto_name.push_str(&aus_str);
        }

        let proto_name_path: Vec<&str> = current_proto_name.split('/').collect();
        // add a new proto in the correct position(path) of hashmap
        match adding_new_proto(&proto_name_path,
                               proto_name,
                               &current_proto_name,
                               &proto_version,
                               &mut proto_map,
                               &parent_type) {
          Ok(()) => continue,
          Err(err) => { return Err(err); }
        };
      }
    }

    // TODO: gestire il caso map e implicit
    // current line contains an attribute of a proto structure
    if line.starts_with("optional") ||
       line.starts_with("repeated") ||
       line.starts_with("required") ||
       line.starts_with("map") ||
       line.starts_with("implicit")
      {
      let words = line.split_whitespace();
      // the attribute is added
      match adding_proto_attribute(&proto_version, words, &current_proto_name, &mut proto_map) {
        Ok(()) => continue,
        Err(err) => { return Err(err); }
      };
    }

    // if the flow reached this point, the content is an attribute without annotation (proto3 ver.)
    if parent_type == KindOf::OneOf { // checking that the attribute is inside a oneof structure
      let words = line.split_whitespace();
      // adding the attribute to the oneof
      assert_eq!(proto_version, ProtoVersion::Proto3);
      match adding_proto_attribute(&proto_version, words, &current_proto_name, &mut proto_map) {
        Ok(()) => continue,
        Err(err) => { return Err(err); }
      };
    }

    // if the flow reached this point, the content is an enum content.
    if parent_type == KindOf::Enum { // checking that the attribute is inside a enum structure
      let words = line.split_whitespace();
      //adding the attribute to the enum
      match adding_enum_attribute(words, &current_proto_name, &mut proto_map) {
        Ok(()) => continue,
        Err(err) => { return Err(err); }
      };
    }
  }
  Ok(proto_map)
}

/// This function allows to add a new Proto structure to the proto_map HashMap.
/// # Arguments
/// * `proto_name_path`: this is a vector containing the path where to add the proto
/// * `proto_name`: the name of the proto to be added
/// * `current_proto_name`: if the proto would added at the 0 level
///   (i.e. directly in the proto_map), this allows to directly search for it in advance
///   (avoiding awkward situations in which this name has already been added before)
/// * `proto_map`: HashMap which contains the .proto content
/// * `proto_type`: which kind of proto add
/// # Returns
/// It returns () if the adding action has been successful, a string of error otherwise
fn adding_new_proto(proto_name_path: &Vec<&str>,
                    proto_name: &str,
                    current_proto_name: &String,
                    proto_version: &ProtoVersion,
                    mut proto_map: &mut HashMap<String, Proto>,
                    proto_type: &KindOf)
  -> Result<(), String> {
  // if the proto will be added at > 0 nesting level
  return if proto_name_path.len() > 1 {
    // this searches and returns the parent to which the new proto structure must be added
    match get_correct_level_hashmap(&mut proto_map, &proto_name_path, 0) {
      Ok(map) => {
        // check to avoid duplicates; i.e. 2 or more message/oneof/whatever.. with the same name
        match map.get(proto_name_path[proto_name_path.len() - 1]) {
          Some(_) => { Err("Cannot insert duplicated values into HashMap.".to_string()) }
          None => {
            match proto_version{
              ProtoVersion::Proto2 => {
                match proto_type { // adding the new proto structure (e.g. message, oneof, ...)
                  KindOf::Message => map.insert(proto_name.to_string(),
                                                Proto::new(ProtoVersion::Proto2,
                                                           KindOf::Message)),
                  KindOf::OneOf => map.insert(proto_name.to_string(),
                                              Proto::new(ProtoVersion::Proto2,
                                                         KindOf::OneOf)),
                  KindOf::Enum => map.insert(proto_name.to_string(),
                                             Proto::new(ProtoVersion::Proto2,
                                                        KindOf::Enum))
                };
              },
              ProtoVersion::Proto3 => {
                match proto_type { // adding the new proto structure (e.g. message, oneof, ...)
                  KindOf::Message => map.insert(proto_name.to_string(),
                                                Proto::new(ProtoVersion::Proto3,
                                                           KindOf::Message)),
                  KindOf::OneOf => map.insert(proto_name.to_string(),
                                              Proto::new(ProtoVersion::Proto3,
                                                         KindOf::OneOf)),
                  KindOf::Enum => map.insert(proto_name.to_string(),
                                             Proto::new(ProtoVersion::Proto3,
                                                        KindOf::Enum))
                };
              }
            };
            Ok(())
          }
        }
      }
      Err(err) => { Err(err) }
    }
  } else {
    // the new proto is added at level 0 (directly inside proto_map hashmap)
    match proto_map.get(current_proto_name) {
      // check to avoid duplicates; i.e. 2 or more message/oneof/whatever.. with the same name
      Some(_) => { Err("Cannot insert duplicated values into HashMap.".to_string()) }
      None => { // adding the new proto structure
        match proto_version {
          ProtoVersion::Proto2 =>  proto_map.insert(proto_name.to_string(),
                                                    Proto::new(ProtoVersion::Proto2,
                                                               KindOf::Message)),
          ProtoVersion::Proto3 =>  proto_map.insert(proto_name.to_string(),
                                                    Proto::new(ProtoVersion::Proto3,
                                                               KindOf::Message)),
        };
        Ok(())
      }
    }
  };
}


// TODO: adattare questo a proto3
/// This function allows to add a new attribute to a proto (message/oneof).
/// # Arguments
/// * `proto_version`: this specifies the proto version (2, or 3 which could omits the annotations)
/// * `words`: the line just red from bufreader
/// * `current_proto_name`: if the proto would be added at the 0 level
///   (i.e. directly in the proto_map), this allows to directly search for it in advance
///   (avoiding awkward situations in which this name has already be added before)
/// * `proto_map`: HashMap which contains the .proto content
/// # Returns
/// It returns () if the adding action has been successful, a string of error otherwise
fn adding_proto_attribute(proto_version: &ProtoVersion,
                          mut words: SplitWhitespace,
                          current_proto_name: &String,
                          mut proto_map: &mut HashMap<String, Proto>) -> Result<(), String> {
  let mut annotation = ProtoAnnotation::default();
  if *proto_version == ProtoVersion::Proto2{ // proto2 must specify an annotation, throw an exception
    annotation = words.next().expect("Cannot get Annotation in .proto2").parse().unwrap();
  }
  if let Some(attribute_type) = words.next() {
    if let Some(attribute_name_with_equals) = words.next() {
      let attribute_name: &str = attribute_name_with_equals
        .split('=')
        .collect::<Vec<&str>>()[0];
      words.next();
      if let Some(tag) = words.next() {
        if let Ok(tag) = tag.split(';').collect::<Vec<&str>>()[0].parse::<i32>() {
          let proto_name_path: Vec<&str> = current_proto_name.split('/').collect();

          // this function searches and returns the proto to which the attribute will be added
          match search_proto_in_hashmap(&mut proto_map, &proto_name_path, 0) {
            Ok(proto) => {
              match proto.attributes.get(&tag) {
                // avoiding duplicates; i.e. 2 or more attributes with the same tag
                Some(_) => { return Err("Cannot insert duplicated msg attributes.".to_string()); }
                None => { // adding the new attribute
                  let mut attribute = ProtoAttribute::new();
                  attribute.annotation = annotation;
                  attribute.attribute_type = attribute_type.parse().unwrap();
                  attribute.attribute_name = attribute_name.parse().unwrap();
                  proto.attributes.insert(tag, attribute);
                }
              };
            }
            Err(err) => { return Err(err); }
          };
        } else {
          return Err("Cannot get TAG from .proto file".to_string());
        }
      }
    }
  }
  Ok(())
}

/// This function allows to add a new attribute to a proto of type Enum.
/// # Arguments
/// * `words`: the line just red from bufreader
/// * `current_proto_name`: if the proto would added at the 0 level
///   (i.e. directly in the proto_map), this allows to directly search for it in advance
///   (avoiding awkward situations in which this name has already be added before)
/// * `proto_map`: HashMap which contains the .proto content
/// # Returns
/// It returns () if the adding action has been successful, a string of error otherwise
fn adding_enum_attribute(mut words: SplitWhitespace,
                         current_proto_name: &String,
                         mut proto_map: &mut HashMap<String, Proto>)
  -> Result<(), String> {
  if let Some(attribute_type) = words.next() {
    words.next();
    if let Some(tag) = words.next() {
      if let Ok(tag) = tag.split(';').collect::<Vec<&str>>()[0].parse::<i32>() {
        let proto_name_path: Vec<&str> = current_proto_name.split('/').collect();

        // this function searches and returns the proto to which the attribute will be added
        match search_proto_in_hashmap(&mut proto_map, &proto_name_path, 0) {
          Ok(proto) => {
            match proto.attributes.get(&tag) {
              // avoiding duplicates; i.e. 2 or more attributes with the same tag
              Some(_) => { return Err("Cannot insert duplicated enum attributes.".to_string()); }
              None => { //adding the new attribute
                let mut attribute = ProtoAttribute::new();
                attribute.annotation = ProtoAnnotation::default();
                attribute.attribute_type = attribute_type.parse().unwrap();
                attribute.attribute_name = String::default();
                proto.attributes.insert(tag, attribute);
              }
            };
          }
          Err(err) => { return Err(err); }
        };
      } else {
        return Err("Cannot get TAG from .proto file".to_string());
      }
    }
  }
  Ok(())
}

