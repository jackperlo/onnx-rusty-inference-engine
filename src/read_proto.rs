pub(crate) mod proto_structure;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::SplitWhitespace;
use proto_structure::*;
use crate::read_proto::proto_structure::ProtoAttribute;

/*
This function create a runtime structure which maps a .proto file content. This structure (i.e. a hashmap) will be used for parsing .onnx file.
- It takes one parameter: the .proto file path.
- It returns: a Result containing a hashmap or a string of error.
 */
pub fn create_struct_from_proto_file(proto_file_path: &str) -> Result<HashMap<String, Proto>, String> {
  //opening file in read mode
  let file = File::open(proto_file_path).expect("Failed to open and read file .proto");
  let reader = BufReader::new(file);

  //See proto_structure.rs -> Struct Proto explanation
  let mut proto_map: HashMap<String, Proto> = HashMap::new(); //this data structure maintains all the message/oneof structures contained in the .proto file (i.e. Proto).
  let mut current_proto_name = String::new(); //contains the path to the considered proto (always remember that a proto is a Message/OneOf)
  let mut _proto_version = 2; //since version 2 and 3 are valid, this application needs to work properly with both versions
  let mut proto_level = 0; //level of nesting
  let mut parent_type = KindOf::default(); //this variable maintains whether the reader is inside of a message or a oneof

  for cur_line in reader.lines() {
    let mut line = cur_line.expect("Failed to read line from .proto file");
    line = line.to_lowercase().trim_start().parse().unwrap(); //trimming line content from initial whitespaces

    //skipping commented lines or empty lines
    if line.starts_with("//") || line == "" {
      continue;
    }

    //skipping the enum version
    if line.starts_with("enum version"){
      proto_level += 1;
      continue;
    }

    //a line starts with this word when an enum/message/oneof closes itself.
    //The nesting level must be decreased and the path must be trimmed by the last level (if it has more than 1 level)
    //(e.g. current proto path: model/tensor/sparseTensor -> model/tensor)
    if line.starts_with("}") {
      proto_level -= 1;
      let proto_name_path: Vec<&str> = current_proto_name.split('/').collect();

      if proto_name_path.len() > 1 {
        current_proto_name = path_trimming(proto_name_path);
      }
    }

    //line starts with "syntax" word (not case sensitive)
    if line.starts_with("syntax") {
      //i.e. syntax = "proto3"; the 3 needs to be extracted and saved
      let trimmed_string = line.trim();
      let mut words = trimmed_string.split_whitespace();
      if let Some(version) = words.nth(2) {
        _proto_version = (&version[6..7]).parse().unwrap();
      } else {
        return Err("Cannot get the specified index of the trimmed line!".to_string());
      }
      continue;
    }

    //line starts with "message"/"oneof" word
    if line.starts_with("message") || line.starts_with("oneof") || line.starts_with("enum"){
      if line.starts_with("message") { //saving in which structure will be added the next read lines
        parent_type = KindOf::Message;
      } else if line.starts_with("oneof"){
        parent_type = KindOf::OneOf;
      } else{
        parent_type = KindOf::Enum;
      }

      proto_level += 1; //increasing nesting level
      let trimmed_string = line.trim();
      let mut words = trimmed_string.split_whitespace();
      if let Some(proto_name) = words.nth(1) { //retrieving the proto name
        if proto_level == 1 { //the proto is at the top level of nesting (e.g. message person{...})
          current_proto_name = String::from(proto_name);
        } else { //nested proto. Saving the path it's way more convenient for searching it, than scan all the hashmap nested levels
          // (i.e. message person{ message address {}}) -> person/address
          let mut aus_str = String::from("/");
          aus_str.push_str(proto_name);
          current_proto_name.push_str(&aus_str);
        }

        let proto_name_path: Vec<&str> = current_proto_name.split('/').collect();
        //println!("{:?}", proto_name_path);
        //add a new proto in the correct position(path) of hashmap
        match adding_new_proto(&proto_name_path, proto_name, &current_proto_name, &mut proto_map, &parent_type) {
          Ok(()) => continue,
          Err(err) => { return Err(err); }
        }
      }
    }

    //current line contains an attribute of a proto structure
    if line.starts_with("optional") || line.starts_with("repeated") || line.starts_with("required") {
      let words = line.split_whitespace();
      //the attribute is added
      match adding_proto_attribute(2, words, &current_proto_name, &mut proto_map) {
        Ok(()) => continue,
        Err(err) => { return Err(err); }
      };
    }

    //if the flow reached this point, the content is an attribute without annotation (proto3 style).
    if parent_type == KindOf::OneOf { //checking that the attribute is inside a oneof structure
      let words = line.split_whitespace();
      //adding the attribute to the oneof
      match adding_proto_attribute(3, words, &current_proto_name, &mut proto_map) {
        Ok(()) => continue,
        Err(err) => { return Err(err); }
      };
    }

    //if the flow reached this point, the content is an enum content.
    if parent_type == KindOf::Enum { //checking that the attribute is inside a enum structure
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

/*
This structure trims the path which is passed in input as a vector of string.
In particular is used when the bufreader meets "}" which means the closure of a message/oneof/enum.
- It takes one parameter as input: the path to be trimmed as a vector of string
- It returns a new path as a string
*/
fn path_trimming(proto_name_path: Vec<&str>) -> String {
  let mut i = 0;
  let mut trimmed_path = String::new();
  while i < proto_name_path.len() - 1 {
    if i == 0 {
      trimmed_path.push_str(proto_name_path[i]);
    } else {
      let mut aus_str = String::from("/");
      aus_str.push_str(proto_name_path[i]);
      trimmed_path.push_str(&aus_str);
    }
    i += 1;
  }
  trimmed_path
}

/*
This function allows to add a new Proto (either Message or OneOf) to the proto_map HashMap.
- It takes 5 parameters.
  ~ proto_name_path: this is a vector containing the path where to add the proto
  ~ proto_name: the name of the proto to be added
  ~ current_proto_name: if the proto would added at the 0 level (i.e. directly in the proto_map), this allows to directly search for it in advance (avoiding bad situations in which this name has already be added before)
  ~ proto_map: HashMap which contains the .proto content
  ~ proto_type: which kind of proto add (message or oneof)
- It returns () if the adding action has been successful, a string of error otherwise
 */
fn adding_new_proto(proto_name_path: &Vec<&str>, proto_name: &str, current_proto_name: &String, mut proto_map: &mut HashMap<String, Proto>, proto_type: &KindOf) -> Result<(), String> {
  return if proto_name_path.len() > 1 { //if the proto will be added at > 0 nesting level
    match get_correct_level_hashmap(&mut proto_map, &proto_name_path, 0) { //this searches and returns the parent to which the new proto will be added
      Ok(map) => {
        match map.get(proto_name_path[proto_name_path.len() - 1]) { //avoiding duplicates; i.e. 2 or more message/oneof with the same name
          Some(_) => { Err("Cannot insert duplicated values into HashMap.".to_string()) }
          None => {
            match proto_type { //adding the new proto (message/oneof/enum)
              KindOf::Message => map.insert(proto_name.to_string(), Proto::new(KindOf::Message)),
              KindOf::OneOf => map.insert(proto_name.to_string(), Proto::new(KindOf::OneOf)),
              KindOf::Enum => map.insert(proto_name.to_string(), Proto::new(KindOf::Enum))
            };
            Ok(())
          }
        }
      }
      Err(err) => { Err(err) }
    }
  } else { //the new proto is added at level 0 (directly inside proto_map hashmap)
    match proto_map.get(current_proto_name) {  //avoiding duplicates; i.e. 2 or more message/oneof with the same name
      Some(_) => { Err("Cannot insert duplicated values into HashMap.".to_string()) }
      None => { //adding the new proto (message/oneof)
        proto_map.insert(proto_name.to_string(), Proto::new(KindOf::Message));
        Ok(())
      }
    }
  };
}

/*
This function allows to add a new attribute to a proto (message/oneof).
- It takes 4 parameters.
  ~ proto_version: this specifies the proto version (2, or 3 which could omits the annotation)
  ~ words: the line just red from bufreader
  ~ current_proto_name: if the proto would added at the 0 level (i.e. directly in the proto_map), this allows to directly search for it in advance (avoiding bad situations in which this name has already be added before)
  ~ proto_map: HashMap which contains the .proto content
- It returns () if the adding action has been successful, a string of error otherwise
 */
fn adding_proto_attribute(proto_version: i32, mut words: SplitWhitespace, current_proto_name: &String, mut proto_map: &mut HashMap<String, Proto>) -> Result<(), String> {
  let mut annotation = ProtoAnnotation::default();
  if proto_version == 2{ //proto2 must specify an annotation
    annotation = words.next().expect("Cannot get Annotation in .proto2").parse().unwrap();
  }
  if let Some(attribute_type) = words.next() {
    if let Some(attribute_name_with_equals) = words.next() {
      let attribute_name: &str = attribute_name_with_equals.split('=').collect::<Vec<&str>>()[0];
      words.next();
      if let Some(tag) = words.next() {
        if let Ok(tag) = tag.split(';').collect::<Vec<&str>>()[0].parse::<i32>() {
          let proto_name_path: Vec<&str> = current_proto_name.split('/').collect();

          match search_proto_in_hashmap(&mut proto_map, &proto_name_path, 0) { //this function searches and returns the proto to which the attribute will be added
            Ok(proto) => {
              match proto.attributes.get(&tag) { //avoiding duplicates; i.e. 2 or more attributes with the same tag
                Some(_) => { return Err("Cannot insert duplicated values into HashMap.".to_string()); }
                None => { //adding the new attribute
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


/*
This function allows to add a new attribute to a proto of type Enum.
- It takes 4 parameters.
  ~ words: the line just red from bufreader
  ~ current_proto_name: if the proto would added at the 0 level (i.e. directly in the proto_map), this allows to directly search for it in advance (avoiding bad situations in which this name has already be added before)
  ~ proto_map: HashMap which contains the .proto content
- It returns () if the adding action has been successful, a string of error otherwise
 */
fn adding_enum_attribute(mut words: SplitWhitespace, current_proto_name: &String, mut proto_map: &mut HashMap<String, Proto>) -> Result<(), String> {
  if let Some(attribute_type) = words.next() {
    words.next();
    if let Some(tag) = words.next() {
      if let Ok(tag) = tag.split(';').collect::<Vec<&str>>()[0].parse::<i32>() {
        let proto_name_path: Vec<&str> = current_proto_name.split('/').collect();

        match search_proto_in_hashmap(&mut proto_map, &proto_name_path, 0) { //this function searches and returns the proto to which the attribute will be added
          Ok(proto) => {
            match proto.attributes.get(&tag) { //avoiding duplicates; i.e. 2 or more attributes with the same tag
              Some(_) => { return Err("Cannot insert duplicated values into HashMap.".to_string()); }
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

/*
This function searches and returns the proto parent to which add a new proto.
- It takes 3 parameters.
  ~ map: HashMap which contains the proto(s)
  ~ proto_name_path: this is the complete path(as a slice of strings) to which add the new proto.
  ~ index: this allows the recursive calls to shift(starting from 0) over the slice (proto_name_path) always getting the desired proto.
- It returns the contents HashMap of the parent to which will be added the new proto, a string of error otherwise
 */
fn get_correct_level_hashmap<'a, 'b>(map: &'a mut HashMap<String, Proto>, proto_name_path: &'b [&str], index: usize) -> Result<&'a mut HashMap<String, Proto>, String> {
  return match map.get_mut(proto_name_path[index]) { //e.g. in proto_map hashmap it searches for the key:  proto_name_path = [graph, node, tensor] -> proto_name_path[0] = graph.
    Some(proto) => {
      if index == proto_name_path.len() - 2 { //when the index is at last level of the path - 1. (-2 because it starts count from 0, furthermore the parent is the desired result)
        Ok(&mut proto.contents)
      } else {
        get_correct_level_hashmap(&mut proto.contents, proto_name_path, index + 1) //recursively shift over the slice increasing the index
      }
    }
    None => { return Err(format!("Cannot get hashmap content. Path: {:?}", proto_name_path)); }
  };
}

/*
This function searches and returns the proto to which add a new attribute.
- It takes 3 parameters.
  ~ map: HashMap which contains the proto(s)
  ~ proto_name_path: this is the complete path(as a slice of strings) to which add the new proto.
  ~ index: this allows the recursive calls to shift(starting from 0) over the slice (proto_name_path) always getting the desired proto.
- It returns the proto to which will be added the new attribute, a string of error otherwise
 */
fn search_proto_in_hashmap<'a, 'b>(map: &'a mut HashMap<String, Proto>, proto_name_path: &'b [&str], index: usize) -> Result<&'a mut Proto, String> {
  return match map.get_mut(proto_name_path[index]) {  //e.g. in proto_map hashmap it searches for the key:  proto_name_path = [graph, node, tensor] -> proto_name_path[0] = graph.
    Some(proto) => {
      if index < proto_name_path.len() - 1 { //when the index is at last level of the path
        search_proto_in_hashmap(&mut proto.contents, proto_name_path, index + 1) //recursively shift over the slice increasing the index
      } else {
        Ok(proto)
      }
    }
    None => { return Err(format!("Cannot find element in hashmap. Path: {:?}", proto_name_path)); }
  };
}

