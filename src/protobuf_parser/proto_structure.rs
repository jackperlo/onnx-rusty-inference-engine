use core::fmt::Debug;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/*
This structure enumerates the possible kind of annotations an attribute of a Proto message could
assume accordingly to Protocol Buffers v2(proto2), v3(proto3) documentation:
(https://protobuf.dev/programming-guides).
  - Optional: means that the attribute could be not present in the Attribute structure assuming its
    default value. Note: in proto3 each attribute without explicit annotation its considered as
    marked optional by default.
  - Repeated: means that the attribute could be present [0..N] times
  - Required: means that the message struct cannot be considered well-formed if this attribute is
    not present; currently this annotation is strongly deprecated used but is maintained
    for backward compatibility
  - Map: means that a certain scalar value has been encoded as "packed"
    (this is done by default in proto3, while must be specified in proto2). e.g. Map<string, i32>
    shows an i32 value which is packed as a string encoding (with a certain LEN).
 */
#[repr(C)]
#[derive(Default, Debug, PartialEq, Clone)]
pub enum ProtoAnnotation{
  #[default]
  Optional,
  Implicit,
  Repeated,
  Map
}
impl FromStr for ProtoAnnotation {
  type Err = ParseProtoAnnotationError;
  // this allows to automatically parse() from a string (red from .proto file)
  // into a ProtoAnnotation Type
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "optional" => Ok(ProtoAnnotation::Optional),
      "implicit" => Ok(ProtoAnnotation::Implicit),
      "repeated" => Ok(ProtoAnnotation::Repeated),
      "map" => Ok(ProtoAnnotation::Map),
      "required" | _ => Err(ParseProtoAnnotationError(s.to_string()))
    }
  }
}

/*
This structure helps to print not supported string annotations
*/
#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub struct ParseProtoAnnotationError(String);
impl fmt::Display for ParseProtoAnnotationError {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Invalid ProtoAnnotation string: '{}'.\n\t\
      Supported are: [Optional, Repeated, Map].\n\
      Note: 'Required' is now strongly deprecated.\n\
      Update to new schema or change annotation type;\n\
      for further info: https://protobuf.dev/programming-guides/proto2/",
      self.0)
  }
}
impl std::error::Error for ParseProtoAnnotationError {}

/*
This structure contains an Attribute of a Message struct in a .proto file.
 (e.g. optional string name = 1;)
  - annotation: this annotation specifies a modifier for the attribute(i.e. optional).
    This is only present in proto2 version, while it could be omitted in proto3 version
  - attribute_name: the name of the attribute (i.e. name)
  - attribute_type: the type of the attribute (i.e. string)
  - tag: this is the number which identifies the attribute (i.e. 1)
 */
#[repr(C)]
#[derive(Default, Debug, Clone)]
pub struct ProtoAttribute {
  pub annotation: ProtoAnnotation,
  pub attribute_name: String,
  pub attribute_type: String
}
impl ProtoAttribute {
  pub fn new() -> Self {
    Self {
      annotation: Default::default(),
      attribute_name: Default::default(),
      attribute_type: Default::default()
    }
  }
}

/*
This enum allows a certain Proto structure to be distinguished between a "message"
or a "one of" or a "enum"
 */
#[repr(C)]
#[derive(Default, Debug, PartialEq, Clone)]
pub enum KindOf{
  #[default]
  Message,
  OneOf,
  Enum
}

/*
This enum allows to distinguish between the two supported version of Protofbuf
 */
#[repr(C)]
#[derive(Default, Debug, PartialEq, Clone)]
pub enum ProtoVersion{
  #[default]
  Proto2,
  Proto3
}
impl FromStr for ProtoVersion {
  type Err = ParseProtoVersionError;
  // this allows to automatically parse() from a string (red from .proto file)
  // into a ProtoVersion Type
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "2" => Ok(ProtoVersion::Proto2),
      "3" => Ok(ProtoVersion::Proto3),
      _ => Err(ParseProtoVersionError(s.to_string()))
    }
  }
}

/*
This structure helps to print not supported proto versions
*/
#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub struct ParseProtoVersionError(String);
impl fmt::Display for ParseProtoVersionError {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Invalid ProtoVersion syntax: '{}'.\n\t Supported are: [Proto2, Proto3].", self.0)
  }
}
impl std::error::Error for ParseProtoVersionError {}

/*
This structure contains a "message" structure or a "one of" structure contained in a .proto file.
Since this structure will be used while parsing .onnx files in order to understand its content,
all the message/one-of contained in the .proto file are stored in a HashMap
(which allows O(1) searches).
Specifically, let's make an example:
.proto file ->
          message Person {
            oneof Address {
              string city = 3;
              int32 number = 5;
            }
            optional string email = 1;
          };

runtime ->
          proto_map: HashMap<String, Proto> = [person, proto1];

            proto1: Proto = {
              kind_of: KindOf::Message,
              attributes: HashMap<i32, ProtoAttribute>[(1, protoAttribute1)],
              contents: HashMap<String, proto2>[(address, proto2)]
            };

              protoAttribute1: ProtoAttribute = {
                annotation: ProtoAnnotation::optional,
                attribute_name: "email",
                attribute_type: "string"
              };
              proto2: Proto = {
                kind_of: KindOf::OneOf,
                attributes: HashMap<i32, ProtoAttribute>[(3, protoAttribute2), (5, protoAttribute3)],
                contents: HashMap<String, proto2>[]
              };

                protoAttribute2: ProtoAttribute = {
                  annotation: ProtoAnnotation::default(),
                  attribute_name: "city",
                  attribute_type: "string"
                };
                protoAttribute3: ProtoAttribute = {
                  annotation: ProtoAnnotation::default(),
                  attribute_name: "number",
                  attribute_type: "int32"
                };

  - kind_of: represents the type of the structure(Message or OneOf or Enum).
  - attributes: this HashMap contains the list of attributes. Each attribute is represented by
    a ProtoAttribute. The HashMap allows to execute O(1) searches once having the Tag(i32)
    key to search.
  - contents: this HashMap allows to contain other "message"/"oneof" structures recursively,
    preserving the O(1) access time.
*/
#[repr(C)]
#[derive(Default, Clone)]
pub struct Proto {
  pub proto_version: ProtoVersion, //one value between [Proto2, Proto3]
  pub kind_of: KindOf, //one value between [Message, OneOf, Enum]
  pub attributes: HashMap<i32, ProtoAttribute>, //<tag, ProtoAttribute>
  pub contents: HashMap<String, Proto> //<name, Proto>, recursion supported
}
impl Proto{
  pub fn new(proto_version: ProtoVersion, kind_of: KindOf) -> Self {
    Self {
      proto_version,
      kind_of,
      attributes: HashMap::new(),
      contents: HashMap::new()
    }
  }
}
impl Debug for Proto{
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{{\n\t{:?} \n\t{:?} \n\tattributes: {:?} \n\tcontents: {:?}\n}}",
           self.proto_version,
           self.kind_of,
           self.attributes,
           self.contents)
  }
}

