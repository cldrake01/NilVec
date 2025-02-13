/// A simple metadata type.
#[derive(Debug, Clone, PartialEq)]
pub enum Metadata {
    Str(String),
    Int(i64),
    Float(f64),
}
