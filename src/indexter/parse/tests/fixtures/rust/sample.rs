use std::fmt;
use crate::helpers::assist;

pub struct Foo {
    pub value: i32,
}

impl Foo {
    pub fn new(value: i32) -> Foo {
        Foo { value }
    }
}

impl fmt::Display for Foo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Foo({})", self.value)
    }
}

impl fmt::Debug for Foo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Foo {{ value: {} }}", self.value)
    }
}

pub trait Greet {
    fn greet(&self) -> String;
}

impl Greet for Foo {
    fn greet(&self) -> String {
        assist(self.value)
    }
}
