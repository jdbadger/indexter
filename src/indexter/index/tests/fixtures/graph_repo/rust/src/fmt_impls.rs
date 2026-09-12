use std::fmt::{self, Debug};

use crate::model::Foo;

pub trait Greet {
    fn greet(&self) -> String;
}

impl Greet for Foo {
    fn greet(&self) -> String {
        format!("Foo({})", self.value)
    }
}

impl Debug for Foo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Foo {{ value: {} }}", self.value)
    }
}
