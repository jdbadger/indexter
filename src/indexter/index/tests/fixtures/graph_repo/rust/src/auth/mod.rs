pub struct Handler {
    pub name: String,
}

impl Handler {
    pub fn new() -> Self {
        Handler { name: String::from("anon") }
    }

    pub fn reset() -> Self {
        Self::new()
    }

    pub fn login(&self) {
        self.validate();
    }

    fn validate(&self) {}
}
