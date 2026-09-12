mod auth;
mod net;
mod model;
mod fmt_impls;

use crate::auth::Handler;

fn main() {
    let h = Handler::new();
    h.login();
}
