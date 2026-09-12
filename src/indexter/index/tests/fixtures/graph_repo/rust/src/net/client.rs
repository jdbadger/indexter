use crate::auth::Handler;

pub fn client_login() {
    let h = Handler::new();
    h.login();
    super::retry();
}
