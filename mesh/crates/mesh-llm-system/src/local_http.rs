//! Client builders for a node's own plain-HTTP endpoints on 127.0.0.1.

/// A builder for clients that only call this node's own plain-HTTP endpoints
/// on 127.0.0.1 (API, console, planner). It trusts no roots: such a client
/// never talks TLS, so building it doesn't load the system's CA certificates,
/// which fails on systems without a CA bundle. Those endpoints never redirect,
/// so it doesn't follow redirects either, and it ignores proxy settings:
/// requests stay on the address they were sent to. Callers add their own
/// timeout.
pub fn local_http_client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .tls_certs_only([])
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
}

/// The blocking counterpart of [`local_http_client_builder`], with the same
/// guarantees.
pub fn local_http_blocking_client_builder() -> reqwest::blocking::ClientBuilder {
    reqwest::blocking::Client::builder()
        .tls_certs_only([])
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
}

#[cfg(test)]
mod tests {
    use super::local_http_blocking_client_builder;

    const BLOCKING_CLIENT_CHILD: &str = "MESH_LLM_TEST_LOCAL_HTTP_BLOCKING_CLIENT_CHILD";
    const BLOCKING_PROXY_CHILD: &str = "MESH_LLM_TEST_LOCAL_HTTP_BLOCKING_PROXY_CHILD";

    // SSL_CERT_FILE and SSL_CERT_DIR point the system root loading at an empty
    // file and directory on Linux; they are set on a child test process only.
    #[test]
    fn blocking_client_needs_no_ca_certificates() {
        let temp = tempfile::tempdir().unwrap();
        let empty_file = temp.path().join("empty.pem");
        std::fs::write(&empty_file, "").unwrap();
        let empty_dir = temp.path().join("certs");
        std::fs::create_dir(&empty_dir).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "local_http::tests::blocking_client_child",
                "--nocapture",
            ])
            .env(BLOCKING_CLIENT_CHILD, "1")
            .env("SSL_CERT_FILE", &empty_file)
            .env("SSL_CERT_DIR", &empty_dir)
            .output()
            .expect("spawn blocking client child");
        assert!(
            output.status.success(),
            "blocking client child failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "child test filter must execute its assertion"
        );
    }

    // The responder answers its first request with a redirect to itself. A
    // client that followed it would connect again, which the listener sees as
    // a second pending connection.
    #[test]
    fn blocking_client_does_not_follow_redirects() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let responder = listener.try_clone().unwrap();
        let location = url.clone();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = responder.accept().unwrap();
            let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            while std::io::BufRead::read_line(&mut reader, &mut line).unwrap() > 2 {
                line.clear();
            }
            let response = format!(
                "HTTP/1.1 307 Temporary Redirect\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            std::io::Write::write_all(&mut stream, response.as_bytes()).unwrap();
        });

        let response = local_http_blocking_client_builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap()
            .get(&url)
            .send()
            .expect("the redirect response itself is the result");
        server.join().unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::TEMPORARY_REDIRECT);
        listener.set_nonblocking(true).unwrap();
        assert_eq!(
            listener.accept().map(|_| ()).unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock,
            "the client must not send a second request"
        );
    }

    // HTTP_PROXY points at a closed local port and NO_PROXY is unset, on a
    // child test process only. A client that used the proxy would fail to
    // connect; the local client must reach the responder directly.
    #[test]
    fn blocking_client_ignores_proxy_settings() {
        let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let proxy = format!("http://{}", closed.local_addr().unwrap());
        drop(closed);
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "local_http::tests::blocking_proxy_child",
                "--nocapture",
            ])
            .env(BLOCKING_PROXY_CHILD, "1")
            .env("HTTP_PROXY", &proxy)
            .env("http_proxy", &proxy)
            .env("ALL_PROXY", &proxy)
            .env("all_proxy", &proxy)
            .env_remove("NO_PROXY")
            .env_remove("no_proxy")
            .output()
            .expect("spawn blocking proxy child");
        assert!(
            output.status.success(),
            "blocking proxy child failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "child test filter must execute its assertion"
        );
    }

    #[test]
    fn blocking_proxy_child() {
        if std::env::var_os(BLOCKING_PROXY_CHILD).is_none() {
            return;
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            while std::io::BufRead::read_line(&mut reader, &mut line).unwrap() > 2 {
                line.clear();
            }
            std::io::Write::write_all(
                &mut stream,
                b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
        });

        let response = local_http_blocking_client_builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap()
            .get(&url)
            .send()
            .expect("the local request must not go through the proxy");
        server.join().unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
    }

    #[test]
    fn blocking_client_child() {
        if std::env::var_os(BLOCKING_CLIENT_CHILD).is_none() {
            return;
        }
        local_http_blocking_client_builder()
            .build()
            .expect("a client without TLS roots builds without system certificates");
    }
}
