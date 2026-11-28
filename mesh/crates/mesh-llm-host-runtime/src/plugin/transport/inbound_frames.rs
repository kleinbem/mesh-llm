//! Length-prefixed envelopes read from a plugin connection.
//!
//! The connection loop selects between reading and writing, so a read can be
//! cancelled at any point. Bytes already read stay in `buffer`, and `next` picks
//! up where the cancelled call stopped instead of losing part of a frame.

use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use tokio::time::Instant;

use super::LocalStream;
use crate::plugin::proto;

const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
const PREFIX_BYTES: usize = 4;
const READ_CHUNK_BYTES: usize = 64 * 1024;

pub(crate) struct InboundFrames {
    buffer: Vec<u8>,
    body_deadline: Option<Instant>,
    prefix_timeout: Duration,
    body_timeout: Duration,
}

impl InboundFrames {
    pub(crate) fn new(prefix_timeout: Duration, body_timeout: Duration) -> Self {
        Self {
            buffer: Vec::new(),
            body_deadline: None,
            prefix_timeout,
            body_timeout,
        }
    }

    /// The next envelope. Cancel-safe: only `LocalStream::read` is awaited, and
    /// what it returns is buffered before the next await.
    pub(crate) async fn next(&mut self, stream: &mut LocalStream) -> Result<proto::Envelope> {
        // Waiting for a prefix restarts its timeout on every call, as an idle
        // connection is only judged between frames.
        let prefix_deadline = Instant::now() + self.prefix_timeout;
        let mut chunk = vec![0u8; READ_CHUNK_BYTES];
        loop {
            if let Some(envelope) = self.take_frame()? {
                return Ok(envelope);
            }
            let (deadline, label, timeout) = match self.body_deadline {
                Some(deadline) => (deadline, "body", self.body_timeout),
                None => (prefix_deadline, "prefix", self.prefix_timeout),
            };
            let read = tokio::time::timeout_at(deadline, stream.read(&mut chunk))
                .await
                .map_err(|_| anyhow!("timeout reading plugin frame {label} after {timeout:?}"))??;
            if read == 0 {
                bail!("plugin connection closed");
            }
            self.buffer.extend_from_slice(&chunk[..read]);
        }
    }

    fn take_frame(&mut self) -> Result<Option<proto::Envelope>> {
        if self.buffer.len() < PREFIX_BYTES {
            return Ok(None);
        }
        let prefix: [u8; PREFIX_BYTES] = self.buffer[..PREFIX_BYTES].try_into().unwrap();
        let len = u32::from_le_bytes(prefix) as usize;
        if len > MAX_FRAME_BYTES {
            bail!("Plugin frame too large");
        }
        let frame_end = PREFIX_BYTES + len;
        if self.buffer.len() < frame_end {
            self.body_deadline
                .get_or_insert_with(|| Instant::now() + self.body_timeout);
            return Ok(None);
        }
        let envelope = prost::Message::decode(&self.buffer[PREFIX_BYTES..frame_end])?;
        self.buffer.drain(..frame_end);
        self.body_deadline = None;
        Ok(Some(envelope))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::task::{Context, Poll, Waker};
    use tokio::io::AsyncWriteExt;

    fn frame(envelope: &proto::Envelope) -> Vec<u8> {
        let mut body = Vec::new();
        prost::Message::encode(envelope, &mut body).unwrap();
        let mut frame = (body.len() as u32).to_le_bytes().to_vec();
        frame.extend_from_slice(&body);
        frame
    }

    fn envelope(request_id: u64) -> proto::Envelope {
        proto::Envelope {
            request_id,
            plugin_id: "fixture".into(),
            ..Default::default()
        }
    }

    fn frames() -> InboundFrames {
        InboundFrames::new(Duration::from_secs(10), Duration::from_secs(10))
    }

    #[tokio::test]
    async fn a_read_cancelled_mid_frame_keeps_the_bytes_it_read() {
        let (host, mut plugin) = tokio::io::duplex(1024);
        let mut host = LocalStream::Memory(host);
        let mut frames = frames();
        let first = frame(&envelope(7));
        let (head, rest) = first.split_at(PREFIX_BYTES + 2);
        plugin.write_all(head).await.unwrap();

        // The connection loop's select drops the read once an outbound envelope is ready.
        {
            let mut read = std::pin::pin!(frames.next(&mut host));
            let mut cx = Context::from_waker(Waker::noop());
            assert!(matches!(read.as_mut().poll(&mut cx), Poll::Pending));
        }

        plugin.write_all(rest).await.unwrap();
        plugin.write_all(&frame(&envelope(8))).await.unwrap();
        assert_eq!(frames.next(&mut host).await.unwrap().request_id, 7);
        assert_eq!(frames.next(&mut host).await.unwrap().request_id, 8);
    }

    #[tokio::test]
    async fn several_frames_in_one_read_are_returned_in_order() {
        let (host, mut plugin) = tokio::io::duplex(4096);
        let mut host = LocalStream::Memory(host);
        let mut frames = frames();
        let mut bytes = frame(&envelope(1));
        bytes.extend(frame(&envelope(2)));
        plugin.write_all(&bytes).await.unwrap();
        assert_eq!(frames.next(&mut host).await.unwrap().request_id, 1);
        assert_eq!(frames.next(&mut host).await.unwrap().request_id, 2);
    }

    #[tokio::test]
    async fn an_oversized_length_is_rejected() {
        let (host, mut plugin) = tokio::io::duplex(1024);
        let mut host = LocalStream::Memory(host);
        plugin
            .write_all(&((MAX_FRAME_BYTES as u32) + 1).to_le_bytes())
            .await
            .unwrap();
        let error = frames().next(&mut host).await.unwrap_err();
        assert_eq!(error.to_string(), "Plugin frame too large");
    }

    #[tokio::test]
    async fn a_closed_connection_ends_the_read() {
        let (host, plugin) = tokio::io::duplex(1024);
        let mut host = LocalStream::Memory(host);
        drop(plugin);
        assert!(frames().next(&mut host).await.is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn a_body_that_stops_arriving_times_out() {
        let (host, mut plugin) = tokio::io::duplex(1024);
        let mut host = LocalStream::Memory(host);
        let bytes = frame(&envelope(3));
        plugin.write_all(&bytes[..PREFIX_BYTES + 1]).await.unwrap();
        let mut frames = InboundFrames::new(Duration::from_secs(600), Duration::from_secs(30));
        let error = frames.next(&mut host).await.unwrap_err();
        assert!(error.to_string().contains("frame body"), "{error}");
    }
}
