use crate::Error;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use crossbeam_channel::{bounded, Receiver, Sender};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::{input, Pixel, Sample};
use ffmpeg_next::media::Type;
use ffmpeg_next::software::resampling::Context as ResampleContext;
use ffmpeg_next::software::scaling::{context::Context as ScaleContext, flag::Flags};
use ffmpeg_next::util::frame::audio::Audio as FFmpegAudioFrame;
use ffmpeg_next::util::frame::video::Video as FFmpegFrame;
use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Position in the media.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Position {
    Time(Duration),
    Frame(u64),
}

impl From<Duration> for Position {
    fn from(t: Duration) -> Self {
        Position::Time(t)
    }
}

impl From<u64> for Position {
    fn from(f: u64) -> Self {
        Position::Frame(f)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Frame {
    data: Vec<u8>,
    width: u32,
    height: u32,
    timestamp: Duration,
    pts: i64,
}

impl Frame {
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            width: 0,
            height: 0,
            timestamp: Duration::ZERO,
            pts: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeResult {
    Video,
    Audio,
    Eos,
}

// Improved ring buffer with better wraparound handling
struct AudioRingBuffer {
    buffer: Vec<f32>,
    capacity: usize,
    read_pos: usize,
    write_pos: usize,
    len: usize,
}

impl AudioRingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity],
            capacity,
            read_pos: 0,
            write_pos: 0,
            len: 0,
        }
    }

    fn write(&mut self, samples: &[f32]) -> usize {
        let available = self.capacity - self.len;
        let to_write = samples.len().min(available);

        if to_write == 0 {
            return 0;
        }

        let first_chunk = (self.capacity - self.write_pos).min(to_write);
        self.buffer[self.write_pos..self.write_pos + first_chunk]
            .copy_from_slice(&samples[..first_chunk]);

        if first_chunk < to_write {
            let remaining = to_write - first_chunk;
            self.buffer[..remaining].copy_from_slice(&samples[first_chunk..to_write]);
        }

        self.write_pos = (self.write_pos + to_write) % self.capacity;
        self.len += to_write;
        to_write
    }

    fn read(&mut self, output: &mut [f32]) -> usize {
        let available = self.len.min(output.len());

        if available == 0 {
            output.fill(0.0);
            return 0;
        }

        let first_chunk = (self.capacity - self.read_pos).min(available);
        output[..first_chunk]
            .copy_from_slice(&self.buffer[self.read_pos..self.read_pos + first_chunk]);

        if first_chunk < available {
            let remaining = available - first_chunk;
            output[first_chunk..available].copy_from_slice(&self.buffer[..remaining]);
        }

        self.read_pos = (self.read_pos + available) % self.capacity;
        self.len -= available;

        if available < output.len() {
            output[available..].fill(0.0);
        }

        available
    }

    fn available(&self) -> usize {
        self.len
    }

    fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
        self.len = 0;
    }
}

#[derive(Debug, Clone)]
pub struct VideoOptions {
    pub frame_buffer_capacity: Option<usize>,
    pub looping: Option<bool>,
    pub speed: Option<f64>,
    pub prebuffer_frames: Option<usize>,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            frame_buffer_capacity: Some(60), // Increased default buffer
            looping: Some(false),
            speed: Some(1.0),
            prebuffer_frames: Some(10), // Increased prebuffer
        }
    }
}

enum DecoderCommand {
    Seek(i64, bool),
    SetPaused(bool),
    SetSpeed(f64),
    Stop,
}

pub(crate) struct Internal {
    pub(crate) id: u64,
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) framerate: f64,
    pub(crate) duration: Duration,
    pub(crate) time_base: ffmpeg::Rational,
    pub(crate) has_audio: bool,
    pub(crate) sample_rate: u32,

    pub(crate) frame: Arc<Mutex<Frame>>,
    pub(crate) upload_frame: Arc<AtomicBool>,
    pub(crate) frame_buffer: Arc<Mutex<VecDeque<Frame>>>,
    pub(crate) frame_buffer_capacity: Arc<AtomicUsize>,

    pub(crate) audio_ring: Arc<Mutex<AudioRingBuffer>>,
    pub(crate) audio_clock: Arc<Mutex<Duration>>,
    pub(crate) samples_played: Arc<AtomicU64>,
    pub(crate) last_audio_pts: Arc<Mutex<Duration>>, // NEW: Track actual audio PTS

    pub(crate) speed: Arc<AtomicU64>,
    pub(crate) looping: Arc<AtomicBool>,
    pub(crate) is_paused: Arc<AtomicBool>,
    pub(crate) is_eos: Arc<AtomicBool>,
    pub(crate) current_pts: Arc<AtomicU64>,
    pub(crate) playback_start: Arc<Mutex<Option<Instant>>>, // Simplified timing
    pub(crate) seek_target: Arc<Mutex<Option<Duration>>>,   // NEW: Track seek target

    pub(crate) command_tx: Sender<DecoderCommand>,
    pub(crate) alive: Arc<AtomicBool>,
    pub(crate) decoder_thread: Option<std::thread::JoinHandle<()>>,
    pub(crate) _audio_stream: Option<Stream>,

    pub(crate) display_width_override: Option<u32>,
    pub(crate) display_height_override: Option<u32>,

    pub(crate) volume: Arc<Mutex<f64>>,
    pub(crate) muted: Arc<AtomicBool>,
}

impl std::fmt::Debug for Internal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Internal")
            .field("id", &self.id)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("framerate", &self.framerate)
            .field("has_audio", &self.has_audio)
            .finish_non_exhaustive()
    }
}

impl Internal {
    pub(crate) fn seek(&self, position: impl Into<Position>, accurate: bool) -> Result<(), Error> {
        let position = position.into();

        let target_seconds = match position {
            Position::Time(duration) => duration.as_secs_f64(),
            Position::Frame(frame_num) => {
                let time_per_frame = 1.0 / self.framerate;
                frame_num as f64 * time_per_frame
            }
        };

        let timestamp = (target_seconds * self.time_base.denominator() as f64
            / self.time_base.numerator() as f64) as i64;

        let target_duration = Duration::from_secs_f64(target_seconds);

        // Clear state FIRST
        self.is_eos.store(false, Ordering::Release);
        self.frame_buffer.lock().clear();
        self.audio_ring.lock().clear();

        // Set seek target - decoder will handle the rest
        *self.seek_target.lock() = Some(target_duration);
        *self.playback_start.lock() = None;

        self.command_tx
            .send(DecoderCommand::Seek(timestamp, accurate))
            .map_err(|_| Error::Sync)?;

        Ok(())
    }

    pub(crate) fn set_speed(&mut self, speed: f64) -> Result<(), Error> {
        if speed <= 0.0 {
            return Err(Error::Framerate(speed));
        }

        self.speed.store(speed.to_bits(), Ordering::SeqCst);
        *self.playback_start.lock() = None; // Reset timing

        self.command_tx
            .send(DecoderCommand::SetSpeed(speed))
            .map_err(|_| Error::Sync)?;

        Ok(())
    }

    pub(crate) fn set_paused(&mut self, paused: bool) {
        let was_paused = self.is_paused.swap(paused, Ordering::SeqCst);

        if was_paused && !paused {
            *self.playback_start.lock() = None; // Reset timing on resume
        }

        let _ = self.command_tx.send(DecoderCommand::SetPaused(paused));

        if self.is_eos.load(Ordering::Acquire) && !paused {
            let _ = self.seek(0, false);
        }
    }

    pub(crate) fn paused(&self) -> bool {
        self.is_paused.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone)]
pub struct Video(pub(crate) Arc<RwLock<Internal>>);

impl Drop for Video {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) == 1 {
            if let Some(mut inner) = self.0.try_write() {
                inner.alive.store(false, Ordering::SeqCst);
                let _ = inner.command_tx.send(DecoderCommand::Stop);

                if let Some(_stream) = inner._audio_stream.take() {
                    // Stream will be dropped and stopped
                }

                if let Some(worker) = inner.decoder_thread.take() {
                    let _ = worker.join();
                }
            }
        }
    }
}

impl Video {
    pub fn new(uri: &url::Url) -> Result<Self, Error> {
        Self::new_with_options(uri, VideoOptions::default())
    }

    pub fn new_with_options(uri: &url::Url, options: VideoOptions) -> Result<Self, Error> {
        ffmpeg::init().map_err(|_| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to initialize FFmpeg",
            ))
        })?;

        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);

        let path = if uri.scheme() == "file" {
            uri.to_file_path()
                .map_err(|_| Error::Uri)?
                .to_string_lossy()
                .into_owned()
        } else {
            uri.as_str().to_string()
        };

        let ictx = input(&path).map_err(|_| Error::Uri)?;

        let video_stream = ictx.streams().best(Type::Video).ok_or(Error::Caps)?;
        let video_stream_index = video_stream.index();

        let codec_params = video_stream.parameters();
        let _codec = ffmpeg::codec::decoder::find(codec_params.id())
            .ok_or(Error::Caps)?
            .video()
            .map_err(|_| Error::Cast)?;

        let mut decoder = ffmpeg::codec::context::Context::from_parameters(codec_params)
            .map_err(|_| Error::Caps)?
            .decoder()
            .video()
            .map_err(|_| Error::Cast)?;

        decoder.set_threading(ffmpeg::threading::Config {
            kind: ffmpeg::threading::Type::Frame,
            count: 0,
        });

        let width = decoder.width() as i32;
        let height = decoder.height() as i32;
        let time_base = video_stream.time_base();

        let framerate = {
            let avg_frame_rate = video_stream.avg_frame_rate();
            if avg_frame_rate.numerator() > 0 && avg_frame_rate.denominator() > 0 {
                avg_frame_rate.numerator() as f64 / avg_frame_rate.denominator() as f64
            } else {
                25.0
            }
        };

        if framerate.is_nan() || framerate.is_infinite() || framerate <= 0.0 {
            return Err(Error::Framerate(framerate));
        }

        let duration = if ictx.duration() > 0 {
            Duration::from_micros(ictx.duration() as u64)
        } else {
            Duration::from_secs(0)
        };

        let audio_stream = ictx.streams().best(Type::Audio);
        let has_audio = audio_stream.is_some();
        let audio_stream_index = audio_stream.map(|s| s.index());

        let frame = Arc::new(Mutex::new(Frame::empty()));
        let upload_frame = Arc::new(AtomicBool::new(false));
        let frame_buffer = Arc::new(Mutex::new(VecDeque::with_capacity(
            options.frame_buffer_capacity.unwrap_or(60),
        )));
        let frame_buffer_capacity = Arc::new(AtomicUsize::new(
            options.frame_buffer_capacity.unwrap_or(60),
        ));

        // Reduced audio buffer - 1 second is plenty
        let audio_ring = Arc::new(Mutex::new(AudioRingBuffer::new(48000 * 2 * 2)));
        let audio_clock = Arc::new(Mutex::new(Duration::ZERO));
        let samples_played = Arc::new(AtomicU64::new(0));
        let last_audio_pts = Arc::new(Mutex::new(Duration::ZERO));

        let speed = Arc::new(AtomicU64::new(options.speed.unwrap_or(1.0).to_bits()));
        let looping = Arc::new(AtomicBool::new(options.looping.unwrap_or(false)));
        let is_paused = Arc::new(AtomicBool::new(false));
        let is_eos = Arc::new(AtomicBool::new(false));
        let current_pts = Arc::new(AtomicU64::new(0));
        let alive = Arc::new(AtomicBool::new(true));
        let volume = Arc::new(Mutex::new(1.0));
        let muted = Arc::new(AtomicBool::new(false));
        let playback_start = Arc::new(Mutex::new(None::<Instant>));
        let seek_target = Arc::new(Mutex::new(None::<Duration>));

        let (command_tx, command_rx) = bounded(100);
        let prebuffer_frames = options.prebuffer_frames.unwrap_or(10);

        let (audio_stream, actual_sample_rate) = if has_audio {
            let audio_ring_for_output = Arc::clone(&audio_ring);
            let volume_for_output = Arc::clone(&volume);
            let muted_for_output = Arc::clone(&muted);
            let is_paused_for_output = Arc::clone(&is_paused);
            let samples_played_for_output = Arc::clone(&samples_played);

            match Self::setup_audio_output(
                audio_ring_for_output,
                volume_for_output,
                muted_for_output,
                is_paused_for_output,
                samples_played_for_output,
            ) {
                Ok((stream, sr)) => (Some(stream), sr),
                Err(e) => {
                    log::warn!("Failed to setup audio: {:?}", e);
                    (None, 48000)
                }
            }
        } else {
            (None, 48000)
        };

        // Clone all Arc references for decoder thread
        let frame_ref = Arc::clone(&frame);
        let upload_frame_ref = Arc::clone(&upload_frame);
        let frame_buffer_ref = Arc::clone(&frame_buffer);
        let frame_buffer_capacity_ref = Arc::clone(&frame_buffer_capacity);
        let audio_ring_ref = Arc::clone(&audio_ring);
        let audio_clock_ref = Arc::clone(&audio_clock);
        let samples_played_ref = Arc::clone(&samples_played);
        let last_audio_pts_ref = Arc::clone(&last_audio_pts);
        let speed_ref = Arc::clone(&speed);
        let looping_ref = Arc::clone(&looping);
        let is_paused_ref = Arc::clone(&is_paused);
        let is_eos_ref = Arc::clone(&is_eos);
        let current_pts_ref = Arc::clone(&current_pts);
        let alive_ref = Arc::clone(&alive);
        let playback_start_ref = Arc::clone(&playback_start);
        let seek_target_ref = Arc::clone(&seek_target);

        let decoder_thread = std::thread::spawn(move || {
            if let Err(e) = Self::decoder_loop(
                path,
                video_stream_index,
                audio_stream_index,
                frame_ref,
                upload_frame_ref,
                frame_buffer_ref,
                frame_buffer_capacity_ref,
                audio_ring_ref,
                audio_clock_ref,
                samples_played_ref,
                last_audio_pts_ref,
                speed_ref,
                looping_ref,
                is_paused_ref,
                is_eos_ref,
                current_pts_ref,
                alive_ref,
                playback_start_ref,
                seek_target_ref,
                command_rx,
                time_base,
                framerate,
                actual_sample_rate,
                prebuffer_frames,
            ) {
                log::error!("Decoder thread error: {:?}", e);
            }
        });

        Ok(Video(Arc::new(RwLock::new(Internal {
            id,
            width,
            height,
            framerate,
            duration,
            time_base,
            has_audio,
            sample_rate: actual_sample_rate,

            frame,
            upload_frame,
            frame_buffer,
            frame_buffer_capacity,

            audio_ring,
            audio_clock,
            samples_played,
            last_audio_pts,

            speed,
            looping,
            is_paused,
            is_eos,
            current_pts,
            playback_start,
            seek_target,

            command_tx,
            alive,
            decoder_thread: Some(decoder_thread),
            _audio_stream: audio_stream,

            display_width_override: None,
            display_height_override: None,

            volume,
            muted,
        }))))
    }

    #[allow(clippy::too_many_arguments)]
    fn decoder_loop(
        path: String,
        video_stream_index: usize,
        audio_stream_index: Option<usize>,
        frame: Arc<Mutex<Frame>>,
        upload_frame: Arc<AtomicBool>,
        frame_buffer: Arc<Mutex<VecDeque<Frame>>>,
        frame_buffer_capacity: Arc<AtomicUsize>,
        audio_ring: Arc<Mutex<AudioRingBuffer>>,
        audio_clock: Arc<Mutex<Duration>>,
        samples_played: Arc<AtomicU64>,
        last_audio_pts: Arc<Mutex<Duration>>,
        speed: Arc<AtomicU64>,
        looping: Arc<AtomicBool>,
        is_paused: Arc<AtomicBool>,
        is_eos: Arc<AtomicBool>,
        current_pts: Arc<AtomicU64>,
        alive: Arc<AtomicBool>,
        playback_start: Arc<Mutex<Option<Instant>>>,
        seek_target: Arc<Mutex<Option<Duration>>>,
        command_rx: Receiver<DecoderCommand>,
        time_base: ffmpeg::Rational,
        framerate: f64,
        output_sample_rate: u32,
        prebuffer_frames: usize,
    ) -> Result<(), Error> {
        let mut ictx = input(&path).map_err(|_| Error::Uri)?;
        let video_stream = ictx.stream(video_stream_index).ok_or(Error::Caps)?;

        let codec_params = video_stream.parameters();
        let mut decoder = ffmpeg::codec::context::Context::from_parameters(codec_params)
            .map_err(|_| Error::Caps)?
            .decoder()
            .video()
            .map_err(|_| Error::Cast)?;

        decoder.set_threading(ffmpeg::threading::Config {
            kind: ffmpeg::threading::Type::Frame,
            count: 0,
        });

        let mut scaler = ScaleContext::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::NV12,
            decoder.width(),
            decoder.height(),
            Flags::LANCZOS,
        )
        .map_err(|_| Error::Caps)?;

        let mut audio_decoder = None;
        let mut audio_resampler = None;
        let mut audio_time_base = ffmpeg::Rational::new(1, 1);

        if let Some(audio_idx) = audio_stream_index {
            let audio_stream = ictx.stream(audio_idx).ok_or(Error::Caps)?;
            audio_time_base = audio_stream.time_base();

            let audio_params = audio_stream.parameters();
            let dec = ffmpeg::codec::context::Context::from_parameters(audio_params)
                .map_err(|_| Error::Caps)?
                .decoder()
                .audio()
                .map_err(|_| Error::Cast)?;

            let resampler = ResampleContext::get(
                dec.format(),
                dec.channel_layout(),
                dec.rate(),
                Sample::F32(ffmpeg::format::sample::Type::Packed),
                ffmpeg::ChannelLayout::STEREO,
                output_sample_rate,
            )
            .map_err(|_| Error::Caps)?;

            audio_decoder = Some(dec);
            audio_resampler = Some(resampler);
        }

        let mut prebuffering = true;
        let has_audio = audio_decoder.is_some();

        while alive.load(Ordering::Acquire) {
            // Process commands
            while let Ok(cmd) = command_rx.try_recv() {
                match cmd {
                    DecoderCommand::Seek(timestamp, _accurate) => {
                        if let Err(e) = ictx.seek(timestamp, ..) {
                            log::error!("Seek failed: {:?}", e);
                        }
                        decoder.flush();
                        if let Some(ref mut ad) = audio_decoder {
                            ad.flush();
                        }
                        is_eos.store(false, Ordering::Release);
                        prebuffering = true;
                    }
                    DecoderCommand::SetPaused(_) => {}
                    DecoderCommand::SetSpeed(_) => {}
                    DecoderCommand::Stop => return Ok(()),
                }
            }

            // Handle EOS
            if is_eos.load(Ordering::Acquire) {
                if looping.load(Ordering::SeqCst) {
                    if let Err(e) = ictx.seek(0, ..) {
                        log::error!("Loop seek failed: {:?}", e);
                    }
                    decoder.flush();
                    if let Some(ref mut ad) = audio_decoder {
                        ad.flush();
                    }
                    is_eos.store(false, Ordering::Release);
                    frame_buffer.lock().clear();
                    audio_ring.lock().clear();
                    samples_played.store(0, Ordering::SeqCst);
                    *audio_clock.lock() = Duration::ZERO;
                    *last_audio_pts.lock() = Duration::ZERO;
                    *playback_start.lock() = None;
                    *seek_target.lock() = None;
                    prebuffering = true;
                } else {
                    std::thread::sleep(Duration::from_millis(16));
                    continue;
                }
            }

            let paused = is_paused.load(Ordering::Acquire);

            // Pre-buffering phase
            if prebuffering && !paused {
                let vbuf_len = frame_buffer.lock().len();
                let audio_buf_level = audio_ring.lock().available();

                let video_ready = vbuf_len >= prebuffer_frames;
                let audio_ready =
                    !has_audio || audio_buf_level >= (output_sample_rate as usize / 2);

                if video_ready && audio_ready {
                    prebuffering = false;

                    // Initialize playback timing
                    if let Some(target) = seek_target.lock().take() {
                        // After seek - use seek target
                        *audio_clock.lock() = target;
                        *last_audio_pts.lock() = target;
                        let samples_for_target =
                            (target.as_secs_f64() * output_sample_rate as f64 * 2.0) as u64;
                        samples_played.store(samples_for_target, Ordering::SeqCst);
                    } else {
                        // Initial playback - start from zero
                        *audio_clock.lock() = Duration::ZERO;
                        *last_audio_pts.lock() = Duration::ZERO;
                        samples_played.store(0, Ordering::SeqCst);
                    }

                    *playback_start.lock() = Some(Instant::now());
                }
            }

            // Decode packets when not paused
            if paused {
                std::thread::sleep(Duration::from_millis(16));
                continue;
            }

            let vbuf_len = frame_buffer.lock().len();
            let vbuf_cap = frame_buffer_capacity.load(Ordering::SeqCst);
            let audio_space = audio_ring.lock().capacity - audio_ring.lock().len;

            // Decode more aggressively during prebuffering
            let should_decode = prebuffering
                || vbuf_len < (vbuf_cap * 3 / 4)  // Keep buffer well filled
                || audio_space > (output_sample_rate * 2) as usize / 4;

            if should_decode {
                match Self::decode_next_packet(
                    &mut ictx,
                    &mut decoder,
                    &mut scaler,
                    audio_decoder.as_mut(),
                    audio_resampler.as_mut(),
                    video_stream_index,
                    audio_stream_index,
                    &frame_buffer,
                    &frame_buffer_capacity,
                    &audio_ring,
                    &last_audio_pts,
                    time_base,
                    audio_time_base,
                    output_sample_rate,
                ) {
                    Ok(DecodeResult::Video) => {}
                    Ok(DecodeResult::Audio) => {}
                    Ok(DecodeResult::Eos) => {
                        is_eos.store(true, Ordering::Release);
                    }
                    Err(e) => {
                        log::error!("Decode error: {:?}", e);
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
            }

            // Present frames at the correct time
            if !prebuffering && !paused {
                let current_speed = f64::from_bits(speed.load(Ordering::SeqCst));
                let current_time = Self::get_playback_time(
                    &playback_start,
                    &samples_played,
                    &audio_clock,
                    output_sample_rate,
                    has_audio,
                    current_speed,
                );

                let mut buf = frame_buffer.lock();

                if let Some(next_frame) = buf.front() {
                    let frame_time = next_frame.timestamp;
                    let time_until_present = frame_time.as_secs_f64() - current_time.as_secs_f64();

                    if time_until_present <= 0.010 {
                        // 10ms tolerance
                        // Time to present this frame
                        let presented = buf.pop_front().unwrap();
                        current_pts.store(presented.pts as u64, Ordering::SeqCst);
                        *frame.lock() = presented.clone();
                        upload_frame.store(true, Ordering::SeqCst);

                        // Skip frames if we're falling behind (> 50ms late)
                        while let Some(next) = buf.front() {
                            let next_time = next.timestamp.as_secs_f64();
                            if next_time < current_time.as_secs_f64() - 0.050 {
                                let skipped = buf.pop_front().unwrap();
                                log::warn!(
                                    "Skipped frame at {}s (behind by {}ms)",
                                    next_time,
                                    ((current_time.as_secs_f64() - next_time) * 1000.0) as i32
                                );
                                *frame.lock() = skipped;
                            } else {
                                break;
                            }
                        }
                    } else if time_until_present > 0.020 {
                        // Too early - sleep for a bit
                        drop(buf);
                        let sleep_ms =
                            (time_until_present * 1000.0 / 2.0).min(10.0).max(1.0) as u64;
                        std::thread::sleep(Duration::from_millis(sleep_ms));
                        continue;
                    }
                }
            }

            // Adaptive sleep based on state
            let sleep_ms = if prebuffering {
                5 // Faster during prebuffer
            } else if paused {
                16
            } else {
                2 // Balance between responsiveness and CPU
            };
            std::thread::sleep(Duration::from_millis(sleep_ms));
        }

        Ok(())
    }

    fn get_playback_time(
        playback_start: &Arc<Mutex<Option<Instant>>>,
        samples_played: &Arc<AtomicU64>,
        audio_clock: &Arc<Mutex<Duration>>,
        sample_rate: u32,
        has_audio: bool,
        speed: f64,
    ) -> Duration {
        if has_audio {
            // Audio clock is the master - use samples played
            let played = samples_played.load(Ordering::SeqCst);
            let audio_time = Duration::from_secs_f64(played as f64 / (sample_rate as f64 * 2.0));
            *audio_clock.lock() = audio_time;
            audio_time
        } else {
            // No audio - use system clock with speed adjustment
            if let Some(start) = *playback_start.lock() {
                let elapsed = start.elapsed();
                let adjusted = Duration::from_secs_f64(elapsed.as_secs_f64() * speed);
                adjusted
            } else {
                Duration::ZERO
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_next_packet(
        ictx: &mut ffmpeg::format::context::Input,
        decoder: &mut ffmpeg::decoder::Video,
        scaler: &mut ScaleContext,
        audio_decoder: Option<&mut ffmpeg::decoder::Audio>,
        audio_resampler: Option<&mut ResampleContext>,
        video_stream_index: usize,
        audio_stream_index: Option<usize>,
        frame_buffer: &Arc<Mutex<VecDeque<Frame>>>,
        frame_buffer_capacity: &Arc<AtomicUsize>,
        audio_ring: &Arc<Mutex<AudioRingBuffer>>,
        last_audio_pts: &Arc<Mutex<Duration>>,
        video_time_base: ffmpeg::Rational,
        audio_time_base: ffmpeg::Rational,
        _output_sample_rate: u32,
    ) -> Result<DecodeResult, Error> {
        let (stream, packet) = match ictx.packets().next() {
            Some((s, p)) => (s, p),
            None => return Ok(DecodeResult::Eos),
        };

        let stream_index = stream.index();

        if stream_index == video_stream_index {
            if let Err(e) = decoder.send_packet(&packet) {
                log::warn!("Send video packet error: {:?}", e);
                return Ok(DecodeResult::Video);
            }

            let mut decoded = FFmpegFrame::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let mut nv12_frame = FFmpegFrame::empty();
                if let Err(e) = scaler.run(&decoded, &mut nv12_frame) {
                    log::error!("Scaling error: {:?}", e);
                    continue;
                }

                let width = nv12_frame.width();
                let height = nv12_frame.height();
                let y_plane = nv12_frame.data(0);
                let uv_plane = nv12_frame.data(1);
                let y_stride = nv12_frame.stride(0) as usize;
                let uv_stride = nv12_frame.stride(1) as usize;

                let mut data = Vec::with_capacity((width * height * 3 / 2) as usize);

                for row in 0..height as usize {
                    let start = row * y_stride;
                    let end = start + width as usize;
                    data.extend_from_slice(&y_plane[start..end]);
                }

                for row in 0..(height / 2) as usize {
                    let start = row * uv_stride;
                    let end = start + width as usize;
                    data.extend_from_slice(&uv_plane[start..end]);
                }

                let pts = decoded.pts().unwrap_or(0);
                let timestamp_secs = pts as f64 * video_time_base.numerator() as f64
                    / video_time_base.denominator() as f64;
                let timestamp = Duration::from_secs_f64(timestamp_secs.max(0.0));

                let new_frame = Frame {
                    data,
                    width,
                    height,
                    timestamp,
                    pts,
                };

                let capacity = frame_buffer_capacity.load(Ordering::SeqCst);
                if capacity > 0 {
                    let mut buf = frame_buffer.lock();
                    buf.push_back(new_frame);
                    while buf.len() > capacity {
                        buf.pop_front();
                    }
                }

                return Ok(DecodeResult::Video);
            }
        }

        if Some(stream_index) == audio_stream_index {
            if let (Some(audio_dec), Some(audio_res)) = (audio_decoder, audio_resampler) {
                if let Err(e) = audio_dec.send_packet(&packet) {
                    log::warn!("Send audio packet error: {:?}", e);
                    return Ok(DecodeResult::Audio);
                }

                let mut decoded = FFmpegAudioFrame::empty();
                while audio_dec.receive_frame(&mut decoded).is_ok() {
                    // Track audio PTS for sync
                    if let Some(pts) = decoded.pts() {
                        let audio_time_secs = pts as f64 * audio_time_base.numerator() as f64
                            / audio_time_base.denominator() as f64;
                        *last_audio_pts.lock() = Duration::from_secs_f64(audio_time_secs.max(0.0));
                    }

                    let mut resampled = FFmpegAudioFrame::empty();
                    if let Err(e) = audio_res.run(&decoded, &mut resampled) {
                        log::error!("Resampling error: {:?}", e);
                        continue;
                    }

                    let plane = resampled.data(0);
                    let sample_count = resampled.samples() * 2;

                    let mut samples = Vec::with_capacity(sample_count);
                    for chunk in plane.chunks_exact(4).take(sample_count) {
                        let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        samples.push(sample);
                    }

                    let mut ring = audio_ring.lock();
                    let written = ring.write(&samples);

                    if written < samples.len() {
                        log::warn!(
                            "Audio buffer overflow, dropped {} samples",
                            samples.len() - written
                        );
                    }

                    return Ok(DecodeResult::Audio);
                }
            }
        }

        Ok(DecodeResult::Video)
    }

    fn setup_audio_output(
        audio_ring: Arc<Mutex<AudioRingBuffer>>,
        volume: Arc<Mutex<f64>>,
        muted: Arc<AtomicBool>,
        is_paused: Arc<AtomicBool>,
        samples_played: Arc<AtomicU64>,
    ) -> Result<(Stream, cpal::SampleRate), Error> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No audio output device",
            ))
        })?;

        let supported_config = device
            .default_output_config()
            .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        let sample_rate = supported_config.sample_rate();
        let channels = supported_config.channels();

        let config = StreamConfig {
            channels,
            sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if is_paused.load(Ordering::Acquire) {
                        data.fill(0.0);
                        return;
                    }

                    let mut ring = audio_ring.lock();
                    let read_count = ring.read(data);

                    // Apply volume
                    let vol = *volume.lock();
                    let is_muted = muted.load(Ordering::Acquire);

                    if is_muted {
                        data.fill(0.0);
                    } else if (vol - 1.0).abs() > 0.001 {
                        for sample in data.iter_mut() {
                            *sample = (*sample * vol as f32).clamp(-1.0, 1.0);
                        }
                    }

                    samples_played.fetch_add(read_count as u64, Ordering::SeqCst);
                },
                |err| log::error!("Audio stream error: {:?}", err),
                None,
            )
            .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        stream
            .play()
            .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        Ok((stream, sample_rate))
    }

    pub(crate) fn read(&self) -> parking_lot::RwLockReadGuard<'_, Internal> {
        self.0.read()
    }

    pub(crate) fn write(&self) -> parking_lot::RwLockWriteGuard<'_, Internal> {
        self.0.write()
    }

    pub fn has_audio(&self) -> bool {
        self.read().has_audio
    }

    pub fn size(&self) -> (i32, i32) {
        (self.read().width, self.read().height)
    }

    pub fn aspect_ratio(&self) -> f32 {
        let (w, h) = self.size();
        if w <= 0 || h <= 0 {
            return 1.0;
        }
        w as f32 / h as f32
    }

    pub fn set_display_width(&self, width: Option<u32>) {
        self.write().display_width_override = width;
    }

    pub fn set_display_height(&self, height: Option<u32>) {
        self.write().display_height_override = height;
    }

    pub fn set_display_size(&self, width: Option<u32>, height: Option<u32>) {
        let mut inner = self.write();
        inner.display_width_override = width;
        inner.display_height_override = height;
    }

    pub fn display_size(&self) -> (u32, u32) {
        let inner = self.read();
        let natural_w = inner.width.max(0) as u32;
        let natural_h = inner.height.max(0) as u32;
        let ar = if natural_h == 0 {
            1.0
        } else {
            natural_w as f32 / natural_h as f32
        };

        match (inner.display_width_override, inner.display_height_override) {
            (Some(w), Some(h)) => (w, h),
            (Some(w), None) => {
                let h = if ar == 0.0 {
                    natural_h
                } else {
                    (w as f32 / ar).round() as u32
                };
                (w, h)
            }
            (None, Some(h)) => {
                let w = ((h as f32) * ar).round() as u32;
                (w, h)
            }
            (None, None) => (natural_w, natural_h),
        }
    }

    pub fn framerate(&self) -> f64 {
        self.read().framerate
    }

    pub fn set_volume(&self, volume: f64) {
        *self.read().volume.lock() = volume.clamp(0.0, 1.0);
    }

    pub fn volume(&self) -> f64 {
        *self.read().volume.lock()
    }

    pub fn set_muted(&self, muted: bool) {
        self.read().muted.store(muted, Ordering::SeqCst);
    }

    pub fn muted(&self) -> bool {
        self.read().muted.load(Ordering::Acquire)
    }

    pub fn eos(&self) -> bool {
        self.read().is_eos.load(Ordering::Acquire)
    }

    pub fn looping(&self) -> bool {
        self.read().looping.load(Ordering::SeqCst)
    }

    pub fn set_looping(&self, looping: bool) {
        self.write().looping.store(looping, Ordering::SeqCst);
    }

    pub fn set_paused(&self, paused: bool) {
        self.write().set_paused(paused)
    }

    pub fn paused(&self) -> bool {
        self.read().paused()
    }

    pub fn seek(&self, position: impl Into<Position>, accurate: bool) -> Result<(), Error> {
        self.write().seek(position, accurate)
    }

    pub fn set_speed(&self, speed: f64) -> Result<(), Error> {
        self.write().set_speed(speed)
    }

    pub fn speed(&self) -> f64 {
        f64::from_bits(self.read().speed.load(Ordering::SeqCst))
    }

    pub fn position(&self) -> Duration {
        let inner = self.read();

        if inner.has_audio {
            *inner.audio_clock.lock()
        } else {
            let current_speed = f64::from_bits(inner.speed.load(Ordering::SeqCst));
            if let Some(start) = *inner.playback_start.lock() {
                let elapsed = start.elapsed();
                Duration::from_secs_f64(elapsed.as_secs_f64() * current_speed)
            } else {
                Duration::ZERO
            }
        }
    }

    pub fn duration(&self) -> Duration {
        self.read().duration
    }

    pub fn restart_stream(&self) -> Result<(), Error> {
        self.write().is_eos.store(false, Ordering::Release);
        self.set_paused(false);
        self.seek(0, false)
    }

    pub fn current_frame_data(&self) -> Option<(Vec<u8>, u32, u32)> {
        let inner = self.read();
        let frame = inner.frame.lock();

        if !frame.data.is_empty() {
            Some((frame.data.clone(), frame.width, frame.height))
        } else {
            None
        }
    }

    pub fn take_frame_ready(&self) -> bool {
        self.read().upload_frame.swap(false, Ordering::SeqCst)
    }

    pub fn set_frame_buffer_capacity(&self, capacity: usize) {
        let inner = self.read();
        inner
            .frame_buffer_capacity
            .store(capacity, Ordering::SeqCst);

        if capacity == 0 {
            inner.frame_buffer.lock().clear();
        } else {
            let mut buf = inner.frame_buffer.lock();
            while buf.len() > capacity {
                buf.pop_front();
            }
        }
    }

    pub fn frame_buffer_capacity(&self) -> usize {
        self.read().frame_buffer_capacity.load(Ordering::SeqCst)
    }

    pub fn pop_buffered_frame(&self) -> Option<(Vec<u8>, u32, u32)> {
        let inner = self.read();
        let maybe_frame = inner.frame_buffer.lock().pop_front();

        maybe_frame.map(|frame| (frame.data, frame.width, frame.height))
    }

    pub fn buffered_len(&self) -> usize {
        self.read().frame_buffer.lock().len()
    }

    pub fn audio_buffer_level(&self) -> f32 {
        let inner = self.read();
        let ring = inner.audio_ring.lock();
        ring.available() as f32 / ring.capacity as f32
    }
}
