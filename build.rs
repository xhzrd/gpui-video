// No build configuration needed for dynamic FFmpeg linking
// The ffmpeg-next crate handles FFmpeg library discovery automatically
// Users must have FFmpeg libraries installed on their system

fn main() {
    // Print cargo metadata for documentation purposes
    println!("cargo:rerun-if-changed=build.rs");

    // Note: FFmpeg libraries are dynamically linked
    // Ensure FFmpeg 4.0+ is installed on your system:
    // - macOS: brew install ffmpeg
    // - Ubuntu/Debian: apt-get install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
    // - Windows: Download FFmpeg shared libraries from ffmpeg.org
}
