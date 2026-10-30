//! PNG IO on top of the `png` crate: decode any PNG to packed RGB8, encode RGB8.

use std::cell::RefCell;
use std::io::{BufReader, Cursor};
use std::path::Path;

use png::{BitDepth, ColorType, Compression, Transformations};

thread_local! {
    static SCRATCH: RefCell<Vec<u8>> = RefCell::new(Vec::new());
}

/// (height, width) from the PNG header only (stops before the first IDAT).
pub fn read_shape(path: &Path) -> Result<(usize, usize), String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let reader = png::Decoder::new(BufReader::new(f))
        .read_info()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let info = reader.info();
    Ok((info.height as usize, info.width as usize))
}

/// Decode `data` into `out` as packed RGB8 (H x W x 3); returns (height, width).
///
/// Matches cv2.IMREAD_COLOR: palettes and low bit depths are expanded, 16-bit samples
/// reduced to their high byte, gray replicated to three channels, alpha dropped.
/// Interlaced (Adam7) images are supported.
pub fn decode_rgb8(data: &[u8], out: &mut [u8]) -> Result<(usize, usize), String> {
    let mut dec = png::Decoder::new(Cursor::new(data));
    dec.set_transformations(Transformations::normalize_to_color8());
    let mut reader = dec.read_info().map_err(|e| e.to_string())?;
    let (w, h) = {
        let info = reader.info();
        (info.width as usize, info.height as usize)
    };
    let need = w * h * 3;
    if out.len() < need {
        return Err(format!("image {w}x{h} exceeds slot capacity"));
    }
    let (color, depth) = reader.output_color_type();
    if depth != BitDepth::Eight {
        return Err(format!("unexpected output bit depth {depth:?}"));
    }
    if color == ColorType::Rgb {
        // already the slot layout: decode straight into pinned memory
        reader.next_frame(&mut out[..need]).map_err(|e| e.to_string())?;
        return Ok((h, w));
    }
    let size = reader.output_buffer_size().ok_or("image too large")?;
    SCRATCH.with(|s| {
        let mut buf = s.borrow_mut();
        buf.resize(size, 0);
        let fi = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
        let ch = match fi.color_type {
            ColorType::Grayscale => 1,
            ColorType::GrayscaleAlpha => 2,
            ColorType::Rgba => 4,
            c => return Err(format!("unexpected output colour type {c:?}")),
        };
        let dst = &mut out[..need];
        for y in 0..h {
            let line = &buf[y * fi.line_size..y * fi.line_size + w * ch];
            let o = &mut dst[y * w * 3..(y + 1) * w * 3];
            for (px, src) in o.chunks_exact_mut(3).zip(line.chunks_exact(ch)) {
                if ch >= 3 {
                    px.copy_from_slice(&src[..3]);
                } else {
                    px.fill(src[0]);
                }
            }
        }
        Ok((h, w))
    })
}

/// Encode packed RGB8 as PNG. `level`: 0 = store, 1 = fast (default), 2-6 = balanced, >=7 = high.
pub fn encode_rgb8(rgb: &[u8], w: usize, h: usize, level: i32) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(w * h * 3 / 2);
    {
        let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
        enc.set_color(ColorType::Rgb);
        enc.set_depth(BitDepth::Eight);
        enc.set_compression(match level {
            i32::MIN..=0 => Compression::NoCompression,
            1 => Compression::Fast,
            2..=6 => Compression::Balanced,
            _ => Compression::High,
        });
        let mut writer = enc.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(&rgb[..w * h * 3]).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let (w, h) = (37, 23);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i * 7919) % 251) as u8).collect();
        for level in [0, 1, 6, 9] {
            let png = encode_rgb8(&rgb, w, h, level).unwrap();
            let mut out = vec![0u8; w * h * 3];
            assert_eq!(decode_rgb8(&png, &mut out).unwrap(), (h, w));
            assert_eq!(out, rgb);
        }
    }

    #[test]
    fn gray_and_rgba_to_rgb() {
        let (w, h) = (5, 3);
        let gray: Vec<u8> = (0..w * h).map(|i| (i * 13) as u8).collect();
        let rgba: Vec<u8> = (0..w * h * 4).map(|i| (i * 7) as u8).collect();
        for (ct, data) in [(ColorType::Grayscale, &gray), (ColorType::Rgba, &rgba)] {
            let mut buf = Vec::new();
            {
                let mut enc = png::Encoder::new(&mut buf, w as u32, h as u32);
                enc.set_color(ct);
                enc.set_depth(BitDepth::Eight);
                enc.write_header().unwrap().write_image_data(data).unwrap();
            }
            let mut out = vec![0u8; w * h * 3];
            decode_rgb8(&buf, &mut out).unwrap();
            for p in 0..w * h {
                let want: [u8; 3] = if ct == ColorType::Grayscale {
                    [gray[p]; 3]
                } else {
                    [rgba[p * 4], rgba[p * 4 + 1], rgba[p * 4 + 2]]
                };
                assert_eq!(out[p * 3..p * 3 + 3], want);
            }
        }
    }
}
