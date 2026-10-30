//! Pinned host slots and the IO worker pool (PNG decode into slots, PNG encode from slots).

use crate::png;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::time::Instant;

/// Plane index inside a slot.
pub const LQ: usize = 0;
pub const GT: usize = 1;
pub const OUT: usize = 2;

/// Raw view of the (page-locked) host buffer owned by the Python worker: `slots` slots of
/// `slot_stride` bytes, each holding three planes (input, ground truth, output) at the
/// offsets / capacities reported by `worker.info()`. The scheduler hands each slot to
/// exactly one owner at a time (IO thread, device batch, or encoder), which makes the
/// aliasing sound.
#[derive(Clone, Copy)]
pub struct SlotMem {
    pub base: usize,
    pub slot_stride: usize,
    pub planes: [(usize, usize); 3],
    /// output H, W = input H, W * scale
    pub scale: usize,
}

impl SlotMem {
    /// # Safety
    /// Caller must own `slot` (see type docs).
    pub unsafe fn plane(&self, slot: usize, plane: usize) -> &'static mut [u8] {
        let (off, cap) = self.planes[plane];
        let p = (self.base + slot * self.slot_stride + off) as *mut u8;
        std::slice::from_raw_parts_mut(p, cap)
    }
}

pub enum Job {
    Decode { req: usize, slot: usize, lq: PathBuf, gt: Option<PathBuf> },
    /// h, w: output image size
    Encode { req: usize, slot: usize, h: usize, w: usize, path: PathBuf, level: i32 },
}

pub enum Done {
    Decoded { req: usize, slot: usize, h: usize, w: usize, ms: f64, err: Option<String> },
    Encoded { req: usize, slot: usize, ms: f64, err: Option<String> },
}

pub struct IoPool {
    pool: rayon::ThreadPool,
    mem: SlotMem,
    tx: Sender<Done>,
}

fn decode_one(mem: &SlotMem, slot: usize, lq: &PathBuf, gt: &Option<PathBuf>) -> Result<(usize, usize), String> {
    let data = std::fs::read(lq).map_err(|e| format!("{}: {e}", lq.display()))?;
    let (h, w) = png::decode_rgb8(&data, unsafe { mem.plane(slot, LQ) }).map_err(|e| format!("{}: {e}", lq.display()))?;
    if let Some(g) = gt {
        let data = std::fs::read(g).map_err(|e| format!("{}: {e}", g.display()))?;
        let (gh, gw) =
            png::decode_rgb8(&data, unsafe { mem.plane(slot, GT) }).map_err(|e| format!("{}: {e}", g.display()))?;
        let s = mem.scale;
        if (gh, gw) != (h * s, w * s) {
            return Err(format!("{}: GT size {gw}x{gh} != expected {}x{} (input {w}x{h} x scale {s})",
                               g.display(), w * s, h * s));
        }
    }
    Ok((h, w))
}

fn run_job(mem: &SlotMem, job: Job) -> Done {
    let t0 = Instant::now();
    match job {
        Job::Decode { req, slot, lq, gt } => match decode_one(mem, slot, &lq, &gt) {
            Ok((h, w)) => Done::Decoded { req, slot, h, w, ms: ms(t0), err: None },
            Err(e) => Done::Decoded { req, slot, h: 0, w: 0, ms: ms(t0), err: Some(e) },
        },
        Job::Encode { req, slot, h, w, path, level } => {
            let rgb = unsafe { &mem.plane(slot, OUT)[..h * w * 3] };
            let r = png::encode_rgb8(rgb, w, h, level)
                .and_then(|bytes| std::fs::write(&path, bytes).map_err(|e| e.to_string()))
                .map_err(|e| format!("{}: {e}", path.display()));
            Done::Encoded { req, slot, ms: ms(t0), err: r.err() }
        }
    }
}

impl IoPool {
    pub fn new(threads: usize, mem: SlotMem, tx: Sender<Done>) -> IoPool {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|i| format!("io-{i}"))
            .build()
            .expect("build IO thread pool");
        IoPool { pool, mem, tx }
    }

    pub fn submit(&self, job: Job) {
        let (mem, tx) = (self.mem, self.tx.clone());
        // FIFO spawn: decodes start in request order, which keeps batches filling evenly
        self.pool.spawn_fifo(move || {
            let _ = tx.send(run_job(&mem, job));
        });
    }

    /// The scheduler only calls this after every job has reported back, so dropping
    /// the pool (which lets idle workers exit) cannot lose work.
    pub fn shutdown(self) {
        drop(self.pool);
    }
}

pub fn ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}
