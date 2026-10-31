//! Pinned host slots and the IO worker pool (PNG decode into slots, PNG encode from slots).

use crate::png;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
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
///
/// Preconditions for building `&mut [u8]` views over this memory (see `plane`):
///   * initialized: the worker allocates it zero-filled (`device.host_buffer` ->
///     `torch.zeros`), so every byte is a valid `u8` even before the first decode;
///   * alive: the Python worker keeps the buffer until `shutdown()` has returned, and the
///     engine calls that only after `IoPool::shutdown` has joined every IO job;
///   * in bounds: `check` validates the reported layout before the first job.
#[derive(Clone, Copy)]
pub struct SlotMem {
    pub base: usize,
    pub slot_stride: usize,
    pub planes: [(usize, usize); 3],
    /// output H, W = input H, W * scale
    pub scale: usize,
}

impl SlotMem {
    /// Layout sanity check, done once when the slots are received from the worker.
    pub fn check(&self, slots: usize, reported_slots: usize) -> Result<(), String> {
        if self.base == 0 || reported_slots < slots {
            return Err(format!("worker host buffer: base {:#x}, {reported_slots} slots (need {slots})", self.base));
        }
        for (i, &(off, cap)) in self.planes.iter().enumerate() {
            if off.checked_add(cap).map_or(true, |end| end > self.slot_stride) {
                return Err(format!("worker host buffer: plane {i} [{off}, +{cap}) exceeds slot stride {}",
                                   self.slot_stride));
            }
        }
        Ok(())
    }

    /// # Safety
    /// Caller must own `slot` (see type docs) and `slot` must be < the slot count passed to
    /// `check`. The returned slice must not be used after the IO job that requested it ends.
    pub unsafe fn plane(&self, slot: usize, plane: usize) -> &'static mut [u8] {
        let (off, cap) = self.planes[plane];
        let p = (self.base + slot * self.slot_stride + off) as *mut u8;
        // SAFETY: [p, p + cap) lies inside the worker's live host buffer (`check`), the bytes
        // are initialized (zero-filled allocation) and no other reference to this plane
        // exists while the caller owns the slot.
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
    /// jobs submitted but not yet finished touching slot memory
    pending: Arc<(Mutex<usize>, Condvar)>,
}

/// Decrements the pending-job count when a job ends (also if it unwinds).
struct JobGuard(Arc<(Mutex<usize>, Condvar)>);

impl Drop for JobGuard {
    fn drop(&mut self) {
        let (n, cv) = &*self.0;
        let mut n = n.lock().unwrap_or_else(|e| e.into_inner());
        *n -= 1;
        cv.notify_all();
    }
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
        IoPool { pool, mem, tx, pending: Arc::new((Mutex::new(0), Condvar::new())) }
    }

    pub fn submit(&self, job: Job) {
        let (mem, tx) = (self.mem, self.tx.clone());
        *self.pending.0.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        let guard = JobGuard(self.pending.clone());
        // FIFO spawn: decodes start in request order, which keeps batches filling evenly
        self.pool.spawn_fifo(move || {
            let _guard = guard;
            let _ = tx.send(run_job(&mem, job));
        });
    }

    /// Block until every submitted job has finished (and stopped touching slot memory).
    fn wait_idle(&self) {
        let (n, cv) = &*self.pending;
        let mut n = n.lock().unwrap_or_else(|e| e.into_inner());
        while *n > 0 {
            n = cv.wait(n).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Wait for all in-flight jobs, then stop the threads. Dropping the rayon pool alone
    /// would not join running jobs, which could then outlive the host buffer they write.
    pub fn shutdown(self) {
        drop(self); // Drop waits
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        self.wait_idle();
    }
}

pub fn ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::channel;
    use std::time::Duration;

    #[test]
    fn check_rejects_bad_layout() {
        let mem = SlotMem { base: 4096, slot_stride: 100, planes: [(0, 30), (30, 30), (60, 40)], scale: 1 };
        assert!(mem.check(4, 4).is_ok());
        assert!(mem.check(4, 3).is_err());
        assert!(SlotMem { planes: [(0, 30), (30, 30), (60, 41)], ..mem }.check(4, 4).is_err());
        assert!(SlotMem { base: 0, ..mem }.check(4, 4).is_err());
    }

    /// A decode blocked on a FIFO must hold up `shutdown` until it finishes.
    #[cfg(unix)]
    #[test]
    fn shutdown_waits_for_running_job() {
        let dir = std::env::temp_dir().join(format!("clarus_io_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fifo = dir.join("blocked.png");
        let _ = std::fs::remove_file(&fifo);
        assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
        let mut buf = vec![0u8; 3 * 64];
        let mem = SlotMem { base: buf.as_mut_ptr() as usize, slot_stride: 64 * 3, planes: [(0, 64), (64, 64), (128, 64)],
                            scale: 1 };
        let (tx, rx) = channel();
        let pool = IoPool::new(1, mem, tx);
        pool.submit(Job::Decode { req: 0, slot: 0, lq: fifo.clone(), gt: None });
        let released = Arc::new(AtomicBool::new(false));
        let writer = {
            let (fifo, released) = (fifo.clone(), released.clone());
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                released.store(true, Ordering::SeqCst);
                std::fs::write(&fifo, b"not a png").unwrap(); // unblocks the reader
            })
        };
        pool.shutdown();
        assert!(released.load(Ordering::SeqCst), "shutdown returned while the job was still running");
        match rx.try_recv() {
            Ok(Done::Decoded { err: Some(_), .. }) => {}
            _ => panic!("expected a failed decode report"),
        }
        writer.join().unwrap();
        drop(buf);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
