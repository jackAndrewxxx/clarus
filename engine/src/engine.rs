//! Scheduler: drives decode -> batch -> device -> (encode) for every request.
//!
//! One device graph serves one (batch size, H, W), so requests are grouped by exact image
//! shape and batches only use the configured `batch_sizes` (a short batch is padded by
//! repeating its last image; the padding results are dropped).
//!
//! * `Batching::Static`: batches are fixed before the run from the file list (per shape, in
//!   name order). Decodes are issued batch by batch, so batch composition, and therefore every
//!   output bit, is the same on every run.
//! * `Batching::Dynamic`: batches form from whatever has finished decoding; a full batch is
//!   sent at once, a partial one only while fewer than two batches are queued on the device
//!   (so it never idles) or when no more inputs can arrive.
//!
//! Slots are the unit of back-pressure: a request owns one slot from decode until its
//! outputs are consumed.

use crate::io::{ms, Done, IoPool, Job, SlotMem};
use crate::pyworker::PyHandle;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::time::Instant;

pub struct Request {
    pub name: String,
    pub lq: PathBuf,
    pub gt: Option<PathBuf>,
    pub out: Option<PathBuf>,
    /// (h, w) from the PNG header
    pub shape: (usize, usize),
    /// set when the request is refused before running (e.g. shape not in a fixed set)
    pub reject: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Batching {
    Static,
    Dynamic,
}

pub struct Config {
    /// graph batch sizes, ascending, non-empty
    pub batch_sizes: Vec<usize>,
    pub max_inflight: usize,
    pub io_threads: usize,
    pub slots: usize,
    pub metrics: bool,
    pub png_level: i32,
    pub verbose: bool,
    pub batching: Batching,
}

impl Config {
    /// (images to take, graph batch size) for `avail` waiting images of one shape.
    fn pick(&self, avail: usize) -> (usize, usize) {
        let max = *self.batch_sizes.last().unwrap();
        let avail = avail.min(max);
        match self.batch_sizes.iter().rev().find(|&&b| b <= avail) {
            Some(&b) => (b, b),
            None => (avail, self.batch_sizes[0]), // pad up to the smallest captured size
        }
    }
}

#[derive(Clone, Serialize, Default)]
pub struct ImageResult {
    pub name: String,
    pub h: usize,
    pub w: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub psnr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssim: Option<f64>,
    pub nonfinite: u64,
    pub decode_ms: f64,
    pub batch_ms: f64,
    pub batch: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Serialize, Default)]
pub struct RunStats {
    pub images: usize,
    pub failed: usize,
    pub wall_s: f64,
    pub images_per_s: f64,
    pub first_result_s: f64,
    pub batches: usize,
    pub padded_images: usize,
    pub mean_batch: f64,
    pub mean_batch_ms: f64,
    pub mean_decode_ms: f64,
    pub mean_encode_ms: f64,
    pub executor_blocked_s: f64,
}

struct State<'a> {
    reqs: &'a [Request],
    cfg: &'a Config,
    scale: usize,
    pool: IoPool,
    rx: Receiver<Done>,
    results: Vec<ImageResult>,
    free: Vec<usize>,
    /// decode order (request indices)
    order: Vec<usize>,
    next_decode: usize,
    /// slot of a decoded request (None: not decoded yet)
    slot_of: Vec<Option<usize>>,
    failed: Vec<bool>,
    // dynamic batching: decoded requests per shape
    ready: BTreeMap<(usize, usize), VecDeque<(usize, usize)>>,
    // static batching: planned batches (request indices, graph batch size)
    plan: Vec<(Vec<usize>, usize)>,
    next_plan: usize,
    inflight: HashMap<u64, Vec<(usize, usize)>>,
    decoding: usize,
    encoding: usize,
    done: usize,
    next_batch: u64,
    stats: RunStats,
    batch_ms_sum: f64,
    batch_items: usize,
    enc_ms_sum: f64,
    enc_n: usize,
    t0: Instant,
}

/// Static plan: per shape (in first-appearance order), consecutive requests in name order.
fn make_plan(reqs: &[Request], cfg: &Config) -> Vec<(Vec<usize>, usize)> {
    let mut buckets: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (i, r) in reqs.iter().enumerate() {
        if r.reject.is_none() {
            buckets.entry(r.shape).or_default().push(i);
        }
    }
    let mut plan = Vec::new();
    for (_, ids) in buckets {
        let mut rest: &[usize] = &ids;
        while !rest.is_empty() {
            let (take, size) = cfg.pick(rest.len());
            plan.push((rest[..take].to_vec(), size));
            rest = &rest[take..];
        }
    }
    plan.sort_by_key(|(ids, _)| ids[0]);
    plan
}

impl<'a> State<'a> {
    fn finish_one(&mut self, slot: usize) {
        self.free.push(slot);
        self.done += 1;
    }

    fn on_io(&mut self, ev: Done) {
        match ev {
            Done::Decoded { req, slot, h, w, ms, err } => {
                self.decoding -= 1;
                let r = &mut self.results[req];
                (r.decode_ms, r.h, r.w) = (ms, h, w);
                if let Some(e) = err {
                    eprintln!("[clarus] decode failed: {e}");
                    r.error = Some(e);
                    self.failed[req] = true;
                    self.finish_one(slot);
                } else {
                    self.slot_of[req] = Some(slot);
                    if self.cfg.batching == Batching::Dynamic {
                        self.ready.entry((h, w)).or_default().push_back((req, slot));
                    }
                }
            }
            Done::Encoded { req, slot, ms, err } => {
                self.encoding -= 1;
                self.enc_ms_sum += ms;
                self.enc_n += 1;
                if let Some(e) = err {
                    eprintln!("[clarus] encode failed: {e}");
                    self.results[req].error = Some(e);
                }
                self.finish_one(slot);
            }
        }
    }

    fn on_batch(&mut self, v: &Value) -> Result<(), String> {
        let id = v["id"].as_u64().ok_or("completion without id")?;
        let items = self.inflight.remove(&id).ok_or("unknown batch id")?;
        let lat = v["latency_ms"].as_f64().unwrap_or(0.0);
        self.batch_ms_sum += lat;
        self.stats.batches += 1;
        self.batch_items += items.len();
        let res = v["results"].as_array().ok_or("completion without results")?;
        let s = self.scale;
        for (&(req, slot), r) in items.iter().zip(res.iter()) {
            let out = &mut self.results[req];
            if self.cfg.metrics {
                // JSON has no Infinity: a perfect reconstruction (mse == 0) arrives as null
                out.psnr = Some(r["psnr"].as_f64().unwrap_or(f64::INFINITY));
                out.ssim = r["ssim"].as_f64();
            }
            out.nonfinite = r["nonfinite"].as_u64().unwrap_or(0);
            out.batch_ms = lat;
            out.batch = items.len();
            if let Some(path) = &self.reqs[req].out {
                let (h, w) = (out.h * s, out.w * s);
                self.pool.submit(Job::Encode { req, slot, h, w, path: path.clone(), level: self.cfg.png_level });
                self.encoding += 1;
            } else {
                self.finish_one(slot);
            }
        }
        if self.stats.first_result_s == 0.0 {
            self.stats.first_result_s = self.t0.elapsed().as_secs_f64();
        }
        Ok(())
    }

    fn feed_decoders(&mut self) -> bool {
        let mut any = false;
        while self.next_decode < self.order.len() {
            let Some(slot) = self.free.pop() else { break };
            let req = self.order[self.next_decode];
            let r = &self.reqs[req];
            let gt = if self.cfg.metrics { r.gt.clone() } else { None };
            self.pool.submit(Job::Decode { req, slot, lq: r.lq.clone(), gt });
            self.next_decode += 1;
            self.decoding += 1;
            any = true;
        }
        any
    }

    fn submit(&mut self, worker: &PyHandle, h: usize, w: usize, items: Vec<(usize, usize)>, size: usize)
              -> Result<(), String> {
        let id = self.next_batch;
        self.next_batch += 1;
        let mut slots: Vec<usize> = items.iter().map(|x| x.1).collect();
        let pad = size - slots.len();
        let last = *slots.last().unwrap();
        slots.extend(std::iter::repeat(last).take(pad)); // fixed-size graph: repeat the last image
        self.stats.padded_images += pad;
        let output = items.iter().any(|(r, _)| self.reqs[*r].out.is_some());
        let req = json!({"id": id, "h": h, "w": w, "slots": slots, "metrics": self.cfg.metrics, "output": output});
        worker.call("submit", &req.to_string())?;
        if self.cfg.verbose {
            eprintln!("[clarus] batch {id}: {h}x{w} x{} (+{pad} pad, in flight {})", items.len(),
                      self.inflight.len() + 1);
        }
        self.inflight.insert(id, items);
        Ok(())
    }

    fn dispatch(&mut self, worker: &PyHandle) -> Result<bool, String> {
        match self.cfg.batching {
            Batching::Static => self.dispatch_static(worker),
            Batching::Dynamic => self.dispatch_dynamic(worker),
        }
    }

    fn dispatch_static(&mut self, worker: &PyHandle) -> Result<bool, String> {
        let mut any = false;
        while self.inflight.len() < self.cfg.max_inflight && self.next_plan < self.plan.len() {
            let (ids, size) = self.plan[self.next_plan].clone();
            if !ids.iter().all(|&r| self.failed[r] || self.slot_of[r].is_some()) {
                break; // members are decoded in plan order, so later batches cannot be ready first
            }
            self.next_plan += 1;
            let items: Vec<(usize, usize)> =
                ids.iter().filter(|&&r| !self.failed[r]).map(|&r| (r, self.slot_of[r].unwrap())).collect();
            if items.is_empty() {
                continue;
            }
            let (h, w) = self.reqs[items[0].0].shape;
            self.submit(worker, h, w, items, size)?;
            any = true;
        }
        Ok(any)
    }

    fn dispatch_dynamic(&mut self, worker: &PyHandle) -> Result<bool, String> {
        let mut any = false;
        let inputs_pending = self.next_decode < self.order.len() || self.decoding > 0;
        let max = *self.cfg.batch_sizes.last().unwrap();
        while self.inflight.len() < self.cfg.max_inflight {
            let allow_partial = self.inflight.len() < 2 || !inputs_pending;
            let full = self.ready.iter().find(|(_, q)| q.len() >= max).map(|(k, _)| *k);
            let key = full.or_else(|| {
                if !allow_partial {
                    return None;
                }
                self.ready.iter().filter(|(_, q)| !q.is_empty()).max_by_key(|(_, q)| q.len()).map(|(k, _)| *k)
            });
            let Some((h, w)) = key else { break };
            let q = self.ready.get_mut(&(h, w)).unwrap();
            let (take, size) = self.cfg.pick(q.len());
            let items: Vec<(usize, usize)> = q.drain(..take).collect();
            self.submit(worker, h, w, items, size)?;
            any = true;
        }
        Ok(any)
    }
}

fn parse(s: &str, what: &str) -> Result<Value, String> {
    serde_json::from_str(s).map_err(|e| format!("bad JSON from worker.{what}: {e}: {s}"))
}

pub fn run(worker: &PyHandle, mem: SlotMem, reqs: &[Request], cfg: &Config)
           -> Result<(Vec<ImageResult>, RunStats), String> {
    let n = reqs.len();
    let (tx, rx) = channel::<Done>();
    let plan = if cfg.batching == Batching::Static { make_plan(reqs, cfg) } else { Vec::new() };
    let order: Vec<usize> = if cfg.batching == Batching::Static {
        plan.iter().flat_map(|(ids, _)| ids.iter().copied()).collect()
    } else {
        (0..n).filter(|&i| reqs[i].reject.is_none()).collect()
    };
    let mut results: Vec<ImageResult> = reqs
        .iter()
        .map(|r| ImageResult { name: r.name.clone(), h: r.shape.0, w: r.shape.1, ..Default::default() })
        .collect();
    let mut pre_done = 0;
    for (i, r) in reqs.iter().enumerate() {
        if let Some(e) = &r.reject {
            results[i].error = Some(e.clone());
            pre_done += 1;
        }
    }
    let mut st = State {
        reqs,
        cfg,
        scale: mem.scale,
        pool: IoPool::new(cfg.io_threads, mem, tx),
        rx,
        results,
        free: (0..cfg.slots).rev().collect(),
        order,
        next_decode: 0,
        slot_of: vec![None; n],
        failed: vec![false; n],
        ready: BTreeMap::new(),
        plan,
        next_plan: 0,
        inflight: HashMap::new(),
        decoding: 0,
        encoding: 0,
        done: pre_done,
        next_batch: 0,
        stats: RunStats::default(),
        batch_ms_sum: 0.0,
        batch_items: 0,
        enc_ms_sum: 0.0,
        enc_n: 0,
        t0: Instant::now(),
    };
    let mut blocked = 0.0f64;
    while st.done < n {
        let mut progressed = st.feed_decoders();
        while let Ok(ev) = st.rx.try_recv() {
            st.on_io(ev);
            progressed = true;
        }
        progressed |= st.dispatch(worker)?;
        if !st.inflight.is_empty() {
            let v = parse(&worker.call("poll", "")?, "poll")?;
            for b in v.as_array().into_iter().flatten() {
                st.on_batch(b)?;
                progressed = true;
            }
        }
        if st.done >= n || progressed {
            continue;
        }
        // nothing to do right now: block on the oldest device batch, else on IO
        let tb = Instant::now();
        if !st.inflight.is_empty() {
            let v = parse(&worker.call("wait", "")?, "wait")?;
            if !v.is_null() {
                st.on_batch(&v)?;
            }
        } else if st.decoding > 0 || st.encoding > 0 {
            match st.rx.recv() {
                Ok(ev) => st.on_io(ev),
                Err(_) => return Err("IO pool terminated unexpectedly".into()),
            }
        } else {
            return Err(format!("scheduler stalled with {}/{n} requests finished (slots {} < batch size?)",
                               st.done, cfg.slots));
        }
        blocked += ms(tb) / 1e3;
    }
    let State { pool, results, mut stats, batch_ms_sum, batch_items, enc_ms_sum, enc_n, t0, .. } = st;
    pool.shutdown();
    stats.wall_s = t0.elapsed().as_secs_f64();
    stats.images = n;
    stats.failed = results.iter().filter(|r| r.error.is_some()).count();
    stats.images_per_s = n as f64 / stats.wall_s;
    stats.mean_batch = batch_items as f64 / stats.batches.max(1) as f64;
    stats.mean_batch_ms = batch_ms_sum / stats.batches.max(1) as f64;
    stats.mean_decode_ms = results.iter().map(|r| r.decode_ms).sum::<f64>() / n.max(1) as f64;
    stats.mean_encode_ms = if enc_n > 0 { enc_ms_sum / enc_n as f64 } else { 0.0 };
    stats.executor_blocked_s = blocked;
    Ok((results, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(sizes: &[usize]) -> Config {
        Config { batch_sizes: sizes.to_vec(), max_inflight: 3, io_threads: 1, slots: 8, metrics: false,
                 png_level: 1, verbose: false, batching: Batching::Static }
    }

    fn req(i: usize, shape: (usize, usize)) -> Request {
        Request { name: format!("{i}"), lq: PathBuf::new(), gt: None, out: None, shape, reject: None }
    }

    #[test]
    fn pick_pads_to_smallest_size() {
        assert_eq!(cfg(&[1, 2]).pick(5), (2, 2));
        assert_eq!(cfg(&[1, 2]).pick(1), (1, 1));
        assert_eq!(cfg(&[2, 4]).pick(3), (2, 2));
        assert_eq!(cfg(&[2, 4]).pick(1), (1, 2));
        assert_eq!(cfg(&[4]).pick(3), (3, 4));
    }

    #[test]
    fn static_plan_groups_by_shape_in_order() {
        let a = (8, 8);
        let b = (4, 4);
        let reqs = vec![req(0, a), req(1, b), req(2, a), req(3, a), req(4, b)];
        let plan = make_plan(&reqs, &cfg(&[1, 2]));
        assert_eq!(plan, vec![(vec![0, 2], 2), (vec![1, 4], 2), (vec![3], 1)]);
    }
}
