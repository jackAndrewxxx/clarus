//! clarus: evaluation / inference engine for image-restoration networks.
//!
//! Rust owns the process: CLI + YAML config, dataset scan, PNG IO thread pool,
//! shape-bucketed batch scheduler and reporting. The network runs in an embedded CPython
//! model worker (clarus, hosted via PyO3): a registered model built on per-op dispatched
//! backends (Triton kernels on CUDA, PyTorch elsewhere), replayed from device graphs.

mod config;
mod engine;
mod io;
mod png;
mod pyworker;

use engine::{Config, Request};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "\
clarus — inference / eval engine for image-restoration networks

USAGE:
  clarus eval  --lq DIR --gt DIR [--save DIR] [--report FILE] [options]
  clarus infer --input DIR --output DIR [--report FILE] [options]
  clarus png-hash DIR

CONFIG:
  --config FILE        engine YAML (default: $CLARUS_CONFIG or configs/nafnet_gopro_width64.yaml)
  --set KEY=VALUE      override any config key, e.g. --set runtime.op_overrides.selective_scan=torch

SHORTCUTS (same as --set; PATH / names are taken literally, not parsed as YAML):
  --ckpt PATH          model.checkpoint
  --device D           runtime.device (\"1\" keeps the device type, or e.g. cuda:1, npu:0, cpu)
  --dtype fp16|bf16|fp32  runtime.dtype
  --backends a,b       runtime.backends (op dispatch order), e.g. triton,torch or torch
  --no-graph           runtime.graphs=false
  --safe               model.options.residual_fp32_from_level=0 (fp32 residual stream)
  --batch-sizes 1,2    scheduler.batch_sizes (captured graph batch sizes)
  --max-batch N        scheduler.batch_sizes=[1,N]
  --batching static|dynamic  scheduler.batching
  --shapes HxW,...     scheduler.shapes (fixed shape set, captured at start-up)
  --inflight N         scheduler.inflight
  --io-threads N       io.threads
  --png-level N        io.png_level: 0 none, 1 fast (default), 2-6 balanced, 7+ high

RUN:
  --limit N            only the first N images
  --verbose            log every batch
  --allow-failures     exit 0 even if some images failed (default: exit 1)
";

/// A config change from the command line, applied in order after loading the YAML.
#[derive(Debug, PartialEq)]
enum Override {
    /// `--set key=value`: value parsed as YAML
    Yaml(String),
    /// shortcut flag: `key` = literal value (no YAML parsing, so paths keep `#`, `:` etc.)
    Value(&'static str, Value),
    /// `--device`: resolved against the configured device type
    Device(String),
}

struct Args {
    cmd: String,
    lq: Option<PathBuf>,
    gt: Option<PathBuf>,
    out: Option<PathBuf>,
    report: Option<PathBuf>,
    config: String,
    sets: Vec<Override>,
    limit: Option<usize>,
    verbose: bool,
    allow_failures: bool,
}

fn parse_args(argv: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut it = argv.into_iter();
    let cmd = it.next().ok_or(USAGE)?;
    if cmd == "-h" || cmd == "--help" {
        return Err(USAGE.into());
    }
    let default_config = std::env::var("CLARUS_CONFIG").unwrap_or_else(|_| env!("CLARUS_BUILD_CONFIG").to_string());
    let mut a = Args { cmd, lq: None, gt: None, out: None, report: None, config: default_config, sets: vec![],
                       limit: None, verbose: false, allow_failures: false };
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        let list = |s: String| format!("[{s}]");
        let names = |s: String| json!(s.split(',').map(str::trim).filter(|p| !p.is_empty()).collect::<Vec<_>>());
        use Override::{Value as Lit, Yaml};
        match flag.as_str() {
            "--lq" | "--input" => a.lq = Some(val()?.into()),
            "--gt" => a.gt = Some(val()?.into()),
            "--save" | "--output" => a.out = Some(val()?.into()),
            "--report" => a.report = Some(val()?.into()),
            "--config" => a.config = val()?,
            "--set" => a.sets.push(Yaml(val()?)),
            "--ckpt" => a.sets.push(Lit("model.checkpoint", json!(val()?))),
            "--device" => a.sets.push(Override::Device(val()?)),
            "--dtype" => a.sets.push(Lit("runtime.dtype", json!(val()?))),
            "--backends" => a.sets.push(Lit("runtime.backends", names(val()?))),
            "--no-graph" => a.sets.push(Yaml("runtime.graphs=false".into())),
            "--safe" => a.sets.push(Yaml("model.options.residual_fp32_from_level=0".into())),
            "--batch-sizes" => a.sets.push(Yaml(format!("scheduler.batch_sizes={}", list(val()?)))),
            "--max-batch" => a.sets.push(Yaml(format!("scheduler.batch_sizes=[1,{}]", val()?))),
            "--batching" => a.sets.push(Lit("scheduler.batching", json!(val()?))),
            "--shapes" => a.sets.push(Lit("scheduler.shapes", config::parse_shapes(&val()?)?)),
            "--inflight" => a.sets.push(Yaml(format!("scheduler.inflight={}", val()?))),
            "--io-threads" => a.sets.push(Yaml(format!("io.threads={}", val()?))),
            "--png-level" => a.sets.push(Yaml(format!("io.png_level={}", val()?))),
            "--limit" => a.limit = Some(val()?.parse().map_err(|_| "bad --limit")?),
            "--verbose" => a.verbose = true,
            "--allow-failures" => a.allow_failures = true,
            "-h" | "--help" => return Err(USAGE.into()),
            f => return Err(format!("unknown option {f}\n\n{USAGE}")),
        }
    }
    match a.cmd.as_str() {
        "eval" if a.lq.is_none() || a.gt.is_none() => Err(format!("eval needs --lq and --gt\n\n{USAGE}")),
        "infer" if a.lq.is_none() || a.out.is_none() => Err(format!("infer needs --input and --output\n\n{USAGE}")),
        "eval" | "infer" => Ok(a),
        c => Err(format!("unknown command {c}\n\n{USAGE}")),
    }
}

fn list_pngs(dir: &Path) -> Result<Vec<String>, String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.to_ascii_lowercase().ends_with(".png"))
        .collect();
    v.sort();
    Ok(v)
}

/// `clarus png-hash DIR`: checksum of every decoded RGB8 image (decoder verification).
fn png_hash(dir: &Path) -> Result<(), String> {
    let lines: Vec<String> = list_pngs(dir)?
        .par_iter()
        .map(|name| -> Result<String, String> {
            let data = std::fs::read(dir.join(name)).map_err(|e| e.to_string())?;
            let (h, w) = png::read_shape(&dir.join(name))?;
            let mut rgb = vec![0u8; w * h * 3];
            png::decode_rgb8(&data, &mut rgb).map_err(|e| format!("{name}: {e}"))?;
            // position-weighted checksum (vectorisable in numpy for the cv2 reference)
            let mut hsh: u64 = 0;
            for (i, b) in rgb.iter().enumerate() {
                let wt = ((i as u64).wrapping_mul(2654435761) & 0xffff_ffff) + 1;
                hsh = hsh.wrapping_add((*b as u64).wrapping_mul(wt));
            }
            Ok(format!("{name} {w}x{h} {hsh:016x}"))
        })
        .collect::<Result<_, _>>()?;
    for l in lines {
        println!("{l}");
    }
    Ok(())
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn main() {
    match real_main() {
        Ok(0) => {}
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn apply_overrides(s: &mut config::Settings, sets: &[Override]) -> Result<(), String> {
    for o in sets {
        match o {
            Override::Yaml(a) => s.set(a)?,
            Override::Value(key, v) => s.set_value(key, v.clone()),
            Override::Device(d) => {
                let spec = config::device_spec(&s.raw["runtime"]["device"], d);
                s.set_value("runtime.device", json!(spec));
            }
        }
    }
    config::check_sections(&s.raw)
}

fn load_settings(args: &Args) -> Result<config::Settings, String> {
    let mut s = config::load(&args.config)?;
    apply_overrides(&mut s, &args.sets)?;
    Ok(s)
}

/// Process exit code on success: 0, or 1 if any image failed (unless --allow-failures).
fn real_main() -> Result<i32, String> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() <= 1 || argv.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(0);
    }
    if argv.len() == 3 && argv[1] == "png-hash" {
        return png_hash(Path::new(&argv[2])).map(|_| 0);
    }
    let args = parse_args(argv.into_iter().skip(1))?;
    let settings = load_settings(&args)?;
    let t_start = Instant::now();
    let metrics = args.cmd == "eval";
    let batch_sizes = settings.batch_sizes()?;
    let inflight = settings.usize_or("scheduler", "inflight", 3)?.max(1);
    let fixed_shapes = settings.shapes()?;
    let unknown = settings.unknown_shape()?;
    let batching = settings.batching()?;
    let io_threads = settings.usize_or("io", "threads", 16)?.max(1);
    let png_level = settings.usize_or("io", "png_level", 1)? as i32;

    let lq_dir = args.lq.clone().unwrap();
    let mut names = list_pngs(&lq_dir)?;
    if let Some(l) = args.limit {
        names.truncate(l);
    }
    if names.is_empty() {
        return Err(format!("no .png files in {}", lq_dir.display()));
    }
    if let Some(o) = &args.out {
        std::fs::create_dir_all(o).map_err(|e| format!("{}: {e}", o.display()))?;
    }
    // shape scan (PNG header only) -> graph warm-up set, slot size, fixed-shape admission
    let found: Vec<(usize, usize)> =
        names.par_iter().map(|n| png::read_shape(&lq_dir.join(n))).collect::<Result<_, _>>()?;
    let allowed: Option<BTreeSet<(usize, usize)>> = fixed_shapes.as_ref().map(|v| v.iter().copied().collect());
    let reqs: Vec<Request> = names
        .iter()
        .zip(found.iter())
        .map(|(n, &shape)| Request {
            name: n.clone(),
            lq: lq_dir.join(n),
            gt: args.gt.as_ref().map(|g| g.join(n)),
            out: args.out.as_ref().map(|o| o.join(n)),
            shape,
            reject: match &allowed {
                Some(set) if !set.contains(&shape) && unknown == config::UnknownShape::Reject => Some(format!(
                    "shape {}x{} (HxW) not in scheduler.shapes (set scheduler.unknown_shape: lazy to admit it)",
                    shape.0, shape.1)),
                _ => None,
            },
        })
        .collect();
    if metrics {
        if let Some(missing) = reqs.iter().find(|r| !r.gt.as_ref().unwrap().exists()) {
            return Err(format!("missing GT for {}", missing.name));
        }
    }
    let admitted: BTreeSet<(usize, usize)> = reqs.iter().filter(|r| r.reject.is_none()).map(|r| r.shape).collect();
    let warm_shapes: BTreeSet<(usize, usize)> = allowed.clone().unwrap_or_else(|| admitted.clone());
    let max_pixels = warm_shapes.iter().chain(admitted.iter()).map(|(h, w)| h * w).max().unwrap_or(1);
    let slots = batch_sizes.last().unwrap() * (inflight + 2);
    let t_scan = t_start.elapsed().as_secs_f64();
    // embedded model worker
    let py = pyworker::Interpreter::start(
        &env_or("CLARUS_PYTHONHOME", env!("CLARUS_BUILD_PYTHONHOME")),
        &env_or("CLARUS_SITE", env!("CLARUS_BUILD_SITE")),
        &env_or("CLARUS_PYTHONPATH", env!("CLARUS_BUILD_PYDIR")),
    )?;
    let wcfg = json!({
        "model": settings.raw["model"], "runtime": settings.raw["runtime"],
        "worker": {"slots": slots, "max_pixels": max_pixels, "batch_sizes": batch_sizes, "inflight": inflight},
    });
    let worker = py.create("clarus.worker", "create", &wcfg.to_string())?;
    let info: Value = serde_json::from_str(&worker.call("info", "")?).map_err(|e| e.to_string())?;
    let t_load = t_start.elapsed().as_secs_f64();
    let shape_list: Vec<[usize; 2]> = warm_shapes.iter().map(|&(h, w)| [h, w]).collect();
    let warm = worker.call("warmup", &json!({"shapes": shape_list, "metrics": metrics}).to_string())?;
    let t_warm = t_start.elapsed().as_secs_f64();
    let model_name = info["model"]["name"].as_str().unwrap_or("?").to_string();
    eprintln!(
        "[clarus] {} {} on {} | {} images, shapes {:?}, batch sizes {:?}, graphs {} | load {:.1}s, warm-up {:.1}s",
        args.cmd, model_name, info["device"].as_str().unwrap_or("?"), reqs.len(), shape_list, batch_sizes,
        info["graphs"], t_load - t_scan, t_warm - t_load
    );
    let rejected = reqs.iter().filter(|r| r.reject.is_some()).count();
    if rejected > 0 {
        eprintln!("[clarus] {rejected} images rejected: shape not in scheduler.shapes");
    }

    let plane = |k: &str| -> (usize, usize) {
        let p = &info["planes"][k];
        (p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize)
    };
    let mem = io::SlotMem {
        base: info["slot_base"].as_u64().unwrap() as usize,
        slot_stride: info["slot_stride"].as_u64().unwrap() as usize,
        planes: [plane("lq"), plane("gt"), plane("out")],
        scale: info["scale"].as_u64().unwrap_or(1) as usize,
    };
    // IO threads write into these slots through raw pointers (see io::SlotMem for the
    // initialized / exclusive-ownership contract); validate the layout before the first job
    mem.check(slots, info["slots"].as_u64().unwrap_or(0) as usize)?;
    let cfg = Config {
        batch_sizes: batch_sizes.clone(),
        max_inflight: inflight,
        io_threads,
        slots,
        metrics,
        png_level,
        verbose: args.verbose,
        batching,
    };
    // Cleanup order on success and error alike: `run` stops dispatching and joins every IO
    // job, then the worker drains its device queue and synchronizes, and only after that is
    // the worker (owner of the host slot memory) dropped, at the end of this function.
    let run = engine::run(&worker, mem, &reqs, &cfg);
    let fin = worker.call("shutdown", "");
    let (results, stats) = match (run, fin.as_ref()) {
        (Ok(r), _) => r,
        (Err(e), Err(fe)) => return Err(format!("{e}\n(worker shutdown also failed: {fe})")),
        (Err(e), Ok(_)) => return Err(e),
    };
    let fin = fin?;
    let total = t_start.elapsed().as_secs_f64();
    let ok: Vec<_> = results.iter().filter(|r| r.error.is_none()).collect();
    let mean = |f: &dyn Fn(&engine::ImageResult) -> Option<f64>| -> Option<f64> {
        let v: Vec<f64> = ok.iter().filter_map(|r| f(r)).collect();
        if v.is_empty() { None } else { Some(v.iter().sum::<f64>() / v.len() as f64) }
    };
    let psnr = mean(&|r| r.psnr);
    let ssim = mean(&|r| r.ssim);
    let nonfinite: u64 = results.iter().map(|r| r.nonfinite).sum();
    println!("==== clarus {} ({model_name}) ====", args.cmd);
    println!("images            {} ({} failed)", stats.images, stats.failed);
    if let (Some(p), Some(s)) = (psnr, ssim) {
        println!("PSNR / SSIM       {p:.4} / {s:.4}   (psnr {p:.6}, ssim {s:.6})");
    }
    println!("pipeline wall     {:.2} s  -> {:.2} img/s ({:.1} ms/img)", stats.wall_s, stats.images_per_s,
             1e3 / stats.images_per_s);
    println!("end-to-end wall   {total:.2} s (scan {t_scan:.2}, load {:.2}, warm-up {:.2})", t_load - t_scan,
             t_warm - t_load);
    println!("batches           {} ({:?}, mean size {:.2}, {} padded, mean device latency {:.1} ms)",
             stats.batches, cfg.batching, stats.mean_batch, stats.padded_images, stats.mean_batch_ms);
    println!("decode / encode   {:.1} / {:.1} ms per image ({} IO threads)", stats.mean_decode_ms,
             stats.mean_encode_ms, cfg.io_threads);
    let fallbacks: Vec<String> = info_ops_fallbacks(&fin);
    if !fallbacks.is_empty() {
        println!("op fallbacks      {}", fallbacks.join(", "));
    }
    if nonfinite > 0 {
        eprintln!("WARNING: {nonfinite} non-finite output values; re-run those inputs with --safe");
    }
    if let Some(path) = &args.report {
        let report = json!({
            "command": args.cmd, "config": settings.path, "device": info["device"], "model": info["model"],
            "settings": settings.raw, "psnr": psnr, "ssim": ssim, "nonfinite": nonfinite,
            "timing": {"total_s": total, "scan_s": t_scan, "load_s": t_load - t_scan, "warmup_s": t_warm - t_load,
                       "warmup": serde_json::from_str::<Value>(&warm).unwrap_or(Value::Null)},
            "stats": stats, "worker": serde_json::from_str::<Value>(&fin).unwrap_or(Value::Null),
            "images": results,
        });
        std::fs::write(path, serde_json::to_string_pretty(&report).unwrap())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        println!("report            {}", path.display());
    }
    if stats.failed > 0 {
        eprintln!("[clarus] {} of {} images failed{}", stats.failed, stats.images,
                  if args.allow_failures { " (--allow-failures: exit 0)" } else { "" });
        if !args.allow_failures {
            return Ok(1);
        }
    }
    Ok(0)
}

/// Ops not served by the first configured backend, e.g. "pixel_shuffle->torch".
fn info_ops_fallbacks(shutdown_json: &str) -> Vec<String> {
    let v: Value = serde_json::from_str(shutdown_json).unwrap_or(Value::Null);
    let first = v["ops"]["backends"][0].as_str().unwrap_or("").to_string();
    let mut out = Vec::new();
    if let Some(ops) = v["ops"]["ops"].as_object() {
        for (op, by) in ops {
            for b in by.as_object().into_iter().flat_map(|m| m.keys()) {
                if *b != first {
                    out.push(format!("{op}->{b}"));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Args {
        parse_args(v.iter().map(|s| s.to_string())).unwrap()
    }

    #[test]
    fn string_shortcuts_are_literal() {
        let dir = std::env::temp_dir().join(format!("clarus_main_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("c.yaml");
        std::fs::write(&cfg, "runtime:\n  device: npu:0\n").unwrap();
        let a = args(&["infer", "--input", "i", "--output", "o", "--config", cfg.to_str().unwrap(),
                       "--ckpt", "weights #final.pth", "--dtype", "bf16", "--backends", "triton, torch",
                       "--batching", "dynamic", "--shapes", "720x1280", "--device", "1", "--inflight", "4",
                       "--set", "model.options.x=7 # comment", "--set", "model.name=false"]);
        let mut s = config::load(cfg.to_str().unwrap()).unwrap();
        apply_overrides(&mut s, &a.sets).unwrap();
        assert_eq!(s.raw["model"]["checkpoint"], json!("weights #final.pth"));
        assert_eq!(s.raw["runtime"]["dtype"], json!("bf16"));
        assert_eq!(s.raw["runtime"]["backends"], json!(["triton", "torch"]));
        assert_eq!(s.raw["runtime"]["device"], json!("npu:1"));
        assert_eq!(s.raw["scheduler"]["batching"], json!("dynamic"));
        assert_eq!(s.raw["scheduler"]["shapes"], json!([[720, 1280]]));
        assert_eq!(s.raw["scheduler"]["inflight"], json!(4));
        // --set stays YAML-typed
        assert_eq!(s.raw["model"]["options"]["x"], json!(7));
        assert_eq!(s.raw["model"]["name"], json!(false));
        // a section replaced by a scalar is an error, not a panic
        let a = args(&["infer", "--input", "i", "--output", "o", "--set", "runtime=1"]);
        assert!(apply_overrides(&mut s, &a.sets).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn allow_failures_flag() {
        assert!(!args(&["infer", "--input", "i", "--output", "o"]).allow_failures);
        assert!(args(&["infer", "--input", "i", "--output", "o", "--allow-failures"]).allow_failures);
    }
}
