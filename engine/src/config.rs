//! Engine configuration: one YAML file (see configs/*.yaml) + command-line overrides.
//!
//! Sections:
//!   model:     forwarded to the Python worker (which model, checkpoint, arch, options)
//!   runtime:   forwarded to the Python worker (device, dtype, op backends, graphs)
//!   scheduler: used here (batch sizes, batching mode, fixed shapes, in-flight depth)
//!   io:        used here (IO threads, PNG compression)
//!
//! Every key can also be overridden with `--set section.key=value` (value parsed as YAML).

use serde_json::{json, Value};
use std::path::Path;

use crate::engine::Batching;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnknownShape {
    /// refuse images whose shape is not in `scheduler.shapes`
    Reject,
    /// prepare a graph for the new shape on first use (may tune kernels mid-run)
    Lazy,
}

pub struct Settings {
    pub raw: Value,
    pub path: String,
}

pub fn load(path: &str) -> Result<Settings, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("config {path}: {e}"))?;
    let mut raw: Value = serde_yaml::from_str(&text).map_err(|e| format!("config {path}: {e}"))?;
    if raw.is_null() {
        raw = json!({});
    }
    for sec in ["model", "runtime", "scheduler", "io"] {
        if raw.get(sec).map_or(true, |v| v.is_null()) {
            raw[sec] = json!({});
        }
    }
    let dir = Path::new(path).canonicalize().map_err(|e| format!("config {path}: {e}"))?;
    raw["model"]["_config_dir"] = json!(dir.parent().unwrap().display().to_string());
    Ok(Settings { raw, path: path.to_string() })
}

impl Settings {
    /// `a.b.c=value`; value is parsed as YAML (numbers, bools, lists, strings).
    pub fn set(&mut self, assignment: &str) -> Result<(), String> {
        let (key, val) = assignment.split_once('=').ok_or_else(|| format!("--set needs key=value: {assignment}"))?;
        let val: Value = serde_yaml::from_str(val).map_err(|e| format!("--set {key}: {e}"))?;
        let mut cur = &mut self.raw;
        let parts: Vec<&str> = key.split('.').collect();
        for p in &parts[..parts.len() - 1] {
            if !cur.get(*p).map_or(false, |v| v.is_object()) {
                cur[*p] = json!({});
            }
            cur = cur.get_mut(*p).unwrap();
        }
        cur[*parts.last().unwrap()] = val;
        Ok(())
    }

    fn sched(&self, key: &str) -> &Value {
        &self.raw["scheduler"][key]
    }

    pub fn batch_sizes(&self) -> Result<Vec<usize>, String> {
        let v = self.sched("batch_sizes");
        let mut sizes: Vec<usize> = if v.is_null() {
            vec![1, 2]
        } else {
            v.as_array()
                .ok_or("scheduler.batch_sizes must be a list")?
                .iter()
                .map(|x| x.as_u64().filter(|&b| b > 0).map(|b| b as usize).ok_or("batch sizes must be > 0"))
                .collect::<Result<_, _>>()?
        };
        sizes.sort_unstable();
        sizes.dedup();
        if sizes.is_empty() {
            return Err("scheduler.batch_sizes is empty".into());
        }
        Ok(sizes)
    }

    pub fn usize_or(&self, sec: &str, key: &str, default: usize) -> Result<usize, String> {
        match &self.raw[sec][key] {
            Value::Null => Ok(default),
            v => v.as_u64().map(|x| x as usize).ok_or_else(|| format!("{sec}.{key} must be an integer")),
        }
    }

    pub fn batching(&self) -> Result<Batching, String> {
        match self.sched("batching").as_str().unwrap_or("static") {
            "static" => Ok(Batching::Static),
            "dynamic" => Ok(Batching::Dynamic),
            o => Err(format!("scheduler.batching must be static|dynamic, got {o}")),
        }
    }

    /// None = "auto" (use the shapes found in the input set).
    pub fn shapes(&self) -> Result<Option<Vec<(usize, usize)>>, String> {
        let v = self.sched("shapes");
        if v.is_null() || v.as_str() == Some("auto") {
            return Ok(None);
        }
        let list = v.as_array().ok_or("scheduler.shapes must be 'auto' or a list of [h, w]")?;
        list.iter()
            .map(|p| match p.as_array().map(|a| a.as_slice()) {
                Some([h, w]) => Ok((h.as_u64().unwrap_or(0) as usize, w.as_u64().unwrap_or(0) as usize)),
                _ => Err(format!("bad shape {p} in scheduler.shapes (want [h, w])")),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }

    pub fn unknown_shape(&self) -> Result<UnknownShape, String> {
        match self.sched("unknown_shape").as_str().unwrap_or("reject") {
            "reject" => Ok(UnknownShape::Reject),
            "lazy" => Ok(UnknownShape::Lazy),
            o => Err(format!("scheduler.unknown_shape must be reject|lazy, got {o}")),
        }
    }
}

/// `--shapes 720x1280,480x640`
pub fn parse_shapes(s: &str) -> Result<Value, String> {
    let v: Result<Vec<Value>, String> = s
        .split(',')
        .map(|p| {
            let (h, w) = p.split_once('x').ok_or_else(|| format!("bad shape {p} (want HxW)"))?;
            let n = |x: &str| x.trim().parse::<u64>().map_err(|_| format!("bad shape {p}"));
            Ok(json!([n(h)?, n(w)?]))
        })
        .collect();
    Ok(Value::Array(v?))
}

/// `--device 1` keeps the device type of the config, `--device npu:0` replaces it.
pub fn device_spec(current: &Value, arg: &str) -> String {
    if arg.contains(':') || arg.parse::<u64>().is_err() {
        return arg.to_string();
    }
    let kind = current.as_str().unwrap_or("cuda:0").split(':').next().unwrap_or("cuda").to_string();
    format!("{kind}:{arg}")
}
