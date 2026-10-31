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
    if !raw.is_object() {
        return Err(format!("config {path}: top level must be a mapping, got {raw}"));
    }
    for sec in SECTIONS {
        if raw.get(sec).map_or(true, |v| v.is_null()) {
            raw[sec] = json!({});
        }
    }
    check_sections(&raw).map_err(|e| format!("config {path}: {e}"))?;
    let dir = Path::new(path).canonicalize().map_err(|e| format!("config {path}: {e}"))?;
    raw["model"]["_config_dir"] = json!(dir.parent().unwrap().display().to_string());
    Ok(Settings { raw, path: path.to_string() })
}

const SECTIONS: [&str; 4] = ["model", "runtime", "scheduler", "io"];

/// Every section must be a mapping (also after `--set model=...`).
pub fn check_sections(raw: &Value) -> Result<(), String> {
    for sec in SECTIONS {
        if !raw[sec].is_object() {
            return Err(format!("section '{sec}' must be a mapping, got {}", raw[sec]));
        }
    }
    Ok(())
}

impl Settings {
    /// `a.b.c=value`; value is parsed as YAML (numbers, bools, lists, strings).
    pub fn set(&mut self, assignment: &str) -> Result<(), String> {
        let (key, val) = assignment.split_once('=').ok_or_else(|| format!("--set needs key=value: {assignment}"))?;
        let val: Value = serde_yaml::from_str(val).map_err(|e| format!("--set {key}: {e}"))?;
        self.set_value(key, val);
        Ok(())
    }

    /// `a.b.c` = `val` as is; missing or non-mapping parents become mappings.
    pub fn set_value(&mut self, key: &str, val: Value) {
        let mut cur = &mut self.raw;
        let parts: Vec<&str> = key.split('.').collect();
        for p in &parts[..parts.len() - 1] {
            if !cur.get(*p).map_or(false, |v| v.is_object()) {
                cur[*p] = json!({});
            }
            cur = cur.get_mut(*p).unwrap();
        }
        cur[*parts.last().unwrap()] = val;
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
                .map(|x| {
                    x.as_u64()
                        .filter(|&b| b > 0)
                        .map(|b| b as usize)
                        .ok_or("scheduler.batch_sizes entries must be positive integers")
                })
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

    /// String-valued scheduler key: None if unset, error if not a string.
    fn sched_str(&self, key: &str) -> Result<Option<&str>, String> {
        match self.sched(key) {
            Value::Null => Ok(None),
            Value::String(s) => Ok(Some(s)),
            v => Err(format!("scheduler.{key} must be a string, got {v}")),
        }
    }

    pub fn batching(&self) -> Result<Batching, String> {
        match self.sched_str("batching")?.unwrap_or("static") {
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
        let dim = |x: &Value| x.as_u64().filter(|&d| d > 0).map(|d| d as usize);
        list.iter()
            .map(|p| match p.as_array().map(|a| a.as_slice()) {
                Some([h, w]) => match (dim(h), dim(w)) {
                    (Some(h), Some(w)) => Ok((h, w)),
                    _ => Err(format!("bad shape {p} in scheduler.shapes (h, w must be positive integers)")),
                },
                _ => Err(format!("bad shape {p} in scheduler.shapes (want [h, w])")),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }

    pub fn unknown_shape(&self) -> Result<UnknownShape, String> {
        match self.sched_str("unknown_shape")?.unwrap_or("reject") {
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
            let n = |x: &str| x.trim().parse::<u64>().ok().filter(|&d| d > 0).ok_or_else(|| format!("bad shape {p}"));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn load_text(name: &str, text: &str) -> Result<Settings, String> {
        let dir = std::env::temp_dir().join(format!("clarus_config_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, text).unwrap();
        load(path.to_str().unwrap())
    }

    fn sched(name: &str, body: &str) -> Settings {
        load_text(name, &format!("scheduler:\n{body}\n")).unwrap()
    }

    #[test]
    fn root_and_sections_must_be_mappings() {
        assert!(load_text("empty.yaml", "").is_ok());
        assert!(load_text("nullsec.yaml", "model:\n").is_ok());
        for (name, text) in [("list.yaml", "[]"), ("scalar.yaml", "3"), ("model.yaml", "model: 1"),
                             ("io.yaml", "io: [1, 2]")] {
            let e = load_text(name, text).err().unwrap_or_else(|| panic!("{text} accepted"));
            assert!(e.contains("mapping"), "{e}");
        }
        let mut s = load_text("ok.yaml", "model: {}").unwrap();
        s.set("runtime=1").unwrap();
        assert!(check_sections(&s.raw).is_err());
    }

    #[test]
    fn shapes_must_be_positive_integers() {
        assert_eq!(sched("s1.yaml", "  shapes: [[720, 1280]]").shapes().unwrap(), Some(vec![(720, 1280)]));
        assert_eq!(sched("s2.yaml", "  shapes: auto").shapes().unwrap(), None);
        for body in ["[[oops, 4]]", "[[0, 4]]", "[[4, -1]]", "[[4.5, 4]]", "[[4]]", "[4, 4]"] {
            assert!(sched("s3.yaml", &format!("  shapes: {body}")).shapes().is_err(), "{body} accepted");
        }
        assert!(parse_shapes("720x1280").is_ok());
        assert!(parse_shapes("0x1280").is_err());
        assert!(parse_shapes("ax1").is_err());
    }

    #[test]
    fn enum_keys_reject_wrong_types_and_values() {
        let s = sched("e1.yaml", "  batching: dynamic\n  unknown_shape: lazy");
        assert_eq!(s.batching().unwrap(), Batching::Dynamic);
        assert_eq!(s.unknown_shape().unwrap(), UnknownShape::Lazy);
        let s = sched("e2.yaml", "  inflight: 2");
        assert_eq!(s.batching().unwrap(), Batching::Static);
        assert_eq!(s.unknown_shape().unwrap(), UnknownShape::Reject);
        let s = sched("e3.yaml", "  batching: 123\n  unknown_shape: false");
        assert!(s.batching().unwrap_err().contains("scheduler.batching"));
        assert!(s.unknown_shape().unwrap_err().contains("scheduler.unknown_shape"));
        let s = sched("e4.yaml", "  batching: sometimes\n  unknown_shape: maybe");
        assert!(s.batching().unwrap_err().contains("scheduler.batching"));
        assert!(s.unknown_shape().unwrap_err().contains("scheduler.unknown_shape"));
        let s = sched("e5.yaml", "  batch_sizes: [1, x]\n  inflight: two");
        assert!(s.batch_sizes().unwrap_err().contains("scheduler.batch_sizes"));
        assert!(s.usize_or("scheduler", "inflight", 3).unwrap_err().contains("scheduler.inflight"));
    }
}
