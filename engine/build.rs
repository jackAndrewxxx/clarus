//! PyO3 links libpython (interpreter chosen by PYO3_PYTHON, see .cargo/config.toml).
//! This script adds an rpath to that libpython and bakes the interpreter's home and
//! site-packages as runtime defaults (overridable via CLARUS_PYTHONHOME / CLARUS_SITE /
//! CLARUS_PYTHONPATH).
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-changed=build.rs");
    let py = std::env::var("PYO3_PYTHON").expect("PYO3_PYTHON must point at the model's Python interpreter");
    let script = "import sys, sysconfig, site\n\
                  print(sysconfig.get_config_var('LIBDIR'))\n\
                  print(sys.base_prefix)\n\
                  print(site.getsitepackages()[0])\n";
    let out = Command::new(&py)
        .arg("-c")
        .arg(script)
        .output()
        .unwrap_or_else(|e| panic!("cannot run PYO3_PYTHON={py}: {e}"));
    assert!(out.status.success(), "PYO3_PYTHON failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let (libdir, home, site) = (lines[0], lines[1], lines[2]);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    println!("cargo:rustc-env=CLARUS_BUILD_PYTHONHOME={home}");
    println!("cargo:rustc-env=CLARUS_BUILD_SITE={site}");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let pydir = std::path::Path::new(&manifest).join("..").join("python");
    let pydir = pydir.canonicalize().unwrap_or(pydir);
    println!("cargo:rustc-env=CLARUS_BUILD_PYDIR={}", pydir.display());
    let cfg = std::path::Path::new(&manifest).join("..").join("configs").join("nafnet_gopro_width64.yaml");
    let cfg = cfg.canonicalize().unwrap_or(cfg);
    println!("cargo:rustc-env=CLARUS_BUILD_CONFIG={}", cfg.display());
}
