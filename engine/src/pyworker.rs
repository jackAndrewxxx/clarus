//! Embedded CPython model worker, hosted through PyO3.
//!
//! Threading model: only the scheduler (main) thread ever attaches to the interpreter;
//! the IO threads are pure Rust. Each worker call attaches for its duration and
//! releases the GIL afterwards, so the hot path has no GIL contention.

use pyo3::prelude::*;
use pyo3::types::PyList;

pub struct Interpreter;

fn err_text(py: Python<'_>, e: PyErr, what: &str) -> String {
    let msg = format!("{what}: {e}");
    e.print(py); // traceback -> stderr
    msg
}

impl Interpreter {
    /// Start CPython with `home` as PYTHONHOME, add `site` (venv site-packages) and `pydir`
    /// (the clarus package root) to sys.path.
    pub fn start(home: &str, site: &str, pydir: &str) -> Result<Interpreter, String> {
        if std::env::var_os("PYTHONHOME").is_none() {
            std::env::set_var("PYTHONHOME", home);
        }
        Python::initialize(); // signal handling stays with the Rust process
        Python::attach(|py| -> PyResult<()> {
            let sys = py.import("sys")?;
            sys.setattr("argv", vec!["clarus"])?;
            py.import("site")?.call_method1("addsitedir", (site,))?;
            sys.getattr("path")?.cast_into::<PyList>()?.insert(0, pydir)?;
            Ok(())
        })
        .map_err(|e| Python::attach(|py| err_text(py, e, "configure sys.path")))?;
        Ok(Interpreter)
    }

    /// `module.factory(arg)` -> worker object
    pub fn create(&self, module: &str, factory: &str, arg: &str) -> Result<PyHandle, String> {
        Python::attach(|py| {
            py.import(module)
                .and_then(|m| m.call_method1(factory, (arg,)))
                .map(|obj| PyHandle { obj: obj.unbind() })
                .map_err(|e| err_text(py, e, &format!("{module}.{factory}")))
        })
    }
}

/// A Python object whose methods take one `str` and return `str` (JSON in, JSON out).
pub struct PyHandle {
    obj: Py<PyAny>,
}

impl PyHandle {
    pub fn call(&self, method: &str, arg: &str) -> Result<String, String> {
        Python::attach(|py| {
            self.obj
                .bind(py)
                .call_method1(method, (arg,))
                .and_then(|r| r.extract::<String>().map_err(PyErr::from))
                .map_err(|e| err_text(py, e, method))
        })
    }
}
