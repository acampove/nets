use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};

fn regenerate_docs() {
    let modules_output = Command::new("cargo")
        .args([
            "modules",
            "dependencies",
            "--lib",
            "--no-fns",
            "--no-uses",
            "--no-externs",
            "--no-sysroot",
            "-p",
            "nets",
        ])
        .current_dir("core")
        .output()
        .expect("failed to run cargo-modules");

    fs::create_dir_all("core/docs").expect("failed to create core/docs");

    let mut dot_child = Command::new("dot")
        .args(["-Tpng", "-o", "docs/dependencies.png"])
        .current_dir("core")
        .stdin(Stdio::piped())
        .spawn()
        .expect("failed to start dot");

    dot_child
        .stdin
        .take()
        .expect("failed to open dot stdin")
        .write_all(&modules_output.stdout)
        .expect("failed to write to dot stdin");

    dot_child.wait().expect("dot process failed");

    Command::new("cargo")
        .args(["doc"])
        .current_dir("core")
        .status()
        .expect("failed to run cargo doc");
}

fn main()
{
    regenerate_docs();
}
