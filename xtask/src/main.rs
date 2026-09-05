use std::process::Command;

fn regenerate_docs()
{
    let modules_output = Command::new("cargo")
        .args([
            "modules",
            "dependencies",
            "--lib",
            "--no-fns",
            "--no-uses",
            "--no-externs",
            "--no-sysroot",
        ])
        .current_dir("core")
        .output()
        .expect("failed to run cargo-modules");

    std::fs::write("core/dependencies.dot", &modules_output.stdout)
        .expect("failed to write dependencies.dot");

    Command::new("dot")
        .args(["-Tpng", "dependencies.dot", "-o", "dependencies.png"])
        .current_dir("core")
        .status()
        .expect("failed to run dot");

    Command::new("cargo")
        .args(["doc"])
        .current_dir("core")
        .status()
        .expect("failed to run cargo doc");

    println!("Docs and dependency graph regenerated.");
}

fn main()
{
    regenerate_docs();
}
