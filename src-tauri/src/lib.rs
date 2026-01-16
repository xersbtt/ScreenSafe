use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

// Global reference to the sidecar process
static SIDECAR_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

fn get_python_dir() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    // Check multiple possible locations
    let possible_paths = [
        exe_dir.join("python"),
        exe_dir.join("..").join("..").join("..").join("python"),
        exe_dir
            .join("..")
            .join("..")
            .join("..")
            .join("..")
            .join("python"),
        std::env::current_dir().unwrap().join("python"),
    ];

    possible_paths
        .iter()
        .find(|p| p.exists())
        .cloned()
        .unwrap_or_else(|| exe_dir.join("python"))
}

fn get_venv_dir() -> PathBuf {
    get_python_dir().join("venv")
}

fn get_setup_marker() -> PathBuf {
    get_venv_dir().join(".setup_complete")
}

#[tauri::command]
fn check_setup_complete() -> bool {
    let marker = get_setup_marker();
    let exists = marker.exists();
    println!(
        "[ScreenSafe] Setup marker check: {:?} exists={}",
        marker, exists
    );
    exists
}

#[tauri::command]
async fn run_setup(app_handle: tauri::AppHandle) -> Result<(), String> {
    use tauri::Emitter;

    let python_dir = get_python_dir();
    let venv_dir = get_venv_dir();
    let requirements = python_dir.join("requirements.txt");

    println!("[ScreenSafe] Starting setup...");
    println!("[ScreenSafe] Python dir: {:?}", python_dir);
    println!("[ScreenSafe] Venv dir: {:?}", venv_dir);

    // Emit initial progress
    let _ = app_handle.emit(
        "setup-progress",
        serde_json::json!({
            "progress": 5,
            "message": "Creating virtual environment...",
            "detail": "Setting up isolated Python environment"
        }),
    );

    // Find Python executable
    let python_cmds = ["python", "python3", "py"];
    let python_cmd = python_cmds
        .iter()
        .find(|cmd| {
            Command::new(cmd)
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        })
        .ok_or("Python not found. Please install Python 3.10+ from python.org")?;

    println!("[ScreenSafe] Using Python: {}", python_cmd);

    // Create virtual environment if it doesn't exist
    if !venv_dir.exists() {
        let _ = app_handle.emit(
            "setup-progress",
            serde_json::json!({
                "progress": 10,
                "message": "Creating virtual environment...",
                "detail": format!("python -m venv {:?}", venv_dir)
            }),
        );

        let output = Command::new(python_cmd)
            .args(["-m", "venv"])
            .arg(&venv_dir)
            .current_dir(&python_dir)
            .output()
            .map_err(|e| format!("Failed to create venv: {}", e))?;

        if !output.status.success() {
            return Err(format!(
                "Failed to create venv: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }

    // Get pip executable path
    #[cfg(windows)]
    let pip_path = venv_dir.join("Scripts").join("pip.exe");
    #[cfg(not(windows))]
    let pip_path = venv_dir.join("bin").join("pip");

    // Upgrade pip first
    let _ = app_handle.emit(
        "setup-progress",
        serde_json::json!({
            "progress": 15,
            "message": "Upgrading pip...",
            "detail": "Ensuring latest package installer"
        }),
    );

    #[cfg(windows)]
    let python_venv = venv_dir.join("Scripts").join("python.exe");
    #[cfg(not(windows))]
    let python_venv = venv_dir.join("bin").join("python");

    let _ = Command::new(&python_venv)
        .args(["-m", "pip", "install", "--upgrade", "pip"])
        .current_dir(&python_dir)
        .output();

    // Install requirements with progress tracking
    let _ = app_handle.emit(
        "setup-progress",
        serde_json::json!({
            "progress": 20,
            "message": "Detecting GPU...",
            "detail": "Checking for NVIDIA CUDA support"
        }),
    );

    // Check for CUDA availability
    let has_cuda = Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    println!("[ScreenSafe] CUDA available: {}", has_cuda);

    if has_cuda {
        // Install PyTorch with CUDA support from pytorch.org
        let _ = app_handle.emit(
            "setup-progress",
            serde_json::json!({
                "progress": 25,
                "message": "Installing PyTorch with CUDA...",
                "detail": "This enables GPU acceleration"
            }),
        );

        println!("[ScreenSafe] Installing CUDA-enabled PyTorch");
        let torch_output = Command::new(&pip_path)
            .args([
                "install",
                "torch",
                "torchvision",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ])
            .current_dir(&python_dir)
            .output();

        if let Ok(output) = torch_output {
            if output.status.success() {
                println!("[ScreenSafe] CUDA PyTorch installed successfully");
            } else {
                println!(
                    "[ScreenSafe] CUDA PyTorch failed, falling back to CPU: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    } else {
        println!("[ScreenSafe] No CUDA detected, will use CPU PyTorch");
    }

    let _ = app_handle.emit(
        "setup-progress",
        serde_json::json!({
            "progress": 40,
            "message": "Installing dependencies...",
            "detail": "This may take several minutes"
        }),
    );

    println!(
        "[ScreenSafe] Installing requirements from {:?}",
        requirements
    );

    // Run pip install with output parsing
    let mut child = Command::new(&pip_path)
        .args(["install", "-r"])
        .arg(&requirements)
        .current_dir(&python_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to run pip install: {}", e))?;

    // Track progress based on output
    let stdout = child.stdout.take().expect("Failed to capture stdout");
    let reader = BufReader::new(stdout);

    let mut progress: f64 = 40.0;
    let progress_increment: f64 = 50.0 / 20.0; // Assume ~20 packages

    for line in reader.lines() {
        if let Ok(line) = line {
            println!("[pip] {}", line);

            // Parse package installation progress
            if line.contains("Collecting")
                || line.contains("Installing")
                || line.contains("Successfully installed")
            {
                progress = (progress + progress_increment).min(90.0_f64);

                let package_name = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("package")
                    .split("==")
                    .next()
                    .unwrap_or("package");

                let _ = app_handle.emit(
                    "setup-progress",
                    serde_json::json!({
                        "progress": progress as i32,
                        "message": "Installing dependencies...",
                        "detail": package_name
                    }),
                );
            }
        }
    }

    let status = child
        .wait()
        .map_err(|e| format!("pip install failed: {}", e))?;

    if !status.success() {
        return Err("Failed to install dependencies. Check your internet connection.".to_string());
    }

    // Create setup marker
    let _ = app_handle.emit(
        "setup-progress",
        serde_json::json!({
            "progress": 95,
            "message": "Finalizing setup...",
            "detail": "Almost done!"
        }),
    );

    std::fs::write(&get_setup_marker(), "setup complete")
        .map_err(|e| format!("Failed to write setup marker: {}", e))?;

    let _ = app_handle.emit(
        "setup-progress",
        serde_json::json!({
            "progress": 100,
            "message": "Setup complete!",
            "detail": "Starting ScreenSafe..."
        }),
    );

    println!("[ScreenSafe] Setup complete!");
    Ok(())
}

fn start_sidecar() -> Result<Child, std::io::Error> {
    let python_dir = get_python_dir();
    let venv_dir = get_venv_dir();
    let main_py = python_dir.join("main.py");

    // Check if we have venv (production) or need system Python (dev)
    #[cfg(windows)]
    let python_venv = venv_dir.join("Scripts").join("python.exe");
    #[cfg(not(windows))]
    let python_venv = venv_dir.join("bin").join("python");

    let (python_cmd, use_venv) = if python_venv.exists() {
        (python_venv.to_string_lossy().to_string(), true)
    } else {
        // Dev mode - use system Python
        ("python".to_string(), false)
    };

    println!(
        "[ScreenSafe] Starting sidecar with Python: {} (venv={})",
        python_cmd, use_venv
    );
    println!("[ScreenSafe] Main script: {:?}", main_py);

    #[cfg(windows)]
    let result = {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        Command::new(&python_cmd)
            .arg(&main_py)
            .arg("--port")
            .arg("9876")
            .current_dir(&python_dir)
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
    };

    #[cfg(not(windows))]
    let result = Command::new(&python_cmd)
        .arg(&main_py)
        .arg("--port")
        .arg("9876")
        .current_dir(&python_dir)
        .spawn();

    match result {
        Ok(child) => {
            println!("[ScreenSafe] Started sidecar (PID: {})", child.id());
            Ok(child)
        }
        Err(e) => {
            eprintln!("[ScreenSafe] Failed to start sidecar: {}", e);
            Err(e)
        }
    }
}

fn stop_sidecar() {
    if let Ok(mut guard) = SIDECAR_PROCESS.lock() {
        if let Some(mut child) = guard.take() {
            println!("[ScreenSafe] Stopping sidecar (PID: {})", child.id());
            std::thread::spawn(move || {
                let _ = child.kill();
                let _ = child.wait();
            });
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Note: We don't start sidecar here anymore - the frontend will do it after setup

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            check_setup_complete,
            run_setup,
            start_sidecar_command
        ])
        .on_window_event(|_window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                stop_sidecar();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    stop_sidecar();
}

#[tauri::command]
fn start_sidecar_command() -> Result<(), String> {
    match start_sidecar() {
        Ok(child) => {
            if let Ok(mut guard) = SIDECAR_PROCESS.lock() {
                *guard = Some(child);
            }
            Ok(())
        }
        Err(e) => Err(format!("Failed to start sidecar: {}", e)),
    }
}
