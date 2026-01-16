use std::process::{Child, Command};
use std::sync::Mutex;

// Global reference to the Python sidecar process
static PYTHON_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

fn start_python_sidecar() -> Result<Child, std::io::Error> {
    // Determine the path to the Python sidecar
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    // Try multiple paths for Python script (handles both dev and prod)
    let possible_paths = [
        exe_dir.join("python").join("main.py"), // Prod: next to exe
        exe_dir
            .join("..")
            .join("..")
            .join("..")
            .join("python")
            .join("main.py"), // Dev: src-tauri/target/debug -> project root
        exe_dir
            .join("..")
            .join("..")
            .join("..")
            .join("..")
            .join("python")
            .join("main.py"), // Dev macOS: deeper nesting
        std::env::current_dir()
            .unwrap()
            .join("python")
            .join("main.py"), // Current working directory
    ];

    let python_script = possible_paths
        .iter()
        .find(|p| p.exists())
        .cloned()
        .unwrap_or_else(|| exe_dir.join("python").join("main.py"));

    println!("[ScreenSafe] Python script path: {:?}", python_script);

    // Try different Python executables
    let python_commands = ["python", "python3", "py"];

    // Get the python directory for working dir
    let python_dir = python_script.parent().unwrap_or(&exe_dir);

    for python_cmd in &python_commands {
        #[cfg(windows)]
        let result = {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            Command::new(python_cmd)
                .arg(&python_script)
                .arg("--port")
                .arg("9876")
                .current_dir(python_dir)
                .creation_flags(CREATE_NO_WINDOW)
                .spawn()
        };

        #[cfg(not(windows))]
        let result = Command::new(python_cmd)
            .arg(&python_script)
            .arg("--port")
            .arg("9876")
            .current_dir(python_dir)
            .spawn();

        if let Ok(child) = result {
            println!(
                "[ScreenSafe] Started Python sidecar with {} (PID: {})",
                python_cmd,
                child.id()
            );
            return Ok(child);
        }
    }

    // If none worked, return the last error
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Could not find Python executable",
    ))
}

fn stop_python_sidecar() {
    if let Ok(mut guard) = PYTHON_PROCESS.lock() {
        if let Some(mut child) = guard.take() {
            println!("[ScreenSafe] Stopping Python sidecar (PID: {})", child.id());
            // Move wait to a separate thread to avoid blocking main thread
            std::thread::spawn(move || {
                let _ = child.kill();
                let _ = child.wait();
            });
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Start Python sidecar before Tauri app
    match start_python_sidecar() {
        Ok(child) => {
            if let Ok(mut guard) = PYTHON_PROCESS.lock() {
                *guard = Some(child);
            }
        }
        Err(e) => {
            eprintln!(
                "[ScreenSafe] Warning: Failed to start Python sidecar: {}",
                e
            );
            eprintln!("[ScreenSafe] OCR features may not be available");
        }
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![greet])
        .on_window_event(|_window, event| {
            // Clean up Python process when app closes
            if let tauri::WindowEvent::Destroyed = event {
                stop_python_sidecar();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    // Ensure cleanup on normal exit
    stop_python_sidecar();
}
