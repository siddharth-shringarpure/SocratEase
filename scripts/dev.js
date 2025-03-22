const { spawn } = require("child_process");
const path = require("path");

// Function to determine the Python executable path in the virtual environment
const getPythonPath = () => {
  if (process.platform === "win32") {
    return path.join(process.cwd(), ".venv", "Scripts", "python.exe");
  }
  return path.join(process.cwd(), ".venv", "bin", "python");
};

// Print Python path being used
console.log("Using Python executable:", getPythonPath());

// Start Flask server
const flaskProcess = spawn(getPythonPath(), ["api/index.py"], {
  stdio: ["inherit", "pipe", "pipe"], // Change to pipe for stdout and stderr
  env: {
    ...process.env,
    FLASK_DEBUG: "1",
    FLASK_RUN_PORT: "5328",
  },
});

// Log Flask output
flaskProcess.stdout.on("data", (data) => {
  console.log(`Flask stdout: ${data}`);
});

flaskProcess.stderr.on("data", (data) => {
  console.error(`Flask stderr: ${data}`);
});

// Start Next.js dev server
const nextProcess = spawn("npm", ["run", "next-dev"], {
  stdio: "inherit",
  shell: true,
});

// Handle process termination
const cleanup = () => {
  flaskProcess.kill();
  nextProcess.kill();
  process.exit();
};

process.on("SIGINT", cleanup);
process.on("SIGTERM", cleanup);
