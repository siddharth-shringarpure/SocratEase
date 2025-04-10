const { spawn } = require("child_process");
const path = require("path");
const ensureModels = require("./ensure-models");

// Ensure models are downloaded before starting servers
ensureModels();

// Start Next.js dev server
const nextDev = spawn("npm", ["run", "next-dev"], {
  stdio: "inherit",
  shell: true,
});

// Start Flask dev server
const flaskDev = spawn("npm", ["run", "flask-dev"], {
  stdio: "inherit",
  shell: true,
  env: {
    ...process.env,
    FLASK_RUN_PORT: "5000",
  },
});

// Handle process termination
const cleanup = () => {
  nextDev.kill("SIGINT");
  flaskDev.kill("SIGINT");
  process.exit();
};

process.on("SIGINT", cleanup);
process.on("SIGTERM", cleanup);
