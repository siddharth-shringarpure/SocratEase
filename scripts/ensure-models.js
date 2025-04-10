const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

function ensureModels() {
  console.log("Checking and downloading models if needed...");

  // Check and download frontend models
  const frontendModelsDir = path.join(process.cwd(), "public", "models");
  if (
    !fs.existsSync(frontendModelsDir) ||
    fs.readdirSync(frontendModelsDir).length === 0
  ) {
    console.log("Downloading frontend models...");
    const frontendResult = spawnSync("npm", ["run", "download-models"], {
      stdio: "inherit",
      shell: true,
    });
    if (frontendResult.status !== 0) {
      console.error("Failed to download frontend models");
      process.exit(1);
    }
  }

  // Check and download backend models
  const backendModelsDir = path.join(process.cwd(), "api", "models");
  if (
    !fs.existsSync(backendModelsDir) ||
    fs.readdirSync(backendModelsDir).length === 0
  ) {
    console.log("Downloading backend models...");
    const backendResult = spawnSync(
      "python3.10",
      ["scripts/download_backend_models.py"],
      {
        stdio: "inherit",
        shell: true,
      }
    );
    if (backendResult.status !== 0) {
      console.error("Failed to download backend models");
      process.exit(1);
    }
  }

  console.log("All required models are available.");
}

// If this script is run directly
if (require.main === module) {
  ensureModels();
}

module.exports = ensureModels;
