/**
 * Deploy the static docs site to a VPS via SSH.
 * Compresses the build output into a tarball, uploads via scp, and extracts on the server.
 *
 * Usage: bun run deploy
 *
 * Requires a .env file with SSH_HOST, SSH_USER, SSH_KEY_PATH, DEPLOY_PATH.
 * See .env.example for reference.
 */

import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { $ } from "bun";
import dotenv from "dotenv";

const ROOT = resolve(import.meta.dirname, "..");
const OUT_DIR = resolve(ROOT, "out");
const ARCHIVE_NAME = "out.tar.gz";

dotenv.config({ path: resolve(ROOT, ".env") });

const host = process.env.SSH_HOST;
const port = process.env.SSH_PORT || "22";
const user = process.env.SSH_USER;
const keyPath = process.env.SSH_KEY_PATH;
const deployPath = process.env.DEPLOY_PATH;

if (!host || !user || !deployPath) {
  console.error("Missing required .env variables: SSH_HOST, SSH_USER, DEPLOY_PATH");
  process.exit(1);
}

if (!existsSync(OUT_DIR)) {
  console.error(`Build output not found at ${OUT_DIR}. Run 'bun run build' first.`);
  process.exit(1);
}

const sshArgs = ["-p", port, ...(keyPath ? ["-i", keyPath] : [])];
const ssh = ["ssh", ...sshArgs];
const scp = ["scp", "-P", port, ...(keyPath ? ["-i", keyPath] : [])];
const remote = `${user}@${host}`;

// 1. Compress (use cd + relative path to avoid Windows C: issue with tar)
console.log("Compressing build output...");
await $`cd ${ROOT} && tar -czf ${ARCHIVE_NAME} -C out .`.throws(true);

// 2. Upload
console.log(`Uploading to ${remote}...`);
await $`cd ${ROOT} && ${scp} ${ARCHIVE_NAME} ${remote}:/tmp/morphoclip-docs.tar.gz`.throws(true);

// 3. Extract on server
console.log(`Extracting to ${deployPath}...`);
const extractCmd = [
  `mkdir -p ${deployPath}`,
  `rm -rf ${deployPath}`,
  `mkdir -p ${deployPath}`,
  `tar -xzf /tmp/morphoclip-docs.tar.gz -C ${deployPath}`,
  `rm /tmp/morphoclip-docs.tar.gz`,
].join(" && ");

await $`${ssh} ${remote} ${extractCmd}`.throws(true);

// 4. Cleanup local archive
await $`cd ${ROOT} && rm ${ARCHIVE_NAME}`.quiet();

console.log("Deploy complete.");
