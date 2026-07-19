import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '../..');

const githubAgentsDir = path.join(repoRoot, '.github', 'agents');
const githubSkillsDir = path.join(repoRoot, '.github', 'skills');

const agentsDestDir = path.join(repoRoot, '.agents', 'agents');
const skillsDestDir = path.join(repoRoot, '.agents', 'skills');

async function main() {
  try {
    // 1. Ensure .agents root directory exists
    await fs.mkdir(path.join(repoRoot, '.agents'), { recursive: true });

    // 2. Link MCP Config
    console.log('Linking MCP configuration...');
    const mcpSrcPath = path.join(repoRoot, '.mcp.json');
    const mcpDestPath = path.join(repoRoot, '.agents', 'mcp_config.json');
    try {
      await fs.lstat(mcpDestPath);
      console.log('  .agents/mcp_config.json already exists.');
    } catch {
      await fs.link(mcpSrcPath, mcpDestPath);
      console.log(
        `  Created hard link for MCP config: ${mcpSrcPath} -> ${mcpDestPath}`,
      );
    }

    // 3. Link Skills
    console.log('Linking skills...');
    try {
      await fs.lstat(skillsDestDir);
      console.log('  .agents/skills already exists.');
    } catch {
      const symlinkType = process.platform === 'win32' ? 'junction' : 'dir';
      await fs.symlink(githubSkillsDir, skillsDestDir, symlinkType);
      console.log(
        `  Created link from ${githubSkillsDir} -> ${skillsDestDir} (${symlinkType})`,
      );
    }

    // 4. Link Agents
    console.log('Linking agents...');
    await fs.mkdir(agentsDestDir, { recursive: true });

    const files = await fs.readdir(githubAgentsDir);
    const agentFiles = files.filter((f) => f.endsWith('.agent.md'));

    for (const file of agentFiles) {
      const agentName = file.replace('.agent.md', '');
      const srcPath = path.join(githubAgentsDir, file);
      const destAgentDir = path.join(agentsDestDir, agentName);
      const destPath = path.join(destAgentDir, 'agent.md');

      await fs.mkdir(destAgentDir, { recursive: true });

      try {
        await fs.lstat(destPath);
        // Already exists, skip
      } catch {
        // Create a hard link
        await fs.link(srcPath, destPath);
        console.log(
          `  Created hard link for agent: ${agentName} (${destPath})`,
        );
      }
    }
    console.log('Integration completed successfully!');
  } catch (error) {
    console.error('Error performing integration:', error);
    process.exit(1);
  }
}

main();
