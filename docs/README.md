# MorphoCLIP Docs

Public documentation site for MorphoCLIP, built with [Nextra 4](https://nextra.site) + Next.js 16.2 + Bun.

**Live site:** <https://morphoclip.suxrobgm.net>

## Development

```bash
bun install       # install dependencies
bun run dev       # start dev server at http://localhost:4000
bun run build     # build static site to out/
```

## Deployment

```bash
cp .env.example .env   # fill in your VPS credentials
bun run build          # build static site
bun run deploy         # compress, upload, and extract on VPS
```

See [.env.example](.env.example) for required variables.

## Content

All documentation lives in `content/` as `.mdx` files:

```text
_internal/                   # Internal docs (not published)
content/
  index.mdx                  # Landing page
  glossary.mdx               # Term definitions
  team.mdx                   # Team members
  getting-started/           # Installation, quick start
  pipeline/                  # Training pipeline, feature extraction, text encoder
  dataset/                   # CPJUMP1 overview, splits, compression
  background/                # Project proposal, literature review
  baselines/                 # CellCLIP, benchmark guides
```

Navigation is defined in `_meta.ts` files alongside content.

## Adding a page

1. Create a `.mdx` file in the appropriate `content/` subdirectory
2. Add an entry to the `_meta.ts` file in that directory
3. Run `bun run dev` to preview

## Nginx

See [morphoclip.conf](morphoclip.conf) for the nginx config with SSL setup instructions.
