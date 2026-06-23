# Mistakes & Lessons Learned

This file documents mistakes made during development of this project.
**Refer to this before writing any code.**

---

## 1. Truncated requirements.txt → Incomplete pyproject.toml

**What happened:** When migrating to uv, `requirements.txt` was only 51 lines. The file appeared complete but was truncated. The resulting `pyproject.toml` was missing ~80 packages (whisper, soundfile, pyneuphonic, fer, etc.), causing `ModuleNotFoundError` crashes at runtime.

**Rule:** Always verify file completeness before using it as source of truth. Cross-reference with actual running imports (`grep -rh "^import\|^from" api/`), not just the requirements file.

---

## 2. Pinning transitive dependencies instead of direct ones

**What happened:** Copied every single package from `requirements.txt` into `pyproject.toml` as pinned dependencies — including transitive deps like `protobuf`, `certifi`, `numpy`, `ml-dtypes`. This caused unsolvable dependency conflicts because:
- `mediapipe==0.10.21` needs `protobuf<5`
- `tensorflow==2.20.0` needs `protobuf>=5`
- `pyneuphonic==1.8.8` needs `certifi>=2025.6.15`
- etc.

**Rule:** Only declare **direct** dependencies in `pyproject.toml` — packages your code actually imports. Let the package manager (uv/pip) resolve transitive deps. Run `grep -rh "^import\|^from" api/` to find real direct imports.

---

## 3. Overriding vulnerable packages instead of removing them

**What happened:** When `pnpm audit` found vulnerabilities in `node-fetch` (via `face-api.js`) and `postcss` (via Next.js), the response was to add version overrides — which masks the problem rather than fixing it.

**Critique from user:** "no why the fuck are you overriding vuln packages?????"

**Rule:** When a vulnerability is found, fix the root cause:
1. Remove the dependency if it's unused (face-api.js wasn't imported anywhere)
2. Upgrade the direct dependency that pulls in the vulnerable transitive dep
3. Only use overrides as a last resort when the upstream hasn't released a fix yet, and document why

---

## 4. Proposing Tailwind v4 without checking breaking changes

**What happened:** Upgraded `tailwindcss` to v4.3.1 without knowing that v4 moved the PostCSS plugin to a separate `@tailwindcss/postcss` package and broke the existing `postcss.config.js`. This crashed the dev server immediately.

**Rule:** Before upgrading a major version of any package, check the migration guide. If the project has existing config files for that package, read them first and verify compatibility. When in doubt, stay on the latest minor of the current major.

---

## 5. Suggesting wrong Next.js version

**What happened:** Told the user the latest Next.js was 15.1.6 when 16.2.9 was actually available. Then updated to 15.5.19, still not the latest.

**Rule:** Don't state version numbers from memory. Use `pnpm outdated` or check npm/the package's GitHub releases to confirm the actual latest version before recommending it.

---

## 6. Using dead/abandoned packages (fer)

**What happened:** Added `fer==22.5.1` to dependencies. `fer` is an unmaintained package with a cascade of broken deps:
- Needs `moviepy<2` (old API `moviepy.editor` removed in v2)
- Needs `pkg_resources` (removed from setuptools 82+)
- Needs `tensorflow`
- All of these conflicted with other packages

Spent multiple iterations trying to patch/shim around it before the correct decision was made: drop it and use `deepface` which was already in the codebase and does the same thing.

**Rule:** Before adding a package, check:
1. When was it last updated? (>2 years = red flag)
2. Are its dependencies modern and maintained?
3. Is there already a better alternative in the codebase?
The answer was already there — `emotion_detection.py` used `deepface` all along.

---

## 7. Writing files manually instead of using CLI tools

**What happened:** When asked how to migrate to uv, the response was to manually create `pyproject.toml` with hand-typed content. The user corrected this — `uv init` already does this correctly.

**Critique from user:** "nope, never make files manually"

**Rule:** Always use the canonical CLI tool for project scaffolding:
- `uv init` for Python/uv projects
- `pnpm init` for Node projects
- `npx create-next-app` for Next.js
Never hand-write boilerplate that a tool generates correctly.

---

## 8. Wrong pnpm config key

**What happened:** Tried to set `enable-pre-post-scripts=true` in `.npmrc` and `pnpm config set enable-pre-post-scripts true` — neither worked. The correct approach was `pnpm approve-builds <package>`.

**Rule:** Verify pnpm config key names against the actual pnpm docs before writing them. Use `pnpm approve-builds` for build script approval, not config flags.

---

## 9. Using deprecated `"pnpm"` field in package.json for overrides

**What happened:** Added a `"pnpm": { "overrides": {...} }` block to `package.json`. pnpm warned: *"The 'pnpm' field in package.json is no longer read by pnpm"*. This silently did nothing.

**Rule:** pnpm settings now live in `pnpm-workspace.yaml` or `.npmrc`. Don't use the `package.json` `"pnpm"` field — it's ignored in modern pnpm versions.

---

## 10. Null ref not guarded before DOM access

**What happened:** In `app/camera/page.tsx`, `videoRef.current.getBoundingClientRect()` was called without checking if the ref was mounted, causing a `TypeError: Cannot read properties of null`.

**Rule:** Always null-check refs before accessing DOM properties, especially inside async callbacks or interval functions where the component may have unmounted:
```ts
const video = videoRef.current;
const canvas = canvasRef.current;
if (!video || !canvas) return;
```

---

## 11. CSS variables in Framer Motion animations

**What happened:** `hsl(var(--primary) / 0.5)` and `rgba(var(--primary), 0.5)` were used as animation values in Framer Motion. These are not animatable — Framer Motion requires resolved color values, not CSS variable references.

**Rule:** Never use CSS custom properties (`var(--*)`) as Framer Motion `animate`/`whileHover`/`whileTap` color values. Resolve them to actual hex/rgb/hsl values first. If the color must respond to theme changes, handle it via a JS variable resolved at render time, not a CSS variable string.

---

## 12. Missing direct dependency (date-fns)

**What happened:** `date-fns` was used in `app/recordings/page.tsx` but was never in `package.json`. It worked previously (likely as a transitive dep of something else) but broke after the dependency cleanup.

**Rule:** Always explicitly declare packages your code imports directly. Don't rely on transitive deps for direct imports — they can disappear when the parent package updates.

---

## 13. Setting `ignore-scripts=false` in .npmrc

**What happened:** Set `ignore-scripts=false` in `.npmrc` to allow `sharp` to run its build script. This globally enables install scripts for every package, which is the primary vector for npm supply chain attacks (malicious `postinstall` scripts exfiltrating secrets, installing backdoors, etc.).

**Rule:** Always set `ignore-scripts=true` as the default. Then use `allowed-scripts=pkg1,pkg2` to explicitly allowlist only the packages that genuinely need native build steps (e.g. `sharp`, `unrs-resolver`). Every entry on that allowlist should be a well-known, audited package with a clear reason for needing a build script.

---

## Summary Checklist (run through before every coding session)

- [ ] Read existing code before writing new code
- [ ] Use CLI tools for scaffolding, never hand-write boilerplate
- [ ] Only pin direct dependencies, not transitive ones
- [ ] Check package health (last release, open issues) before adding it
- [ ] Verify version numbers from live sources, not memory
- [ ] Check migration guides before major version upgrades
- [ ] Guard all ref accesses with null checks in async contexts
- [ ] Never use CSS variables in Framer Motion animation values
- [ ] Fix vulnerabilities at the root, don't mask with overrides
- [ ] Cross-reference imports in code vs declared dependencies
