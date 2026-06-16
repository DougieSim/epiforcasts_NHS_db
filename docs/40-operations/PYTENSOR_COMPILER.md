# PyTensor / PyMC: C++ compiler (faster sampling)

## What the warning means

If you see:

```text
g++ not detected! PyTensor will be unable to compile C-implementations and will default to Python.
Performance may be severely degraded.
```

PyMC’s backend (**PyTensor**) is using **slow pure-Python** implementations because it cannot find a **C++ toolchain** (`g++` on MinGW-style installs, or MSVC `cl.exe` when using the Visual Studio toolchain).

**Fixing this is the main way to speed up** gradient evaluation and NUTS steps on CPU.

> **Note on uv:** this project uses **uv**, which installs Python packages from
> PyPI only — it does **not** bundle a C/C++ toolchain the way the old Pixi
> setup did (via conda-forge `cxx-compiler`). Install a compiler at the
> **system level** using one of the resolutions below.

---

## Resolution A (Windows, recommended): Microsoft C++ Build Tools

1. Install **[Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)** (free).
2. In the installer, select **“Desktop development with C++”** (includes MSVC, Windows SDK).
3. Open a **“x64 Native Tools Command Prompt”** or restart the terminal so `cl.exe` is on `PATH`.
4. Re-run inference (`uv run epiforcasts inference --fast`) — the `g++` warning should disappear and draws per second should improve.

PyTensor works with **either** MSVC `cl.exe` or a MinGW `g++` on `PATH`; on
Windows the MSVC Build Tools are the most reliable system-wide option.

---

## Resolution B (Linux / macOS)

- **Linux:** `sudo apt install build-essential` (Debian/Ubuntu) or your distro’s `gcc` / `g++` package.
- **macOS:** `xcode-select --install` for Apple’s command-line tools.

After installing, re-run inference; the warning should be gone.

---

## Verify

From the project directory:

```bash
uv run python -c "import shutil; print('g++:', shutil.which('g++')); print('cl: ', shutil.which('cl'))"
```

You want at least one of `g++` or `cl` to be non-`None` before running PyMC.

Then run a short fit; the PyTensor warning about `g++` should not appear, and step times (e.g. `s/draw`) should drop versus the pure-Python fallback.

---

## If you must suppress the warning only (not recommended)

This **does not** restore speed — it only hides the message. Prefer installing a compiler (above).

---

## Assumptions and limits

- **Cloud / CI:** Use a Docker image or build-agent image that includes `build-essential` (Linux) or MSVC + SDK (Windows) on `PATH` alongside the uv-managed Python environment.
- **Apple Silicon:** Use native arm64 Python and compilers; avoid Rosetta-only mixed toolchains for fewer surprises.
