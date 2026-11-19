# Workstation setup

**Finish setup before the lab.** Follow the setup pages in the lab guide in order. Use this file for installer links and help when a command fails. Use one project folder and run the lab commands from that folder.

## Install Git, uv and an editor

| System | Terminal | Notes |
|---|---|---|
| Windows x64 | PowerShell | Use a short local path such as `C:/ai53/phase1`. Keep the project outside OneDrive or a network share. |
| macOS, Apple Silicon | Terminal with zsh or bash | Use the command-line tools in the lab guide. If you already use Homebrew, `brew install git uv` installs both tools. |
| Linux, x86_64 or ARM64 | bash | Use your distribution's package manager to install Git and curl. |
| WSL2 | bash inside WSL | Install Git, uv and Python inside WSL. Keep the project in the Linux filesystem and open it with VS Code Remote - WSL. |

Keep several GB free for the packages, data and models. If PyTorch wheels are unavailable for your Intel Mac or Windows ARM setup, use a supported Linux machine.

On Windows, use [Git for Windows](https://git-scm.com/downloads/win) and the [uv installer](https://docs.astral.sh/uv/getting-started/installation/) if WinGet is unavailable. If you choose VS Code, [download it](https://code.visualstudio.com/download) or run `winget install --id Microsoft.VisualStudioCode -e` in PowerShell. Reopen your terminal and editor after installation.

## Use the project's Python

- `pyproject.toml` lists the package requirements.
- `uv.lock` records the exact versions for Python 3.12, including CPU-only PyTorch and torchvision.
- `.venv/` holds the installed packages for this project. Git ignores it.

Choose the project's Python in your editor. In VS Code, open **Python: Select Interpreter** in the Command Palette and select `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on macOS, Linux and WSL. Use the same interpreter and project folder in other editors. Use `uv run --locked` in the terminal.

## Connect GitHub

A Git remote is required for the project. Use your own GitHub account.

1. [Create a GitHub account](https://docs.github.com/en/account-and-profile/how-tos/account-management/creating-an-account-on-github) if you need one and verify your email.
2. [Create an empty repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository). Leave the README, license and `.gitignore` options off.
3. Copy the repository's HTTPS URL. Section E of the lab guide shows how to connect it as `origin`.

If `origin` already points to a different repository, run `git remote set-url origin YOUR-REPOSITORY-URL` with your URL and check with `git remote -v`. During the first push in Milestone 0, follow the sign-in prompt. If Git asks for a password or the sign-in window fails to open, use the GitHub CLI steps below.

Install [GitHub CLI](https://cli.github.com/), then run `gh auth login`. Choose **GitHub.com**, choose **HTTPS**, and answer **Yes** when asked to authenticate Git. Complete the browser sign-in, then retry the push. Keep credentials out of project files and remote URLs. [GitHub's HTTPS credential setup](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git)

## Check the included dataset

Check that `dataset-archives/` inside the project contains `train.zip`, `validation.zip` and `test.zip`. The three files take about 180 MB. In Milestone 2, choose the included files or download the same dataset. Both commands verify and extract the images into `data/raw/`. Git ignores the archives and the raw images.

## Troubleshooting

| Problem | What to check |
|---|---|
| `uv` or `git` is missing | Reopen your editor and terminal. In PowerShell use `Get-Command git,uv` or in bash/zsh use `command -v git uv` to check whether the commands are on PATH. |
| Missing Python imports | Check that the interpreter path shown by the environment check is inside this project's `.venv/`. |
| No PyTorch wheel | Check that your OS and CPU match a supported setup above and that you are using Python 3.12. |
| DLL error when importing PyTorch on Windows | Install the [Microsoft Visual C++ x64 runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist). |
| Proxy or certificate error | Check your network's proxy and certificate settings. If you are working offline, you need both the data archives and a prepared dependency cache. |
| `No module named src` | From the project folder, follow the guide's commands using `uv run --locked python -m src.MODULE`, with the module name in place of `MODULE`. |
| PowerShell blocks environment activation | Run the commands through `uv run --locked`. |
| MLflow page does not open in Milestone 6 | Check that the server is running in terminal 2. If port 5000 is taken, change the port in the server command and `tracking.uri` in `params.yaml`. Commit the setting before training and open the matching address in your browser. |
| Windows console shows `UnicodeEncodeError` | For tools that report this error, set `$env:PYTHONUTF8="1"` in PowerShell before running uv again. |
| `git push` fails | Check the URL with `git remote -v`, confirm you made a commit and follow the GitHub sign-in steps above. |
| A dataset archive fails its shape or checksum check | Replace the damaged or wrong archive. Get the exact files listed in `DATASET_SOURCE.md`. |

The supplied lockfile was tested on Windows x64. Verify the setup on any other OS you use.

[uv installation](https://docs.astral.sh/uv/getting-started/installation/) · [uv and PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/) · [Python in VS Code](https://code.visualstudio.com/docs/languages/python/)
