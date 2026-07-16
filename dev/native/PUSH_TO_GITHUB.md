# Pushing Easy Rigify to GitHub (for the crypto CI)

Goal: get the repo onto GitHub so the **build-crypto** Action can compile the
macOS + Linux binaries you can't build on Windows.

> ⚠️ **The repository MUST be PRIVATE.** It contains the crypto source
> (`dev/native/ercrypto.c` holds the masked key) and your build scripts.
> A public repo would hand out everything. The `.gitignore` already keeps the
> multi-GB datasets, backups, and the plaintext models (`dev/models_src/`) out.

---

## 0. Install Git (one time)
Git isn't installed yet. In PowerShell:
```powershell
winget install --id Git.Git -e
```
Close and reopen the terminal, then check:
```powershell
git --version
```

## 1. Initialize the repo
From `d:\rig_addon_pro`:
```powershell
git init -b main
git add .
```

## 2. SAFETY CHECK — before you commit, confirm nothing huge/secret is staged
```powershell
# Should print a SMALL number (a few hundred), NOT tens of thousands:
(git ls-files).Count
# These must print NOTHING (they're ignored):
git ls-files | Select-String "models_src|/Backup/|\.blend$|hand_dataset|face_dataset"
```
If those last commands print files, stop — the `.gitignore` isn't taking
effect (run `git rm -r --cached <that path>` and re-check).

## 3. Commit
```powershell
git commit -m "Easy Rigify: addon + native model-crypto + CI"
```

## 4. Create the PRIVATE repo on GitHub
1. Go to <https://github.com/new>
2. Name it (e.g. `easy-rigify`).
3. **Visibility: Private.** ← important
4. Do **NOT** add a README, .gitignore, or license (you already have files).
5. Click **Create repository**.

## 5. Connect and push
Copy the two lines GitHub shows under "…or push an existing repository",
they look like this (use YOUR username/repo):
```powershell
git remote add origin https://github.com/<you>/easy-rigify.git
git push -u origin main
```
(First push may ask you to sign in to GitHub in a browser popup.)

## 6. Let Actions write back the binaries (one time)
On GitHub: **Settings → Actions → General → Workflow permissions →**
select **"Read and write permissions" → Save.**
(The workflow needs this to commit the built binaries into `native/`.)

## 7. Run the build
The **build-crypto** workflow runs automatically on the first push (it sees
`dev/native/ercrypto.c`). Otherwise: **Actions tab → build-crypto → Run
workflow**. Watch it build on windows/ubuntu/macos. When the `commit` job
finishes it pushes the binaries into `native/`.

## 8. Pull the new binaries down
```powershell
git pull
```
`native/` now contains all three:
```
_ercrypto_win_amd64.pyd
_ercrypto_linux_x86_64.so
_ercrypto_macos_universal2.so
```
Now `blender --command extension build` produces a zip that works on every OS.

---

## Everyday flow after this
- Change code → `git add . ; git commit -m "…" ; git push`.
- Change the crypto (`ercrypto.c`) → push → CI rebuilds all 3 binaries and
  commits them back → `git pull`. Remember to run `dev/native/rotate_models.py`
  locally first if you changed the KEY (see README.md).
