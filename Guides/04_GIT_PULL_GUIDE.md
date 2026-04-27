# Git Pull Guide - Getting the Latest Course Materials

This guide shows you how to update your local copy of the course repository.

## Before Each Session

Open your terminal, navigate to your course folder, and pull the latest changes:

```bash
# cd path/to/Machine_Learning_Course_Unitec
git pull origin main
```

If successful, you will see a list of updated files.

## Common Issue: Local Changes Blocking Pull

If you get an error like `Your local changes would be overwritten by merge`, it means you have modified files that conflict with the update.

### Option 1: Save Your Changes, Then Pull

This keeps your local edits safe:

```bash
git stash
git pull origin main
git stash pop
```

- `git stash` saves your changes temporarily
- `git pull origin main` gets the latest files
- `git stash pop` re-applies your saved changes on top

### Option 2: Discard Your Changes and Pull

Use this if you don't need to keep your local edits:

```bash
git checkout -- .
git pull origin main
```

## Clean Start (Hard Reset)

If things are messy and you want your folder to exactly match the remote repository, use a hard reset. This **deletes all local changes** and gives you a fresh copy:

```bash
git fetch origin main
git reset --hard origin/main
```

### If Hard Reset Fails (e.g. "Operation timed out")

This can happen if OneDrive or another sync service is locking files. Try:

```bash
rm -rf .git/logs
git fetch origin main
git reset --hard origin/main
```

## Remove Extra Files

After a reset, you may have leftover files that are not part of the repository. To clean them up:

```bash
git clean -fd
```

**Warning:** This removes all untracked files and folders. Make sure you don't have personal files in the repo folder that you want to keep.

## Quick Reference

| What you want | Command |
|---|---|
| Get latest updates | `git pull origin main` |
| Save changes, pull, restore changes | `git stash` then `git pull origin main` then `git stash pop` |
| Discard changes and pull | `git checkout -- .` then `git pull origin main` |
| Full clean start | `git fetch origin main` then `git reset --hard origin/main` |
| Remove extra files | `git clean -fd` |

## Tips

- Always pull before each class session to get the latest notebooks and materials
- If you want to keep your own notes or modified notebooks, save them outside the repo folder
- Never run `git push` on this repository — it is read-only for students

---

[← Previous: GitHub Setup Guide](03_GITHUB_SETUP_GUIDE.md) | [Index](README.md) | [Next: Workflow Guide →](05_WORKFLOW_GUIDE.md)
