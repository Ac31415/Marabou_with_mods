Ac31415: how to do Git Large File Storage for a repo?

GitHub Copilot: Git Large File Storage (LFS) is a Git extension that replaces large files with text pointers inside Git, while storing the file contents on a remote server. Here's how to set it up for your repository:

## Installation

First, install Git LFS on your Mac:

```bash
brew install git-lfs
```

## Initialize Git LFS

Navigate to your repository and initialize Git LFS:

```bash
cd /Users/wen-chungcheng/Marabou_with_mods
git lfs install
```

## Track Large Files

Specify which file types or specific files should be tracked by LFS. Common examples:

```bash
# Track all files with specific extensions
git lfs track "*.zip"
git lfs track "*.tar.gz"
git lfs track "*.bin"
git lfs track "*.exe"
git lfs track "*.dll"
git lfs track "*.so"
git lfs track "*.dylib"

# Track files larger than a certain size (e.g., 100MB)
git lfs track "large-dataset.csv"

# Track all files in a specific directory
git lfs track "models/**"
```

## Commit the .gitattributes File

Git LFS creates a .gitattributes file to track your LFS patterns:

```bash
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

## Add and Commit Large Files

Now add your large files normally:

```bash
git add your-large-file.zip
git commit -m "Add large file via LFS"
git push origin main
```

## Verify LFS Tracking

Check which files are being tracked by LFS:

```bash
git lfs ls-files
```

View LFS tracking patterns:

```bash
git lfs track
```

## For Existing Large Files

If you have large files already committed to Git history, you'll need to migrate them:

```bash
# Migrate specific file types from history
git lfs migrate import --include="*.zip"

# Force push the rewritten history
git push --force-with-lease origin main
```

## Clone LFS Repository

When others clone your repository, they need to:

```bash
git clone <repository-url>
cd <repository-name>
git lfs pull
```

## Useful LFS Commands

```bash
# Check LFS status
git lfs status

# Pull LFS files
git lfs pull

# Push LFS files
git lfs push origin main

# Show LFS environment info
git lfs env
```

Git LFS is particularly useful for repositories containing binary assets, datasets, models, or any files larger than 100MB (GitHub's file size limit).

Ac31415: can not upload new objects to public fork

GitHub Copilot: This error typically occurs when trying to push to a public fork that you don't have write permissions to, or when there are Git LFS quota/permission issues. Here are the most common solutions:

## 1. Check Repository Permissions

Make sure you're pushing to your own fork, not the original repository:

```bash
# Check your remote URLs
git remote -v

# Should show your fork, not the original repo
# If it shows the original repo, update it:
git remote set-url origin https://github.com/YOUR_USERNAME/Marabou_with_mods.git
```

## 2. Git LFS Quota Issues

If you're using Git LFS, you might have hit quota limits:

```bash
# Check LFS quota status
git lfs env

# Try pushing without LFS first
git push origin main --no-verify
```

## 3. Authentication Issues

Ensure you're properly authenticated:

```bash
# For HTTPS (use personal access token)
git config --global credential.helper osxkeychain

# Or switch to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/Marabou_with_mods.git
```

## 4. Large File Issues

If you have large files causing problems:

```bash
# Check for large files
find . -size +100M -type f

# Remove large files from Git history if needed
git filter-branch --tree-filter 'rm -f path/to/large/file' HEAD

# Or use BFG Repo Cleaner
brew install bfg
bfg --strip-blobs-bigger-than 100M
```

## 5. Force Push (Use Carefully)

If you need to overwrite the remote branch:

```bash
# Only if you're sure about overwriting
git push --force-with-lease origin main
```

## 6. Create New Fork

If permissions are completely broken:

1. Delete your current fork on GitHub
2. Create a new fork of the original repository
3. Clone the new fork:

```bash
git clone https://github.com/YOUR_USERNAME/Marabou_with_mods.git
cd Marabou_with_mods
```

What specific error message are you seeing when trying to upload? This would help provide a more targeted solution.

Ac31415: batch response: @Ac31415 can not upload new objects to public fork Ac31415/Marabou_with_mods

GitHub Copilot: Based on your Git error, this is a GitHub-specific issue with Git LFS (Large File Storage) on public forks. The error "can not upload new objects to public fork" typically occurs when:

## Root Cause
GitHub doesn't allow Git LFS uploads to public forks for security and quota management reasons. LFS objects can only be uploaded to repositories you own or have admin access to.

## Solutions

### Option 1: Remove Git LFS (Recommended for forks)
If the large files aren't essential, remove them from LFS tracking:

```bash
# See what files are tracked by LFS
git lfs ls-files

# Untrack specific file types
git lfs untrack "*.zip"
git lfs untrack "*.bin"
# ... untrack other large file types

# Remove LFS objects and add files normally
git lfs migrate export --include="*"

# Commit the changes
git add .gitattributes
git commit -m "Remove Git LFS tracking"
git push origin master
```

### Option 2: Use Git LFS with Your Own Repository
Instead of forking, create your own repository:

```bash
# Create a new repo on GitHub first, then:
git remote add upstream https://github.com/NeuralNetworkVerification/Marabou.git
git remote set-url origin https://github.com/Ac31415/YOUR_NEW_REPO_NAME.git
git push origin master
```

### Option 3: Fork to a Private Repository
If you need LFS, make your fork private:
1. Go to your fork settings on GitHub
2. Change repository visibility to Private
3. Try pushing again

### Option 4: Remove Large Files Entirely
Find and remove large files from your repository:

```bash
# Find files larger than 100MB
find . -size +100M -type f -not -path "./.git/*"

# Remove them from Git history
git filter-branch --index-filter 'git rm --cached --ignore-unmatch path/to/large/file' HEAD

# Force push the cleaned history
git push --force-with-lease origin master
```

### Quick Fix for Immediate Push
If you need to push immediately, try:

```bash
# Skip LFS for this push
git push origin master --no-verify

# Or push without LFS objects
GIT_LFS_SKIP_SMUDGE=1 git push origin master
```

**Recommendation**: For public forks of open-source projects like Marabou, Option 1 (removing Git LFS) is usually the best approach since most development work doesn't require the large binary files that LFS typically manages.

Ac31415: I've already removed the new_build folder:

> git push origin master:master
remote: error: Trace: fe659d7f26d0c4d9f9d9d3f9308dc5df010bb700b9cab2de8eca3727cb169e65        
remote: error: See https://gh.io/lfs for more information.        
remote: error: File new_build/libMarabouHelperTest.a is 118.07 MB; this exceeds GitHub's file size limit of 100.00 MB        
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.        
To https://github.com/Ac31415/Marabou_with_mods.git
 ! [remote rejected]   master -> master (pre-receive hook declined)
error: failed to push some refs to 'https://github.com/Ac31415/Marabou_with_mods.git'

what happened

GitHub Copilot: The issue is that even though you've removed the `new_build` folder from your current working directory, the large file `libMarabouHelperTest.a` (118.07 MB) still exists in your **Git history**. GitHub checks the entire commit history, not just the current state of files.

Here's what happened and how to fix it:

## The Problem
When you committed the `new_build` folder earlier, Git stored the large file in the repository's history. Even after deleting it, that large file remains in previous commits, and GitHub rejects the entire push.

## Solution: Remove from Git History

You need to completely remove the large file from Git history:

### Option 1: Using git filter-branch (Built-in Git command)

````bash
# Remove the large file from entire Git history
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch new_build/libMarabouHelperTest.a' \
--prune-empty --tag-name-filter cat -- --all

# Clean up
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
````

### Option 2: Using BFG Repo-Cleaner (Faster and easier)

````bash
# Install BFG Repo-Cleaner
brew install bfg

# Remove all files larger than 100MB from history
bfg --strip-blobs-bigger-than 100M

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive
````

### Option 3: Remove entire new_build directory from history

````bash
# Remove the entire new_build folder from Git history
git filter-branch --force --index-filter \
'git rm -rf --cached --ignore-unmatch new_build' \
--prune-empty --tag-name-filter cat -- --all

# Clean up
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
````

## After Cleaning History

Once you've removed the large file from history:

````bash
# Force push the cleaned history
git push --force-with-lease origin master
````

## Prevention: Add to .gitignore

To prevent this in the future, add build directories to your .gitignore:

````bash
# Add to .gitignore file
echo "new_build/" >> .gitignore
echo "build/" >> .gitignore
echo "*.a" >> .gitignore
echo "*.o" >> .gitignore

git add .gitignore
git commit -m "Add build artifacts to .gitignore"
````

**Warning**: The `--force-with-lease` flag will rewrite Git history. If others have cloned your repository, they'll need to re-clone after you force push.

The key takeaway is that removing files from your working directory doesn't remove them from Git history - you need to explicitly clean the history to remove large files completely.

Ac31415: (Webots-Sim) wen-chungcheng@Wen-Chungs-MacBook-Pro Marabou_with_mods % git push --force-with-lease origin master
To https://github.com/Ac31415/Marabou_with_mods.git
 ! [rejected]          master -> master (stale info)
error: failed to push some refs to 'https://github.com/Ac31415/Marabou_with_mods.git'

GitHub Copilot: The `--force-with-lease` command failed because your local repository has "stale info" - meaning your local copy doesn't have the latest information about the remote repository state. This is a safety feature to prevent accidentally overwriting changes.

Here's how to fix this:

## Solution 1: Fetch and Try Again (Safest)

```bash
# Fetch the latest remote information
git fetch origin

# Try the force push again
git push --force-with-lease origin master
```

## Solution 2: If You're Sure About Overwriting

If you're certain you want to overwrite the remote branch with your cleaned history:

```bash
# Regular force push (less safe, but will work)
git push --force origin master
```

## Solution 3: Check What's Different

To see what's causing the conflict:

```bash
# See the difference between local and remote
git log origin/master..master --oneline

# See remote commits you don't have locally
git log master..origin/master --oneline
```

## What Likely Happened

When you cleaned your Git history using `git filter-branch` or BFG, you created a completely new commit history. The `--force-with-lease` flag detected that your local branch has diverged significantly from the remote branch and rejected the push as a safety measure.

## Recommended Approach

```bash
# 1. Fetch latest remote state
git fetch origin

# 2. Check if there are any important remote changes
git log master..origin/master --oneline

# 3. If no important changes, force push
git push --force-with-lease origin master

# 4. If that still fails, use regular force push
git push --force origin master
```

**Important**: After force pushing, anyone who has cloned your repository will need to re-clone it or reset their local copy, as the Git history has been completely rewritten.

The `--force-with-lease` is safer than `--force` because it prevents you from accidentally overwriting work that others might have pushed while you were cleaning your history.