#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 08:58:58 2025

a helper script to keep all meg_utils submodules in sync

@author: simon
"""

import os
import subprocess
import textwrap

def find_git_dirs(base_dir, target_folder):
    matches = []
    for root, dirs, files in os.walk(base_dir):
        if target_folder in dirs:
            full_path = os.path.join(root, target_folder)
            if os.path.exists(os.path.join(full_path, ".git")):
                matches.append(full_path)
    return matches

def pull_git_repo(repo_path):
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "pull"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def summarize_git_pull_output_line(output):
    lines = output.strip().split('\n')
    summary = {
        "commit_range": lines[0] if lines else "No update info",
        "files_changed": 0,
        "insertions": 0,
        "deletions": 0,
        "created": [],
        "modified": [],
    }

    for line in lines:
        if "file changed" in line:
            parts = line.split(',')
            for part in parts:
                part = part.strip()
                if "file changed" in part or "files changed" in part:
                    summary["files_changed"] = int(part.split()[0])
                elif "insertion" in part:
                    summary["insertions"] = int(part.split()[0])
                elif "deletion" in part:
                    summary["deletions"] = int(part.split()[0])
        elif "create mode" in line:
            summary["created"].append(line.split()[-1])
        elif "|" in line:
            summary["modified"].append(line.split("|")[0].strip())

    created = f"created: {', '.join(summary['created'])}" if summary["created"] else ""
    modified = f"modified: {', '.join(summary['modified'])}" if summary["modified"] else ""
    parts = [
        f"{summary['files_changed']} files",
        f"{summary['insertions']} inserts",
        f"{summary['deletions']} dels",
        created,
        modified
    ]
    return " | ".join(part for part in parts if part)


if __name__ == "__main__":
    base_folder = "~/Nextcloud/ZI"
    base_directory = os.path.expanduser(base_folder)
    target = "meg_utils"
    print(f"Searching for '{target}' repositories in {base_directory}...\n")

    repos = find_git_dirs(base_directory, target)
    if not repos:
        raise Exception("No repositories found.")

    for repo in repos:
        print(f"Found {repo}")

    input('Continue? [enter]')

    print()
    for repo in repos:
        print(f"git pull ..{repo.replace(base_directory, '')} -> ", end=' ')
        success, output = pull_git_repo(repo)
        summary = textwrap.indent(summarize_git_pull_output_line(output), '   | ')
        if success:
            if "Already up to date" in output:
                print(f"ok\n")
            else:
                print(f"ok\n{summary}\n")
        else:
            print(f"ERROR\n{textwrap.indent(output, '    | ')}\n")
